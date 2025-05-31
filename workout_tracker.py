from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import os
import razorpay
import json
import logging
import hmac
import hashlib
from jose import jwt, JWTError
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from passlib.context import CryptContext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI").replace("<db_password>", os.getenv("MONGO_PASSWORD"))
mongo_db_name = os.getenv("MONGO_DB_NAME", "workout_tracker")
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'), tls=True)
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB Atlas!")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise Exception(f"MongoDB connection failed: {str(e)}")

db = client[mongo_db_name]
users_collection = db["users"]
workouts_collection = db["workouts"]

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Razorpay setup (Test Mode)
try:
    razorpay_client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
    razorpay_client.set_app_details({"title": "Workout Tracker", "version": "1.0"})
    logger.info("Razorpay client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Razorpay client: {str(e)}")
    raise Exception(f"Razorpay initialization failed: {str(e)}")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(
    title="Workout Tracker API",
    description="API for processing workout prompts and tracking fitness data per user",
    version="1.0.0"
)

# Mount static files for serving the subscription frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Exercise database
EXERCISE_DB = {
    "bench press": {"type": "strength", "muscle_group": "chest"},
    "squat": {"type": "strength", "muscle_group": "legs"},
    "deadlift": {"type": "strength", "muscle_group": "back"},
    "pull up": {"type": "strength", "muscle_group": "back"},
    "push up": {"type": "bodyweight", "muscle_group": "chest"}
}

# Azure Open AI configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
ai_client = AzureOpenAI(
    api_key=AZURE_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=AZURE_ENDPOINT
)

# Pydantic models
class WorkoutPrompt(BaseModel):
    prompt: str

class WorkoutData(BaseModel):
    id: str
    user_id: str
    workout_name: str
    sets: int
    reps: int
    weight_kg: Optional[float] = None
    calories_burned: Optional[float] = None
    duration_minutes: Optional[float] = None
    timestamp: datetime
    muscle_group: Optional[str] = None

class UserProfileCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    created_at: datetime
    total_workouts: int = 0
    subscription_status: Optional[str] = "inactive"
    subscription_plan: Optional[str] = None
    subscription_order_id: Optional[str] = None
    subscription_end_date: Optional[datetime] = None

class StatsResponse(BaseModel):
    total_sets: int
    total_reps: int
    total_volume: float
    muscle_group_breakdown: Dict[str, float]
    daily: Dict[str, List[WorkoutData]]
    weekly: Dict[str, List[WorkoutData]]
    monthly: Dict[str, List[WorkoutData]]

class Token(BaseModel):
    access_token: str
    token_type: str

class SubscriptionRequest(BaseModel):
    plan_id: str

class SubscriptionResponse(BaseModel):
    order_id: str
    razorpay_key: str

# Helper functions
def estimate_calories(workout_name: str, sets: int, reps: int, weight_kg: float) -> float:
    return round(sets * reps * (weight_kg or 0) * 0.02, 1)

def calculate_volume(workout: dict) -> float:
    return workout["sets"] * workout["reps"] * (workout["weight_kg"] or 1)

def extract_workout_data(prompt: str) -> dict:
    try:
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": """You are a fitness assistant. Extract structured workout data from the user's prompt.
                    Return JSON with: workout_name, sets, reps, weight_kg (null if not specified), and muscle_group (if identifiable).
                    Match workout_name to common exercises where possible."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )
        extracted = eval(response.choices[0].message.content)
        workout_name = extracted.get("workout_name", "unknown").lower()
        muscle_group = extracted.get("muscle_group")
        if workout_name in EXERCISE_DB:
            muscle_group = EXERCISE_DB[workout_name]["muscle_group"]
        return {
            "workout_name": workout_name,
            "sets": extracted.get("sets", 1),
            "reps": extracted.get("reps", 1),
            "weight_kg": extracted.get("weight_kg"),
            "muscle_group": muscle_group
        }
    except Exception as e:
        logger.error(f"AI extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process prompt with AI: {str(e)}")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_receipt_id(user_id: str) -> str:
    # Use a short prefix, first 12 chars of user_id, and a 6-digit timestamp
    timestamp = str(int(datetime.utcnow().timestamp()))[-6:]
    return f"wt_{user_id[:12]}_{timestamp}"

# Authentication dependencies
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = users_collection.find_one({"user_id": user_id})
    if user is None:
        raise credentials_exception
    return UserProfile(**user)

async def get_premium_user(token: str = Depends(oauth2_scheme)):
    user = await get_current_user(token)
    if user.subscription_status != "active":
        raise HTTPException(status_code=403, detail="Premium subscription required")
    if user.subscription_end_date and user.subscription_end_date < datetime.utcnow():
        users_collection.update_one(
            {"user_id": user.user_id},
            {"$set": {"subscription_status": "inactive"}}
        )
        raise HTTPException(status_code=403, detail="Subscription expired")
    return user

# Authentication Endpoints
@app.post("/api/auth/register", response_model=UserProfile)
async def register(user: UserProfileCreate):
    try:
        if users_collection.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)
        user_data = {
            "user_id": user_id,
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_password,
            "height_cm": user.height_cm,
            "weight_kg": user.weight_kg,
            "age": user.age,
            "created_at": datetime.utcnow(),
            "total_workouts": 0,
            "subscription_status": "inactive",
            "subscription_plan": None,
            "subscription_order_id": None,
            "subscription_end_date": None
        }
        result = users_collection.insert_one(user_data)
        logger.info(f"Registered user with ID: {user_id}, Mongo ID: {result.inserted_id}")
        return UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Failed to register user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        logger.info(f"Login attempt with email: {form_data.username}")
        user = users_collection.find_one({"email": form_data.username})
        if not user:
            logger.warning(f"No user found with email: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_password(form_data.password, user["hashed_password"]):
            logger.warning(f"Password verification failed for email: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["user_id"]}, expires_delta=access_token_expires
        )
        logger.info(f"User logged in: {user['user_id']}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Failed to login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# Subscription Endpoints
@app.get("/api/subscriptions/plans/")
async def get_subscription_plans():
    try:
        plans = [
            {"name": "Basic Plan", "plan_id": "plan_basic", "amount": 199, "description": "Access to workout logging and basic stats"},
            {"name": "Pro Plan", "plan_id": "plan_pro", "amount": 499, "description": "Access to advanced analytics and recommendations"},
            {"name": "Elite Plan", "plan_id": "plan_elite", "amount": 999, "description": "Access to all features including virtual coaching"}
        ]
        logger.info("Fetched subscription plans")
        return plans
    except Exception as e:
        logger.error(f"Failed to fetch plans: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch plans: {str(e)}")

@app.post("/api/subscriptions/create/", response_model=SubscriptionResponse)
async def create_subscription(request: SubscriptionRequest, current_user: UserProfile = Depends(get_current_user)):
    try:
        if current_user.subscription_status == "active":
            logger.warning(f"User {current_user.user_id} already has an active subscription")
            raise HTTPException(status_code=400, detail="User already has an active subscription")

        logger.info(f"Starting subscription creation for user {current_user.user_id}")
        logger.info(f"Razorpay Key ID: {os.getenv('RAZORPAY_KEY_ID')}")
        logger.info(f"Razorpay Key Secret: {os.getenv('RAZORPAY_KEY_SECRET')[:4]}... (partial for security)")

        # Map plan_id to amount
        plan_amounts = {
            "plan_basic": 19900,  # ₹199 in paise
            "plan_pro": 49900,    # ₹499 in paise
            "plan_elite": 99900   # ₹999 in paise
        }
        plan_names = {
            "plan_basic": "Basic Plan",
            "plan_pro": "Pro Plan",
            "plan_elite": "Elite Plan"
        }
        amount_paise = plan_amounts.get(request.plan_id)
        plan_name = plan_names.get(request.plan_id)
        if not amount_paise or not plan_name:
            logger.error(f"Invalid plan_id: {request.plan_id}")
            raise HTTPException(status_code=400, detail="Invalid plan ID")

        # Generate a short receipt ID
        receipt_id = generate_receipt_id(current_user.user_id)
        logger.info(f"Generated receipt_id: {receipt_id}")

        # Create Razorpay Order
        order_data = {
            "amount": amount_paise,
            "currency": "INR",
            "receipt": receipt_id,
            "notes": {
                "name": current_user.username,
                "email": current_user.email,
                "contact": "9999999999",
                "source": "Workout Tracker App",
                "plan_id": request.plan_id
            }
        }

        try:
            order_response = razorpay_client.order.create(data=order_data)
            order_id = order_response['id']
            logger.info(f"Razorpay Order Created for user {current_user.user_id}! Order ID: {order_id}")
        except razorpay.errors.BadRequestError as e:
            logger.error(f"Razorpay API Error (Bad Request): {str(e)}")
            raise HTTPException(status_code=400, detail=f"Razorpay API Error (Bad Request): {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while creating order: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error while creating order: {str(e)}")

        # Update user with order ID and plan
        try:
            users_collection.update_one(
                {"user_id": current_user.user_id},
                {"$set": {
                    "subscription_order_id": order_id,
                    "subscription_status": "pending",
                    "subscription_plan": request.plan_id
                }}
            )
            logger.info(f"Updated user {current_user.user_id} with order ID: {order_id}")
        except Exception as e:
            logger.error(f"Failed to update user in database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to update user in database: {str(e)}")

        return SubscriptionResponse(
            order_id=order_id,
            razorpay_key=os.getenv("RAZORPAY_KEY_ID")
        )
    except Exception as e:
        logger.error(f"Failed to create subscription for user {current_user.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create subscription: {str(e)}")

@app.post("/api/webhooks/razorpay/")
async def razorpay_webhook(request: Request):
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        signature = request.headers.get("x-razorpay-signature")
        webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

        if not webhook_secret:
            logger.error("Webhook secret not set in environment variables")
            raise HTTPException(status_code=500, detail="Webhook secret not configured")

        # Verify webhook signature
        expected_signature = hmac.new(
            key=webhook_secret.encode('utf-8'),
            msg=body,
            digestmod=hashlib.sha256
        ).hexdigest()

        logger.info(f"Expected signature: {expected_signature}, Received signature: {signature}")
        if not hmac.compare_digest(expected_signature, signature):
            logger.error("Webhook signature verification failed")
            raise HTTPException(status_code=400, detail="Webhook signature verification failed")

        event = json.loads(body_str)
        logger.info(f"Webhook event received: {event['event']}")

        if event["event"] == "payment.authorized":
            payment_entity = event["payload"]["payment"]["entity"]
            order_id = payment_entity["order_id"]
            logger.info(f"Processing payment.authorized for order ID: {order_id}")

            user = users_collection.find_one({"subscription_order_id": order_id})
            if not user:
                logger.error(f"User not found for order ID: {order_id}")
                raise HTTPException(status_code=404, detail="User not found")

            # Update user subscription status
            plan_duration = {
                "plan_basic": 30,  # 30 days for Basic
                "plan_pro": 60,    # 60 days for Pro
                "plan_elite": 90   # 90 days for Elite
            }
            duration_days = plan_duration.get(user["subscription_plan"], 30)
            users_collection.update_one(
                {"subscription_order_id": order_id},
                {"$set": {
                    "subscription_status": "active",
                    "subscription_end_date": datetime.utcnow() + timedelta(days=duration_days)
                }}
            )
            logger.info(f"Updated subscription status to active for user: {user['user_id']}")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Webhook verification failed: {str(e)}")

# Protected API Endpoints
@app.post("/api/workout/log/", response_model=WorkoutData)
async def log_workout(workout: WorkoutPrompt, current_user: UserProfile = Depends(get_premium_user)):
    try:
        extracted_data = extract_workout_data(workout.prompt)
        
        workout_data = {
            "id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "workout_name": extracted_data["workout_name"],
            "sets": extracted_data["sets"],
            "reps": extracted_data["reps"],
            "weight_kg": extracted_data["weight_kg"],
            "calories_burned": estimate_calories(
                extracted_data["workout_name"],
                extracted_data["sets"],
                extracted_data["reps"],
                extracted_data["weight_kg"]
            ),
            "duration_minutes": None,
            "timestamp": datetime.utcnow(),
            "muscle_group": extracted_data["muscle_group"]
        }
        
        result = workouts_collection.insert_one(workout_data)
        logger.info(f"Workout inserted with ID: {result.inserted_id} for user: {current_user.user_id}")
        
        users_collection.update_one(
            {"user_id": current_user.user_id},
            {"$inc": {"total_workouts": 1}}
        )
        
        return WorkoutData(**workout_data)
    except Exception as e:
        logger.error(f"Failed to log workout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/workout/history/", response_model=List[WorkoutData])
async def get_workout_history(current_user: UserProfile = Depends(get_premium_user)):
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        logger.info(f"Found {len(workouts)} workouts for user_id: {current_user.user_id}")
        return [WorkoutData(**{**w, "id": str(w["id"])}) for w in workouts]
    except Exception as e:
        logger.error(f"Failed to fetch workout history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/workout/stats/", response_model=StatsResponse)
async def get_workout_stats(current_user: UserProfile = Depends(get_premium_user)):
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        if not workouts:
            return StatsResponse(
                total_sets=0,
                total_reps=0,
                total_volume=0,
                muscle_group_breakdown={},
                daily={},
                weekly={},
                monthly={}
            )
        
        total_sets = 0
        total_reps = 0
        total_volume = 0
        muscle_group_breakdown = {}
        daily_summary = {}
        weekly_summary = {}
        monthly_summary = {}
        
        for log in workouts:
            total_sets += log["sets"]
            total_reps += log["reps"]
            volume = calculate_volume(log)
            total_volume += volume
            
            muscle = log["muscle_group"] or "other"
            muscle_group_breakdown[muscle] = muscle_group_breakdown.get(muscle, 0) + volume
            
            date_str = log["timestamp"].strftime("%Y-%m-%d")
            if date_str not in daily_summary:
                daily_summary[date_str] = []
            daily_summary[date_str].append(WorkoutData(**{**log, "id": str(log["id"])}))
            
            if log["timestamp"] >= datetime.utcnow() - timedelta(days=7):
                week_str = log["timestamp"].strftime("%Y-%m-%d")
                if week_str not in weekly_summary:
                    weekly_summary[week_str] = []
                weekly_summary[week_str].append(WorkoutData(**{**log, "id": str(log["id"])}))
            
            if log["timestamp"] >= datetime.utcnow() - timedelta(days=30):
                month_str = log["timestamp"].strftime("%Y-%m-%d")
                if month_str not in monthly_summary:
                    monthly_summary[month_str] = []
                monthly_summary[month_str].append(WorkoutData(**{**log, "id": str(log["id"])}))
        
        return StatsResponse(
            total_sets=total_sets,
            total_reps=total_reps,
            total_volume=total_volume,
            muscle_group_breakdown=muscle_group_breakdown,
            daily=daily_summary,
            weekly=weekly_summary,
            monthly=monthly_summary
        )
    except Exception as e:
        logger.error(f"Failed to fetch stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/user/profile/", response_model=UserProfile)
async def create_or_update_user_profile(profile: UserProfileCreate):
    return await register(profile)

@app.get("/api/user/profile/", response_model=UserProfile)
async def get_user_profile(current_user: UserProfile = Depends(get_current_user)):
    try:
        logger.info(f"Fetched profile for user_id: {current_user.user_id}")
        return current_user
    except Exception as e:
        logger.error(f"Failed to fetch profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")