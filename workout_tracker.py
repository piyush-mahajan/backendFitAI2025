from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import os
from jose import jwt, JWTError
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from passlib.context import CryptContext
import logging

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

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(
    title="Workout Tracker API",
    description="API for processing workout prompts and tracking fitness data per user",
    version="1.0.0"
)

# Middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    body = await request.body()
    logger.info(f"Body: {body.decode('utf-8') if body else 'Empty'}")
    response = await call_next(request)
    return response

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

class StatsResponse(BaseModel):
    total_sets: int
    total_reps: int
    total_volume: float
    muscle_group_breakdown: Dict[str, float]
    daily: Dict[str, List[WorkoutData]]
    weekly: Dict[str, List[WorkoutData]]
    monthly: Dict[str, List] = None
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
        extracted = eval(response.choices[0].message.content)  # Use json.loads in production
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

# Authentication Endpoints
@app.post("/api/auth/register", response_model=UserProfile)
async def register(user: UserProfileCreate):
    """Register a new user"""
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
            "total_workouts": 0
        }
        result = users_collection.insert_one(user_data)
        logger.info(f"Registered user with ID: {user_id}, Mongo ID: {result.inserted_id}")
        return UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Failed to register user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and generate JWT token"""
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

# Protected API Endpoints
@app.post("/api/workout/log/", response_model=WorkoutData)
async def log_workout(workout: WorkoutPrompt, current_user: UserProfile = Depends(get_current_user)):
    """Log a new workout for the authenticated user"""
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
    
    try:
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
async def get_workout_history(current_user: UserProfile = Depends(get_current_user)):
    """Retrieve workout history for the authenticated user"""
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        logger.info(f"Found {len(workouts)} workouts for user_id: {current_user.user_id}")
        return [WorkoutData(**{**w, "id": str(w["id"])}) for w in workouts]
    except Exception as e:
        logger.error(f"Failed to fetch workout history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/workout/stats/", response_model=StatsResponse)
async def get_workout_stats(current_user: UserProfile = Depends(get_current_user)):
    """Get detailed workout statistics and summaries for the authenticated user"""
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
    """Create a new user profile (public endpoint for registration)"""
    return await register(profile)

@app.get("/api/user/profile/", response_model=UserProfile)
async def get_user_profile(current_user: UserProfile = Depends(get_current_user)):
    """Fetch profile data for the authenticated user"""
    try:
        logger.info(f"Fetched profile for user_id: {current_user.user_id}")
        return current_user
    except Exception as e:
        logger.error(f"Failed to fetch profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")