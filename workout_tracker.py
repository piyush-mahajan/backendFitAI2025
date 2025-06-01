from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import os
import redis.asyncio as redis
import logging
import json
from jose import jwt, JWTError
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError
from passlib.context import CryptContext
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment validation
required_env = ["MONGO_URI", "MONGO_PASSWORD", "JWT_SECRET_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_DEPLOYMENT_NAME"]
missing_env = [var for var in required_env if not os.getenv(var)]
if missing_env:
    raise Exception(f"Missing environment variables: {missing_env}")

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI").replace("<db_password>", os.getenv("MONGO_PASSWORD"))
mongo_db_name = os.getenv("MONGO_DB_NAME", "workout_tracker")
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    client.admin.command("ping")
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise Exception(f"MongoDB connection failed: {str(e)}")

db = client[mongo_db_name]
users_collection = db["users"]
workouts_collection = db["workouts"]

# Create indexes
try:
    users_collection.create_index("email", unique=True)
    users_collection.create_index("user_id")
    workouts_collection.create_index([("user_id", 1), ("timestamp", -1)])
    logger.info("MongoDB indexes created")
except DuplicateKeyError:
    logger.warning("Index creation skipped: duplicate key")
    try:
        users_collection.delete_many({"email": None})
        users_collection.create_index("email", unique=True)
    except Exception as e:
        logger.error(f"Failed to clean null emails: {str(e)}")
except Exception as e:
    logger.error(f"Failed to create indexes: {str(e)}")
    raise Exception(f"Index creation failed: {str(e)}")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing with argon2
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(
    title="Workout Tracker API",
    description="API for tracking workouts and fitness data",
    version="1.0.2"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rate limiting
@app.on_event("startup")
async def startup():
    try:
        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        await FastAPILimiter.init(redis_client)
        logger.info("Rate limiter initialized")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {str(e)}")
    # Migrate workouts_collection
    try:
        workouts = workouts_collection.find({"duration_minutes": None})
        for workout in workouts:
            duration = estimate_duration(
                workout["workout_name"], workout["sets"], workout["reps"]
            )
            workouts_collection.update_one(
                {"_id": workout["_id"]},
                {"$set": {"duration_minutes": duration}}
            )
        logger.info("Migrated workouts with null duration_minutes")
    except Exception as e:
        logger.error(f"Failed to migrate workouts: {str(e)}")

# Exercise registry
EXERCISE_REGISTRY = {
    "bench press": {"type": "strength", "muscle_group": "chest", "met": 6.0},
    "squat": {"type": "strength", "muscle_group": "legs", "met": 5.0},
    "deadlift": {"type": "strength", "muscle_group": "back", "met": 6.0},
    "pull up": {"type": "bodyweight", "muscle_group": "back", "met": 3.8},
    "push up": {"type": "bodyweight", "muscle_group": "chest", "met": 3.8},
    "overhead press": {"type": "strength", "muscle_group": "shoulders", "met": 5.0},
    "incline dumbbell press": {"type": "strength", "muscle_group": "chest", "met": 5.0}
}

# Azure Open AI with custom HTTP client
http_client = httpx.Client(
    timeout=30.0,
    follow_redirects=True,
    transport=httpx.HTTPTransport(retries=3)
)
ai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    http_client=http_client
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
    calories_burned: float
    duration_minutes: Optional[float] = None
    timestamp: datetime
    muscle_group: Optional[str] = None

class WorkoutRecommendation(BaseModel):
    workout_name: str
    sets: int
    reps: int
    weight_kg: Optional[float] = None
    muscle_group: Optional[str] = None
    description: str
    motivational_message: str
    difficulty_level: str = Field(..., pattern="^(Beginner|Intermediate|Advanced)$")
    estimated_duration_minutes: float
    equipment_needed: List[str]

class UserProfileCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[float] = None
    fitness_goals: Optional[str] = None

class UserProfileUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[float] = None
    fitness_goals: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[float] = None
    fitness_goals: Optional[str] = None
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

class CalorieCalculation(BaseModel):
    bmr: float
    daily_calories: float
    maintenance_calories: float
    weight_loss_calories: float
    weight_gain_calories: float
    guidelines: str

class ExerciseResponse(BaseModel):
    name: str
    type: str
    muscle_group: str
    met: float

# Helper functions
def estimate_calories(workout_name: str, sets: int, reps: int, weight_kg: float, duration_minutes: float) -> float:
    normalized_name = normalize_exercise_name(workout_name)
    met = EXERCISE_REGISTRY.get(normalized_name, {"met": 3.5}).get("met", 3.5)
    duration_hours = duration_minutes / 60.0
    if weight_kg is None:
        weight_kg = 70.0
    return round(met * weight_kg * duration_hours, 1)

def estimate_duration(workout_name: str, sets: int, reps: int) -> float:
    normalized_name = normalize_exercise_name(workout_name)
    exercise_type = EXERCISE_REGISTRY.get(normalized_name, {"type": "strength"}).get("type", "strength")
    if exercise_type == "bodyweight":
        time_per_set = (reps * 2) / 60.0
        rest_time = 0.5 * sets
    else:
        time_per_set = (reps * 3) / 60.0
        rest_time = 1.0 * sets
    return round((time_per_set * sets) + rest_time, 1)

def calculate_volume(workout: dict) -> float:
    return workout["sets"] * workout["reps"] * (workout["weight_kg"] or 1)

def normalize_exercise_name(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").split())

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_workout_data(prompt: str, user_id: str) -> dict:
    try:
        user = users_collection.find_one({"user_id": user_id}, {"weight_kg": 1})
        user_weight_kg = user.get("weight_kg") if user else None

        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": """Extract workout data from the prompt.
                    Return JSON with: workout_name, sets, reps, weight_kg (null if not specified), muscle_group (lowercase), exercise_type (bodyweight or strength).
                    Example: {"workout_name": "push ups", "sets": 3, "reps": 12, "weight_kg": null, "muscle_group": "chest", "exercise_type": "bodyweight"}"""
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        extracted = json.loads(response.choices[0].message.content)
        workout_name = extracted.get("workout_name", "unknown")
        muscle_group = extracted.get("muscle_group", "").lower()
        weight_kg = extracted.get("weight_kg")
        exercise_type = extracted.get("exercise_type", "strength").lower()

        normalized_name = normalize_exercise_name(workout_name)
        if normalized_name in EXERCISE_REGISTRY:
            exercise_type = EXERCISE_REGISTRY[normalized_name]["type"]
            muscle_group = EXERCISE_REGISTRY[normalized_name]["muscle_group"]

        if exercise_type == "bodyweight" and weight_kg is None and user_weight_kg:
            weight_kg = user_weight_kg

        return {
            "workout_name": normalized_name,
            "sets": extracted.get("sets", 1),
            "reps": extracted.get("reps", 1),
            "weight_kg": weight_kg,
            "muscle_group": muscle_group or None
        }
    except Exception as e:
        logger.error(f"AI extraction failed: {str(e)}")
        return {
            "workout_name": normalize_exercise_name(prompt.split()[0] if prompt else "unknown"),
            "sets": 1,
            "reps": 1,
            "weight_kg": user_weight_kg if user_weight_kg else None,
            "muscle_group": None
        }

def calculate_bmr(age: int, gender: str, height_cm: float, weight_kg: float) -> float:
    if gender and gender.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
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

# Endpoints
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
            "gender": user.gender,
            "activity_level": user.activity_level,
            "fitness_goals": user.fitness_goals,
            "created_at": datetime.utcnow(),
            "total_workouts": 0
        }
        users_collection.insert_one(user_data)
        return UserProfile(**user_data)
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = users_collection.find_one({"email": form_data.username})
        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        access_token = create_access_token(
            data={"sub": user["user_id"]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/workout/log/", response_model=WorkoutData)
async def log_workout(workout: WorkoutPrompt, current_user: UserProfile = Depends(get_current_user)):
    extracted_data = extract_workout_data(workout.prompt, user_id=current_user.user_id)
    duration_minutes = estimate_duration(
        extracted_data["workout_name"], extracted_data["sets"], extracted_data["reps"]
    )
    calories_burned = estimate_calories(
        extracted_data["workout_name"], extracted_data["sets"], extracted_data["reps"],
        extracted_data["weight_kg"], duration_minutes
    )
    workout_data = {
        "id": str(uuid.uuid4()),
        "user_id": current_user.user_id,
        "workout_name": extracted_data["workout_name"],
        "sets": extracted_data["sets"],
        "reps": extracted_data["reps"],
        "weight_kg": extracted_data["weight_kg"],
        "calories_burned": calories_burned,
        "duration_minutes": duration_minutes,
        "timestamp": datetime.utcnow(),
        "muscle_group": extracted_data["muscle_group"]
    }
    try:
        workouts_collection.insert_one(workout_data)
        users_collection.update_one(
            {"user_id": current_user.user_id},
            {"$inc": {"total_workouts": 1}}
        )
        return WorkoutData(**workout_data)
    except Exception as e:
        logger.error(f"Failed to log workout: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/api/workout/history/", response_model=List[WorkoutData])
async def get_workout_history(current_user: UserProfile = Depends(get_current_user)):
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        return [WorkoutData(**{**w, "id": str(w["id"]), "duration_minutes": w.get("duration_minutes", 0.0)}) for w in workouts]
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/api/workout/stats/", response_model=StatsResponse)
async def get_workout_stats(current_user: UserProfile = Depends(get_current_user)):
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        if not workouts:
            return StatsResponse(
                total_sets=0, total_reps=0, total_volume=0,
                muscle_group_breakdown={}, daily={}, weekly={}, monthly={}
            )
        total_sets = total_reps = total_volume = 0
        muscle_group_breakdown = {}
        daily_summary = weekly_summary = monthly_summary = {}
        for log in workouts:
            total_sets += log["sets"]
            total_reps += log["reps"]
            volume = calculate_volume(log)
            total_volume += volume
            muscle = log["muscle_group"] or "other"
            muscle_group_breakdown[muscle] = muscle_group_breakdown.get(muscle, 0) + volume
            date_str = log["timestamp"].strftime("%Y-%m-%d")
            daily_summary.setdefault(date_str, []).append(
                WorkoutData(**{**log, "id": str(log["id"]), "duration_minutes": log.get("duration_minutes", 0.0)})
            )
            if log["timestamp"] >= datetime.utcnow() - timedelta(days=7):
                weekly_summary.setdefault(date_str, []).append(
                    WorkoutData(**{**log, "id": str(log["id"]), "duration_minutes": log.get("duration_minutes", 0.0)})
                )
            if log["timestamp"] >= datetime.utcnow() - timedelta(days=30):
                monthly_summary.setdefault(date_str, []).append(
                    WorkoutData(**{**log, "id": str(log["id"]), "duration_minutes": log.get("duration_minutes", 0.0)})
                )
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
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/api/workout/recommend/", response_model=List[WorkoutRecommendation])
async def recommend_workout(current_user: UserProfile = Depends(get_current_user)):
    try:
        recent_workouts = list(workouts_collection.find(
            {"user_id": current_user.user_id}, sort=[("timestamp", -1)], limit=5
        ))
        workout_context = "\n".join([
            f"{w['workout_name']}: {w['sets']} sets, {w['reps']} reps, {w['weight_kg']}kg"
            for w in recent_workouts
        ]) or "No recent workouts."
        prompt = f"""
        Create a workout plan for {current_user.username}.
        Profile: Age: {current_user.age or 'unknown'}, Weight: {current_user.weight_kg or 'unknown'}kg,
        Height: {current_user.height_cm or 'unknown'}cm, Gender: {current_user.gender or 'unknown'},
        Activity Level: {current_user.activity_level or 'unknown'}, Goals: {current_user.fitness_goals or 'general fitness'},
        Recent Workouts: {workout_context}
        Suggest 3 workouts. For bodyweight exercises, use {current_user.weight_kg or 'unknown'}kg if no weight.
        Return JSON array of 3 objects with: workout_name, sets, reps, weight_kg, muscle_group, description,
        motivational_message (address {current_user.username}), difficulty_level (Beginner|Intermediate|Advanced),
        estimated_duration_minutes, equipment_needed.
        """
        response = ai_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": "Return valid JSON"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_data = json.loads(response.choices[0].message.content)
        recommendations = response_data if isinstance(response_data, list) else response_data.get("workouts", [])
        if not recommendations:
            raise HTTPException(status_code=500, detail="No recommendations")
        validated_recommendations = []
        for rec in recommendations:
            normalized_name = normalize_exercise_name(rec.get("workout_name", ""))
            if (normalized_name in EXERCISE_REGISTRY and
                EXERCISE_REGISTRY[normalized_name]["type"] == "bodyweight" and
                rec.get("weight_kg") is None and current_user.weight_kg):
                rec["weight_kg"] = current_user.weight_kg
            validated_recommendations.append(WorkoutRecommendation(**rec))
        return validated_recommendations
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation error")

@app.get("/api/workout/progress/")
async def get_workout_progress(current_user: UserProfile = Depends(get_current_user)):
    try:
        workouts = list(workouts_collection.find({"user_id": current_user.user_id}))
        if not workouts:
            return {
                "type": "line",
                "data": {
                    "labels": [],
                    "datasets": [{"label": "Workout Volume (kg)", "data": [], "borderColor": "#4CAF50", "fill": False}]
                }
            }
        labels = []
        data = []
        today = datetime.utcnow().date()
        for i in range(30):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily_volume = sum(
                calculate_volume(w) for w in workouts if w["timestamp"].date() == date
            )
            labels.append(date_str)
            data.append(daily_volume)
        return {
            "type": "line",
            "data": {
                "labels": labels[::-1],
                "datasets": [{
                    "label": "Workout Volume (kg)",
                    "data": data[::-1],
                    "borderColor": "#4CAF50",
                    "fill": False
                }]
            }
        }
    except Exception as e:
        logger.error(f"Progress fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Progress fetch failed")

@app.get("/api/exercises/", response_model=Dict[str, List[ExerciseResponse]])
async def get_exercises():
    try:
        exercises = [ExerciseResponse(name=name, **details) for name, details in EXERCISE_REGISTRY.items()]
        return {"exercises": exercises}
    except Exception as e:
        logger.error(f"Failed to fetch exercises: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch exercises")

@app.post("/api/user/profile/", response_model=UserProfile)
async def create_or_update_user_profile(profile: UserProfileCreate):
    return await register(profile)

@app.get("/api/user/profile/", response_model=UserProfile)
async def get_user_profile(current_user: UserProfile = Depends(get_current_user)):
    return current_user

@app.put("/api/user/profile/", response_model=UserProfile)
async def update_user_profile(profile_update: UserProfileUpdate, current_user: UserProfile = Depends(get_current_user)):
    try:
        update_data = {k: v for k, v in profile_update.dict().items() if v is not None}
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        if "email" in update_data and users_collection.find_one({"email": update_data["email"], "user_id": {"$ne": current_user.user_id}}):
            raise HTTPException(status_code=400, detail="Email already in use")
        users_collection.update_one(
            {"user_id": current_user.user_id},
            {"$set": update_data}
        )
        updated_user = users_collection.find_one({"user_id": current_user.user_id})
        return UserProfile(**updated_user)
    except Exception as e:
        logger.error(f"Profile update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Profile update failed")

@app.get("/api/user/calorie-calculator/", response_model=CalorieCalculation)
async def calorie_calculator(current_user: UserProfile = Depends(get_current_user)):
    try:
        if not all([current_user.age, current_user.gender, current_user.height_cm, current_user.weight_kg, current_user.activity_level]):
            raise HTTPException(status_code=400, detail="Complete profile required")
        bmr = calculate_bmr(
            current_user.age, current_user.gender, current_user.height_cm, current_user.weight_kg
        )
        daily_calories = round(bmr * current_user.activity_level, 1)
        maintenance_calories = daily_calories
        weight_loss_calories = round(daily_calories - 500, 1)
        weight_gain_calories = round(daily_calories + 500, 1)
        guidelines = f"""
        - Maintenance: ~{maintenance_calories} cal/day
        - Weight Loss: ~{weight_loss_calories} cal/day
        - Weight Gain: ~{weight_gain_calories} cal/day
        - Balanced diet recommended
        """
        return CalorieCalculation(
            bmr=bmr,
            daily_calories=daily_calories,
            maintenance_calories=maintenance_calories,
            weight_loss_calories=weight_loss_calories,
            weight_gain_calories=weight_gain_calories,
            guidelines=guidelines
        )
    except Exception as e:
        logger.error(f"Calorie calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Calorie calculation failed")

@app.get("/health/")
async def health_check():
    try:
        client.admin.command("ping")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "database": "disconnected"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)