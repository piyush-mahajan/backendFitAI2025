from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import os
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
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
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB Atlas!")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise Exception(f"MongoDB connection failed: {str(e)}")

db = client[mongo_db_name]
users_collection = db["users"]
workouts_collection = db["workouts"]

app = FastAPI(
    title="Workout Tracker API",
    description="API for processing workout prompts and tracking fitness data per user",
    version="1.0.0"
)

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
    user_id: str

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
    user_id: str
    username: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    age: Optional[int] = None

class UserProfile(BaseModel):
    user_id: str
    username: str
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

# API Endpoints
@app.post("/api/workout/log/", response_model=WorkoutData)
async def log_workout(workout: WorkoutPrompt):
    """Log a new workout for a user"""
    extracted_data = extract_workout_data(workout.prompt)
    
    workout_data = {
        "id": str(uuid.uuid4()),
        "user_id": workout.user_id,
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
        # Check if user exists
        user = users_collection.find_one({"user_id": workout.user_id})
        if not user:
            logger.warning(f"No profile found for user_id: {workout.user_id}")
            raise HTTPException(status_code=400, detail="User profile must be created first")
        
        # Insert workout
        result = workouts_collection.insert_one(workout_data)
        logger.info(f"Workout inserted with ID: {result.inserted_id}")
        
        # Update user's total_workouts
        users_collection.update_one(
            {"user_id": workout.user_id},
            {"$inc": {"total_workouts": 1}}
        )
        
        return WorkoutData(**workout_data)
    except Exception as e:
        logger.error(f"Failed to log workout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/workout/history/{user_id}", response_model=List[WorkoutData])
async def get_workout_history(user_id: str):
    """Retrieve workout history for a specific user"""
    try:
        workouts = list(workouts_collection.find({"user_id": user_id}))
        logger.info(f"Found {len(workouts)} workouts for user_id: {user_id}")
        return [WorkoutData(**{**w, "id": str(w["id"])}) for w in workouts]
    except Exception as e:
        logger.error(f"Failed to fetch workout history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/workout/stats/{user_id}", response_model=StatsResponse)
async def get_workout_stats(user_id: str):
    """Get detailed workout statistics and summaries for a user"""
    try:
        if not users_collection.find_one({"user_id": user_id}):
            logger.warning(f"No profile found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        workouts = list(workouts_collection.find({"user_id": user_id}))
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
    """Create or update a user profile with personal data"""
    try:
        existing_profile = users_collection.find_one({"user_id": profile.user_id})
        
        if existing_profile:
            updated_profile = {
                "user_id": profile.user_id,
                "username": profile.username or existing_profile["username"],
                "height_cm": profile.height_cm if profile.height_cm is not None else existing_profile["height_cm"],
                "weight_kg": profile.weight_kg if profile.weight_kg is not None else existing_profile["weight_kg"],
                "age": profile.age if profile.age is not None else existing_profile["age"],
                "created_at": existing_profile["created_at"],
                "total_workouts": existing_profile["total_workouts"]
            }
            users_collection.update_one(
                {"user_id": profile.user_id},
                {"$set": updated_profile}
            )
            logger.info(f"Updated profile for user_id: {profile.user_id}")
        else:
            updated_profile = {
                "user_id": profile.user_id,
                "username": profile.username,
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "age": profile.age,
                "created_at": datetime.utcnow(),
                "total_workouts": 0
            }
            result = users_collection.insert_one(updated_profile)
            logger.info(f"Created profile for user_id: {profile.user_id} with ID: {result.inserted_id}")
        
        return UserProfile(**updated_profile)
    except Exception as e:
        logger.error(f"Failed to create/update profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/user/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Fetch user profile data"""
    try:
        profile = users_collection.find_one({"user_id": user_id})
        if not profile:
            logger.warning(f"No profile found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        logger.info(f"Fetched profile for user_id: {user_id}")
        return UserProfile(**profile)
    except Exception as e:
        logger.error(f"Failed to fetch profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def main():
    test_prompt = "I did 3 sets of bench press with 50kg for 10 reps"
    test_user_id = "user123"
    print(f">>> POST /api/workout/log/ with prompt: '{test_prompt}' for user: {test_user_id}")
    extracted = extract_workout_data(test_prompt)
    print(f"Extracted data: {extracted}")

if __name__ == "__main__":
    main()