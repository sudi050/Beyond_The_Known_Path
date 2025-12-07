"""Configuration file for the travel recommendation system"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# API Keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 

# Database Configuration
DATABASE_PATH = "data/travel_places.db"
VECTOR_DB_PATH = "data/chroma_db"

# Scraping Configuration
SCRAPING_DELAY = 2  # seconds between requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Model Configuration (pinned for compatibility)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLASSIFICATION_MODEL_PATH = "models/attribute_classifier.pkl"

# User Type Categories
USER_TYPES = ["Food Explorer", "Trekker", "Vlogger", "History Buff", "Peace Seeker"]

# Attribute Tags (matches rule-based classifier)
ATTRIBUTE_TAGS = {
    "vibe": ["calm", "adventurous", "photogenic", "crowded", "peaceful", "energetic"],
    "difficulty": ["easy", "moderate", "challenging", "extreme"],
    "best_time": ["morning", "afternoon", "evening", "night", "sunrise", "sunset"],
    "accessibility": ["public_transport", "private_vehicle", "walking", "bike"],
    "cost": ["free", "budget", "moderate", "expensive"],
    "duration": ["quick_visit", "half_day", "full_day", "multi_day"]
}

# Travel Group Types
GROUP_TYPES = ["solo", "couple", "group", "family"]

# LLM Configuration (for planner.py)
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.7

# Version pins for dependency stability
DEPENDENCY_VERSIONS = {
    "sentence-transformers": "2.2.2",
    "transformers": "4.32.0",
    "huggingface-hub": "0.25.2",
    "chromadb": "0.4.24",
    "streamlit": "1.32.0",
    "openai": "1.40.0"  
}
