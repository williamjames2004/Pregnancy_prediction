from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def connect_db():
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client.get_database("test")  # or your DB name
        print("MongoDB connected successfully")
        return db
    except Exception as e:
        print("MongoDB connection failed:", e)
        exit(1)