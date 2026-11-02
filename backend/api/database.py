from pymongo import AsyncMongoClient
from gridfs import AsyncGridFSBucket
from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
host = os.getenv("MONGO_HOST", "localhost")

client = AsyncMongoClient(f"mongodb://{username}:{password}@{host}:27017/")
db = client.models_db
collection = db['models_collection']
# Collezione dedicata al log delle validazioni/scarti dei filologi (Human-in-the-loop Fase 1)
feedback_collection = db['feedback_collection']
fs = AsyncGridFSBucket(db)
