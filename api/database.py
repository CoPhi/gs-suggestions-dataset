from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client.models_db   
collection = db['models_collection']
fs = GridFS(db)