from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os

load_dotenv()

username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
host = os.getenv("MONGO_HOST", "localhost") #di default prende come valore `localhost`
client = MongoClient(f"mongodb://{ username }:{ password }@{ host }:27017/")
db = client.models_db   
collection = db['models_collection']
fs = GridFS(db)