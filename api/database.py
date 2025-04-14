from pymongo import MongoClient
from api import MONGO_URI
from gridfs import GridFS

client = MongoClient(MONGO_URI)
db = client.models_db
collection = db['models_collection']
fs = GridFS(db)
    