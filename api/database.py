from pymongo import MongoClient
from config.settings import MONGO_URI

client = MongoClient(MONGO_URI)
db = client.models_db
collection = db['models_collection']
    