from api.database import collection
from bson import ObjectId

def serial_model(id: str) -> dict:
    document = collection.find_one({"_id": ObjectId(id)})
    if not document:
        return {}
    document["_id"] = str(document["_id"])  # Convert ObjectId to string
    return document
