from api.database import collection
from bson import ObjectId

def serial_model(id: str) -> dict:
    document = collection.find_one({"_id": ObjectId(id)})
    if not document:
        return {}
    return {key: value for key, value in document.items() if key != "_id"}
