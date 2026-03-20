from api.database import collection
from bson import ObjectId


def serial_model(id: str) -> dict:
    """
    Serializza un documento del database in un dizionario Python, convertendo l'ObjectId in stringa, e lo cerca nel db. 
    
    Args:
    `id (str)` L'ID del documento da serializzare.

    Returns:
    `dict` Un dizionario contenente i dati del documento, con l'ID convertito in stringa. Se il documento non viene trovato, restituisce un dizionario vuoto.
    """
    
    document = collection.find_one({"_id": ObjectId(id)})
    if not document:
        return {}
    
    document["_id"] = str(document["_id"])
    return document
