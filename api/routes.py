from fastapi import FastAPI, HTTPException
from models.infer import generate_k_suggests

app = FastAPI()

@app.get("/models/{id}")
def get_model(id: int):
    return {
        "id": id
    } 

@app.get("/models")
def get_models(): 
    pass

@app.post("/models") #usata per caricare i modelli
def create_models(): 
    pass

@app.get("/predictions")
def restore(context: str, num_words: int, id: int):
    return {
        "context": context, 
        "num_words": num_words, 
        "id" : id
    } 