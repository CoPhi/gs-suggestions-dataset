from fastapi import FastAPI
from api.route import router

app = FastAPI(title="gs-api", version="0.1.0")
app.include_router(router, include_in_schema=True)