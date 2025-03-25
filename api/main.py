from fastapi import FastAPI
from api.route import router

app = FastAPI()
app.include_router(router, include_in_schema=True)