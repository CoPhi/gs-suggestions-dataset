from fastapi import FastAPI
from api.route import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="gs-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #Da cambiare con "http://localhost:4200" e porta usata da Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, include_in_schema=True)