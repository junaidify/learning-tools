from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()
app.mount('/static', StaticFiles(directory='backend/static'), name="static")
templates = Jinja2Templates(directory="backend/templates")

mongo_connection = ""

if not mongo_connection: 
    raise ValueError('MONGO_URL not found in .env file')
client = MongoClient(mongo_connection)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})