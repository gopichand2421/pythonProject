from typing import Union

import cv2
from fastapi import FastAPI
from fastapi import File, UploadFile
from src.model import train_model
import os


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello":"world"}

@app.post("/triggermodel")
def trigger_root(file: UploadFile = File(...)):
    contents = file.file.read()
    with open(os.path.join("src/images",file.filename), 'wb') as f:
        f.write(contents)
    img = cv2.imread(os.path.join("src/images",file.filename), cv2.IMREAD_UNCHANGED)[:,:,0]
    val = train_model(img)

    return {"message": str(val)}