# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:17:37 2020

@author: gauravvijayvergiya
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile
from components.Model import Model
from io import BytesIO 
from PIL import Image
import numpy as np

app = FastAPI() 


lables = ['Buildings', 'Sea', 'Glacier', 'Mountain', 'Forest', 'Street']

@app.post("/predict/image") 
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(BytesIO(await file.read()))
    model = Model()
    model.load_model()
    pred_result = model.predict_img(image)
    response = []
    for i in range(0,len(pred_result[0])):
        resp = {}
        resp["class"] = lables[i]
        resp["confidence"] = f"{pred_result[0][i]*100:0.2f} %"
        response.append(resp)
    print(response)
    return response
    
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)