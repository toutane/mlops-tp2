from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import numpy as np

app = FastAPI()

mlflow.set_tracking_uri(uri="http://mlflow:8000")
model = mlflow.pyfunc.load_model("models:/tracking-quickstart/latest")

@app.post("/predict")
async def predict(x: list[list[float]]):
    X = np.array(x)
    y = model.predict(X)
    return {"y_pred": y.tolist()} 

class Item(BaseModel):
    version: str

@app.post("/update-model", status_code=200)
async def updateModel(item: Item):
    try:
        model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{item.version}")
        return {
                "status": "success",
                "message": "Successfully changed model"
                }
    except:
        raise HTTPException(
                status_code=404,
                detail="Model not found"
                )
