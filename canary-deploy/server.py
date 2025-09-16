from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import numpy as np

app = FastAPI()

mlflow.set_tracking_uri(uri="http://localhost:8000")
current_model = mlflow.pyfunc.load_model("models:/tracking-quickstart/1")
next_model = current_model

rng = np.random.default_rng()
p = 0.7

@app.post("/predict")
async def predict(x: list[list[float]]):
    model = None
    if (rng.random() < p):
        print("Predicting using current model")
        model = current_model
    else:
        print("Predicting using next model")
        model = next_model
    X = np.array(x)
    y = model.predict(X)
    return {"y_pred": y.tolist()} 

class Item(BaseModel):
    version: str

@app.post("/update-model", status_code=200)
async def updateModel(item: Item):
    try:
        next_model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{item.version}")
        print(f"Updating next model to version {item.version}")
        return {
                "status": "success",
                "message": "Successfully update the next model"
                }
    except:
        raise HTTPException(
                status_code=404,
                detail="Model not found"
                )

@app.get("/accept-next-model", status_code=200)
async def acceptNextModel():
    print("Current model is now next model")
    current_model = next_model
    return {
            "status": "success",
            "message": "Successfully accept the next model"
            }
