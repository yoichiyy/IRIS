from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

app = FastAPI()

class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
model = pickle.load(open('model_iris', 'rb'))

@app.get('/')
async def index():
    return {"Iris":'iris_prediction'}

@app.post('/make_predictions')
async def make_predictions(features: iris):
    return({'prediction': str(model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])[0])})
