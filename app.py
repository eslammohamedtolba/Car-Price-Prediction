from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import uvicorn
import joblib

# Paths where the encoders and model saved
label_encoders_path = 'PrepareModel/label_encoders.sav'
random_forest_model_path = 'PrepareModel/Random_forest_model.sav'

# Load the encoders and the model
label_encoders = joblib.load(label_encoders_path)
model = joblib.load(random_forest_model_path)

# Create application
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Route for the home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route for the predict page
@app.post("/predict")
async def predict(request: Request, name: str = Form(...), year: int = Form(...), km_driven: int = Form(...),
    fuel: str = Form(...), seller_type: str = Form(...), transmission: str = Form(...), owner: str = Form(...),
    mileage: float = Form(...), engine: int = Form(...), max_power: float = Form(...), seats: int = Form(...)):

    # Prepare the input data for prediction
    input_data = [[name, fuel, seller_type, transmission, owner]]
    input_data_numeric = [year, np.log(km_driven), mileage, engine, max_power, seats]

    # Encode categorical features using the loaded label encoders
    for i, col in enumerate(['name', 'fuel', 'seller_type', 'transmission', 'owner']):
        encoder = label_encoders[col]
        input_data[0][i] = encoder.transform([input_data[0][i]])[0]

    # Combine encoded categorical features with numeric features
    input_data = input_data[0] + input_data_numeric
    input_data = [input_data]

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


