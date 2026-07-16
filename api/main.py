from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.predict import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Waste Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = predict_image(image)

    return result