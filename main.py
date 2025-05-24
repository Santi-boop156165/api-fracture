from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from predictions import predict,predict_with_heatmap
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ["http://localhost:5173"] si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = "./temp"

# Asegúrate de que exista la carpeta temporal
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Bone Fracture Detection API"}

@app.post("/predict/")
async def predict_fracture(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        bone_type = predict(file_path, "Parts")
        result = predict_with_heatmap(file_path, bone_type)

        return {
            "filename": file.filename,
            "bone_type": bone_type,
            "fracture_result": result
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})