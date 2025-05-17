from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil, os, traceback

from dotenv import load_dotenv
load_dotenv()
    
from studio.image_classification import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from your local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = graph.invoke({"image": temp_path})
        print("🧠 FINAL RESULT:", result)  # <== keep this too

        os.remove(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        print("❌ ERROR during graph processing:")
        traceback.print_exc()  # <== 👈 this will show exact line and reason
        return JSONResponse(status_code=500, content={"error": str(e)})

