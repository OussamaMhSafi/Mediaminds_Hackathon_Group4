# app.py

import os
import uuid
import traceback
import pprint
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles # Uncomment if you add a static folder for CSS/JS
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# --- Load Environment Variables FIRST ---
# This is crucial so that image_classifier_workflow.py can access them when it's imported
load_dotenv()

# --- Import from your LangGraph workflow module ---
try:
    from image_classification import graph, ImageClassificationState, Decision
    print("Successfully imported 'graph' from image_classifier_workflow.")
except ImportError as e:
    print(f"Error importing from image_classifier_workflow: {e}")
    print("Please ensure 'image_classifier_workflow.py' is in the same directory or Python path.")
    print("And that all its dependencies (langchain, obs, etc.) are installed.")
    graph = None # Set to None to indicate failure
    # Define dummy classes if import fails to prevent NameError later, though app shouldn't run
    class ImageClassificationState(dict): pass
    class Decision(dict): pass


# --- FastAPI App Setup ---
app = FastAPI(title="Image Classifier API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup for templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Configuration for uploads
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FastAPI Path Operations (Routes) ---

@app.on_event("startup")
async def startup_event():
    if graph is None:
        print("CRITICAL ERROR: LangGraph 'graph' object not loaded. The application will not function correctly.")
    # You can also do API key checks here
    required_env_vars = ["AccessKeyID", "SecretAccessKey", "OPENAI_API_KEY", "SERPAPI_API_KEY", "SIGHTENGINE_API_USER", "SIGHTENGINE_API_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"WARNING: The following environment variables are not set: {', '.join(missing_vars)}")
        print("The application might not function correctly if these are needed by the workflow.")


@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def get_index(request: Request):
    """Serves the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request, "graph_loaded": graph is not None})

@app.post("/upload", response_class=HTMLResponse, tags=["Interface"])
async def upload_and_classify_image(
    request: Request,
    image_file: UploadFile = File(...)
):
    """Handles image upload, triggers classification, and shows results."""
    if graph is None:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "error": "Classification engine (LangGraph) is not loaded. Please check server logs.",
            }
        )

    if not image_file.filename:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No file selected.", "graph_loaded": True})

    if not allowed_file(image_file.filename):
        return templates.TemplateResponse("index.html", {"request": request, "error": "File type not allowed.", "graph_loaded": True})

    unique_suffix = str(uuid.uuid4())
    # Sanitize filename before joining, though UUID makes it mostly safe
    safe_original_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in image_file.filename)
    temp_filename = f"{unique_suffix}_{safe_original_filename}"
    filepath = UPLOAD_DIR / temp_filename

    try:
        contents = await image_file.read()
        if len(contents) > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {"request": request, "error": "File is too large.", "graph_loaded": True})
        
        with open(filepath, "wb") as buffer:
            buffer.write(contents)

        # --- Prepare initial state for LangGraph ---
        # This needs to match the ImageClassificationState TypedDict from your workflow module
        initial_state: ImageClassificationState = {
            "image": str(filepath),
            "image_data": {}, # Will be populated by load_image
            "obs_url": "",    # Will be populated by load_image
            "description": "",# Will be populated
            "web_scrape_results": None,
            "search_query": "",
            "classification": "Fake", # Default, will be overwritten by 'decision'
            "sources": [], # operator.add
            "decision": None,
            "similar_images_count": 0,
            "visual_match_urls": [], # operator.add
            "visual_match_contents": [], # operator.add
            "ai_generated_likelihood": 0.0,
            "deepfake_likelihood": 0.0,
            "is_portrait": False
        }

        print(f"FastAPI app: Starting classification for: {filepath}")
        # --- Invoke the imported graph ---
        # Add recursion_limit if your graph is deep or has cycles handled by conditions
        final_state = graph.invoke(initial_state, config={"recursion_limit": 150})

        print("FastAPI app: Classification complete by imported graph. Final state:")
        pprint.pprint(final_state)

        # Extract results for the template
        # The 'decision' should be a dict matching the Pydantic Decision model
        result_data_dict = final_state.get("decision")
        if result_data_dict and isinstance(result_data_dict, dict):
            # You could optionally re-validate with Pydantic if desired:
            # validated_decision = Decision(**result_data_dict)
            # result_display = validated_decision.model_dump()
            result_display = result_data_dict
        else:
            result_display = {"error": "Classification process did not produce a valid decision."}
            if isinstance(result_data_dict, str): # if it's an error string
                 result_display["explanation_detail"] = result_data_dict


        return templates.TemplateResponse("results.html", {
            "request": request,
            "result": result_display, # This should be the dict from decision.model_dump()
            "obs_url": final_state.get("obs_url"),
            "image_description": final_state.get("description"),
            "ai_likelihood": final_state.get("ai_generated_likelihood"),
            "deepfake_likelihood": final_state.get("deepfake_likelihood"),
            "is_portrait": final_state.get("is_portrait"),
            "similar_images_count": final_state.get("similar_images_count")
        })

    except Exception as e:
        print(f"Error during FastAPI upload/processing: {e}")
        traceback.print_exc()
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": f"An unexpected server error occurred: {str(e)}",
            "full_trace": traceback.format_exc() # Careful in production
        })
    finally:
        if filepath.exists():
            try:
                os.remove(filepath)
                print(f"FastAPI app: Cleaned up temporary file: {filepath}")
            except Exception as e_remove:
                print(f"FastAPI app: Error removing temporary file {filepath}: {e_remove}")
        if image_file:
            await image_file.close()


@app.get("/health", tags=["Utilities"])
async def health_check():
    return {"status": "healthy", "graph_loaded": graph is not None}

# --- Main execution for Uvicorn ---
if __name__ == '__main__':
    import uvicorn
    print("Starting FastAPI app with Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")