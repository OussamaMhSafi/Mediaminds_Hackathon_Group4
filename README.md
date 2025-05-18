# Mediaminds_Hackathon_Group4

<p align="center">
  <img src="studio/static/logo.png" alt="Logo 1" width="400"/>
  <img src="studio/static/logo2.png" alt="Logo 2" width="400"/>
</p>


Welcome to Authentiq Eye, our award-winning solution developed during the Mediaminds Hackathon, organized by Huawei Cloud and Al Jazeera. This innovative platform leverages Huawei Cloud technologies, specifically Object Storage Service (OBS) and Elastic Cloud Server (ECS), integrated with LangChain's multi-agent architecture for robust processing capabilities.

Our team addressed the critical challenge of detecting manipulated media content, including fake images, misleading news, and deepfake materials. 

Authentiq Eye implements a sophisticated multi-tool single agent architecture that combines:
- Web scraping for comprehensive content analysis
- Reverse image search capabilities
- AI-generated image detection
- Advanced deepfake manipulation detection

This solution earned second place recognition in the Mediaminds Hackathon, validating its effectiveness and innovation in combating digital misinformation.

## Setup Instructions to Run the Langgraph Graph locally

### Prerequisites
- Python 3.11

### Installation Steps

1. Create a virtual environment:
```python
py -3.11 -m venv lg-env
```

2. Activate the virtual environment:

**Linux/git bash:**
```
source lg-env/Scripts/activate
```

3. Enter Studio Directory:
```bash
cd studio
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing Using LangChain

1. Create a virtual environment:
```python
py -3.11 -m venv lg-env
```

2. Activate the virtual environment:

**Linux/git bash:**
```
source lg-env/Scripts/activate
```

3. Enter Studio Directory:
```bash
cd studio
```

4. Install dependencies:
```bash
langgraph dev
```

## Testing Using FastAPI (locally)

1. Create a virtual environment:
```python
py -3.11 -m venv lg-env
```

2. Activate the virtual environment:

**Linux/git bash:**
```
source lg-env/Scripts/activate
```

3. Enter Studio Directory:
```bash
cd studio
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Go Back to Main Folder:

```bash
cd ..
```

5. Run Backend:

```bash
uvicorn serve:app --reload
```

6. Run Frontend (on another terminal) and go to localhost:8080:

```bash
cd frontent/frontend
python -m http.server 8080
```