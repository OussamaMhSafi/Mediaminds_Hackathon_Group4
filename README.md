# Mediaminds_Hackathon_Group4

<p align="center">
  <img src="studio/static/logo.png" alt="Logo 1" width="400"/>
  <img src="studio/static/logo2.png" alt="Logo 2" width="400"/>
</p>


This is the codebase for the Solution of GroupD as part of the Mediaminds hackathon organised by Huawei Cloud and Aljazeera. The solution utilises Huawei cloud technologies (OBS bucker, ECS) for storage and hosting toghether with langchain for multiagent architecture.

Our team received second place in the hackathon.

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