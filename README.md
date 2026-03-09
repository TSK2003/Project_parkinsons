# Parkinson's Detection Voice API

This project contains a Python Flask API that detects Parkinson's disease from voice features. It provides an endpoint to upload or stream audio chunks and predict if the voice belongs to a healthy person or someone with Parkinson's.

## How to Run the Project on Windows

Follow these steps to set up the project locally on a Windows machine:

### 1. Prerequisites
- **Python**: Make sure you have Python 3.8+ installed. You can download it from [python.org](https://www.python.org/).
- **Git** (optional): Used to clone this repository.

### 2. Setup the Project
1. Open your Command Prompt (`cmd`) or PowerShell.
2. Navigate to the project directory where you want to run this application.

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

Run the following command to create a virtual environment named `venv`:
```cmd
python -m venv venv
```

### 4. Activate the Virtual Environment
Before installing packages or running the app, you need to activate the virtual environment.

For **Command Prompt (cmd)**:
```cmd
venv\Scripts\activate.bat
```

For **PowerShell**:
```powershell
venv\Scripts\Activate.ps1
```
*(Note: If you get an execution policy error in PowerShell, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator once.)*

Once activated, you should see `(venv)` appear on the left side of your terminal prompt.

### 5. Install Dependencies
With the virtual environment activated, install all the required Python packages using pip:
```cmd
pip install -r requirements.txt
```

### 6. Run the Application
Finally, start the Flask development server:
```cmd
python app.py
```

The API will start running locally at `http://127.0.0.1:5000`. You can test it by opening `http://127.0.0.1:5000/` in your web browser.

### Key API Endpoints
- `GET /` - Checks if the API is live.
- `POST /stream-chunk` - Endpoint for streaming continuous audio chunks.
- `POST /analyze-voice` - Endpoint for analyzing a complete audio file.
