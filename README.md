# 12-Lead ECG Classification API 🫀

A high-performance **FastAPI** backend for classifying 12-lead ECG signals using **Convolutional Neural Networks (CNN)** and **Stationary Wavelet Transform (SWT)**.

Designed to integrate with React/SciChart dashboards by providing both diagnostic predictions and raw signal data for visualization.

## ⚡ Quick Start

Follow these steps immediately after cloning the repository.

### 1. Environment Setup (Mandatory)

**Do not** run this project in your global Python environment. Create an isolated virtual environment to avoid dependency conflicts.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

🚀 ECG Classification API — Setup Guide (Mac / Linux)

1. Create Virtual Environment
   python3 -m venv venv
   source venv/bin/activate

2. Install Dependencies

Install all required packages:

pip install -r requirements.txt

⚠️ Important:
If you encounter this error:

ModuleNotFoundError: No module named 'pywt'

Install PyWavelets manually:

pip install PyWavelets

3. Place the Model File

Make sure your trained model is located in the project root.

Filename: model_ecg.keras

Location: Same directory as main.py

4. Run the Server

Start your FastAPI server:

python main.py

🔗 URLs

API Base URL: http://localhost:8000

Swagger UI: http://localhost:8000/docs

🔌 API Specification
POST /predict — Predict ECG Classification

Upload a .zip file containing WFDB records (.dat + .hea files).

Endpoint: http://localhost:8000/predict

Content-Type: multipart/form-data

📤 Request Body
Field Type Description
file ZIP file Contains .dat and .hea WFDB records
📥 Example Response (JSON)

{
"status": "success",
"filename": "record_100.zip",
"total_beats_detected": 12,
"prediction": "MI",
"confidence": "89.5%",
"probabilities": {
"NORM": 0.05,
"MI": 0.89,
"STTC": 0.06
},
"signal_data": [
[0.12, 0.15, 0.18, ...],
[0.05, 0.08, 0.09, ...]
]
}

prediction: predicted ECG class (e.g., MI, NORM, STTC)

signal_data: 12 arrays (Lead I → Lead V6), ready for frontend plotting
