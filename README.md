

````markdown
# 🫀 12-Lead ECG Classification API

A high-performance **FastAPI** backend for classifying 12-lead ECG signals using multiple Deep Learning architectures (**ResNet, CNN, Attention, Multi-Scale**) and **Stationary Wavelet Transform (SWT)** for feature extraction.

Designed to integrate seamlessly with React/SciChart dashboards by providing diagnostic predictions, confidence scores, and raw signal data for visualization.

---

## ⚡ Quick Start

Follow these steps immediately after cloning the repository.

### 1. Environment Setup (Mandatory)

**Do not** run this project in your global Python environment.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
````

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

-----

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Common Issue:**
> If you encounter `ModuleNotFoundError: No module named 'pywt'`, install it manually:
>
> ```bash
> pip install PyWavelets
> ```

-----

### 3\. Folder Structure & Models

Ensure your directory looks like this. The API expects a `model` folder containing the `.keras` files.

```text
project_root/
├── model/                  <-- Create this folder
│   ├── ResNet.keras
│   ├── CNN.keras
│   ├── Attention.keras
│   └── MultiScale.keras
├── main.py
├── requirements.txt
└── ...
```

-----

### 4\. Run the Server

```bash
python main.py
```

The server will start at `http://0.0.0.0:8000`.

-----

## 🔗 Documentation

  - **API Base URL:** [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000)
  - **Interactive Swagger UI:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

-----

# 🔌 API Specification

## `POST /predict`

Upload a `.zip` file containing WFDB records (`.dat` + `.hea`) and select a model architecture for prediction.

### Query Parameters

| Parameter | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model_name` | `string` | `resnet` | `resnet`, `cnn`, `attention`, `multiscale` | The specific model architecture to use for inference. |

### Request Body (multipart/form-data)

| Field | Type | Description |
| :--- | :--- | :--- |
| `file` | `File` | A **ZIP** file containing the ECG `.dat` and `.hea` files. |

-----

### 📥 Example Response

```json
{
  "status": "success",
  "model_used": "resnet",
  "filename": "record_100.zip",
  "total_beats_detected": 12,
  "prediction": "MI",
  "confidence": "89.50%",
  "probabilities": {
    "NORM": 0.05,
    "MI": 0.89,
    "STTC": 0.06,
    "CD": 0.00,
    "HYP": 0.00
  },
  "lead_names": [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
  ],
  "signal_data": [
    [0.12, 0.15, 0.18, ...], 
    [0.05, 0.08, 0.09, ...]
  ]
}
```

### 🗝️ Response Fields

  * **model\_used**: The architecture used for this specific prediction.
  * **prediction**: The class with the highest probability (e.g., `MI`, `NORM`).
  * **confidence**: The probability percentage of the predicted class.
  * **signal\_data**: Array of 12 arrays (Lead I through Lead V6), pre-filtered and ready for SciChart rendering.


```
```
