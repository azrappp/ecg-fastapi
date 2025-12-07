import os
import shutil
import uuid
import zipfile
import numpy as np
import wfdb
import pywt
from scipy.signal import butter, lfilter
from fastapi import FastAPI, UploadFile, File, HTTPException, Query # <--- MODIFIKASI: Tambah Query
from contextlib import asynccontextmanager
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal # <--- MODIFIKASI: Untuk validasi pilihan model

# ==========================================
# 1. KONFIGURASI & KONSTANTA
# ==========================================
FS = 100
BEFORE = 40
AFTER = 120
LOWCUT = 0.5
HIGHCUT = 24.0
WAVELET = "db4"
LEVEL = 2

CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"] 

# ==========================================
# MODIFIKASI: KONFIGURASI MODEL
# ==========================================
# Mapping nama query param -> nama file model
MODEL_FILES = {
    "resnet": "./model/ResNet.keras",
    "cnn": "./model/CNN.keras",
    "attention": "./model/Attention.keras",
    "multiscale": "./model/MultiScale.keras"
}

# Variabel Global untuk menyimpan SEMUA Model
loaded_models = {}

# ==========================================
# 2. LOAD MODEL SAAT STARTUP (MODIFIKASI)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_models
    print("🚀 Memulai proses loading model...")
    
    # Loop untuk memuat semua model yang terdaftar
    for key, path in MODEL_FILES.items():
        try:
            if os.path.exists(path):
                print(f"   ⏳ Memuat {key} dari {path}...")
                loaded_models[key] = tf.keras.models.load_model(path)
                print(f"   ✅ {key} BERHASIL dimuat!")
            else:
                print(f"   ❌ File tidak ditemukan: {path}")
        except Exception as e:
            print(f"   ❌ Gagal memuat {key}: {e}")
            
    yield
    
    loaded_models.clear()
    print("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. FUNGSI PREPROCESSING (TETAP SAMA)
# ==========================================
# ... (Bagian ini tidak perlu diubah, sama seperti kodemu sebelumnya) ...

def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal, axis=0)

def derivative_filter(ecg_signal):
    kernel = np.array([-1, -2, 0, 1, 2]) * (1/8)
    return np.convolve(ecg_signal, kernel, mode='same')

def moving_window(signal, fs, window_ms=150):
    window_size = int((window_ms/1000)*fs)
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def manual_Rpeaks(signal, fs):
    if np.max(signal) == 0: return np.array([], dtype=int)
    threshold = (np.mean(signal) + np.max(signal)) / 2
    min_dist = round(fs * 0.200)
    peaks = []
    last_peak = -min_dist
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            if (i - last_peak) >= min_dist:
                peaks.append(i)
                last_peak = i
    return np.array(peaks, dtype=int)

def extract_beats_12lead(signal_12, r_peaks, fs, before, after):
    beats = []
    seg_len = before + after
    for r in r_peaks:
        start = r - before
        end = r + after
        if start >= 0 and end <= len(signal_12):
            beat = signal_12[start:end, :]
            if beat.shape[0] == seg_len:
                beats.append(beat)
    if len(beats) == 0:
        return np.empty((0, seg_len, signal_12.shape[1]))
    return np.stack(beats, axis=0)

def apply_swt_multilevel(beat_12lead, wavelet="db4", level=2):
    swt_output = []
    for lead in range(beat_12lead.shape[1]):
        sig = beat_12lead[:, lead]
        coeffs = pywt.swt(sig, wavelet, level=level)
        cA_last = coeffs[-1][0]
        swt_output.append(cA_last)
    return np.stack(swt_output, axis=1)

# ==========================================
# 4. API ENDPOINT (MODIFIKASI)
# ==========================================

# Definisi tipe agar muncul dropdown di Swagger UI (Docs)
ModelSelection = Literal["resnet", "cnn", "attention", "multiscale"]

@app.post("/predict")
async def predict_ecg(
    file: UploadFile = File(...),
    model_name: ModelSelection = Query("resnet", description="Pilih arsitektur model") # <--- MODIFIKASI: Query Param
):
    """
    Input: File ZIP berisi .dat dan .hea
    Query Param: model_name (resnet, cnn, attention, multiscale)
    """
    
    # 1. Cek apakah model yang dipilih sudah dimuat
    if model_name not in loaded_models:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' tidak tersedia atau gagal dimuat saat startup.")
    
    selected_model = loaded_models[model_name] # Ambil model spesifik

    # A. Persiapan Folder Sementara
    request_id = str(uuid.uuid4())
    temp_dir = f"temp_{request_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # B. Simpan & Ekstrak ZIP
        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # C. Cari file .hea
        hea_file = None
        for f in os.listdir(temp_dir):
            if f.endswith(".hea"):
                hea_file = f
                break
        
        if not hea_file:
            raise HTTPException(status_code=400, detail="File .hea tidak ditemukan dalam ZIP")
            
        record_name = os.path.splitext(hea_file)[0]
        record_path = os.path.join(temp_dir, record_name)
        
        # D. Preprocessing Pipeline
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
            
            if record.fs != FS:
                 raise HTTPException(status_code=400, detail=f"Sampling rate harus {FS}Hz, terdeteksi {record.fs}Hz")

            filtered = bandpass_filter(sig, LOWCUT, HIGHCUT, FS)

            lead_ii = filtered[:, 1]
            der = derivative_filter(lead_ii)
            sqr = der**2
            win = moving_window(sqr, FS)
            r_peaks = manual_Rpeaks(win, FS)

            if len(r_peaks) == 0:
                raise HTTPException(status_code=400, detail="Tidak ada R-Peak terdeteksi")

            beats_raw = extract_beats_12lead(filtered, r_peaks, FS, BEFORE, AFTER)
            
            if len(beats_raw) == 0:
                 raise HTTPException(status_code=400, detail="Gagal ekstrak beats.")

            beats_swt = []
            for beat in beats_raw:
                beats_swt.append(apply_swt_multilevel(beat, WAVELET, LEVEL))
            
            X_input = np.stack(beats_swt).astype(np.float32)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saat preprocessing: {str(e)}")

        # E. Prediksi Menggunakan Model YANG DIPILIH
        # Gunakan selected_model, bukan variabel global 'model'
        predictions = selected_model.predict(X_input) 
        
        # F. Agregasi Hasil
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(avg_prediction[predicted_index])
        
        # Untuk grafik (hanya ambil data 12 lead pertama dari hasil filter)
        signal_for_chart = filtered.T.tolist()

        return {
            "status": "success",
            "model_used": model_name, # Info model apa yang dipakai
            "filename": file.filename,
            "total_beats_detected": int(len(beats_raw)),
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}",
            "probabilities": {
                class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, avg_prediction)
            },
            "lead_names": record.sig_name,
            "signal_data": signal_for_chart
        }

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)