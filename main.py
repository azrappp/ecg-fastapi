import os
import shutil
import uuid
import zipfile
import numpy as np
import wfdb
import pywt
from scipy.signal import butter, lfilter
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware # <--- 1. Import ini


# ==========================================
# 1. KONFIGURASI & KONSTANTA
# ==========================================
# Pastikan nilai ini SAMA PERSIS dengan saat training
FS = 100
BEFORE = 40
AFTER = 120
LOWCUT = 0.5
HIGHCUT = 24.0
WAVELET = "db4"
LEVEL = 2

# Mapping kelas (Sesuaikan urutan dengan classes.npy Anda)
# Contoh urutan alfabetis jika menggunakan LabelEncoder default
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"] 

# Variabel Global untuk Model
model = None

# ==========================================
# 2. LOAD MODEL SAAT STARTUP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model sekali saat server nyala
    global model
    try:
        model_path = "./model/ResNet.keras"  # Ganti dengan path model Anda
        model = tf.keras.models.load_model(model_path)
        print("✅ Model berhasil dimuat!")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
    yield
    # Code clean up jika ada (opsional)
    print("🛑 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua domain (React, Postman, HP, dll)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua method (GET, POST, dll)
    allow_headers=["*"],  # Mengizinkan semua header
)
# ==========================================
# 3. FUNGSI PREPROCESSING (COPY-PASTE DARI TRAINING)
# ==========================================

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
    # Tambahkan safety check jika signal flat
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
        # Pastikan tidak keluar batas array
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
        # Padding jika perlu (SWT butuh panjang genap/kelipatan 2^level)
        # Tapi karena panjang kita 160 (genap dan habis dibagi 4), aman.
        coeffs = pywt.swt(sig, wavelet, level=level)
        cA_last = coeffs[-1][0] # Ambil approximation level terakhir
        swt_output.append(cA_last)
    return np.stack(swt_output, axis=1)

# ==========================================
# 4. API ENDPOINT
# ==========================================

@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    """
    Input: File ZIP berisi .dat dan .hea
    Output: Klasifikasi Penyakit
    """
    
    # A. Persiapan Folder Sementara (Unik per request)
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
            
        # C. Cari file .hea untuk mendapatkan nama record
        hea_file = None
        for f in os.listdir(temp_dir):
            if f.endswith(".hea"):
                hea_file = f
                break
        
        if not hea_file:
            raise HTTPException(status_code=400, detail="File .hea tidak ditemukan dalam ZIP")
            
        record_name = os.path.splitext(hea_file)[0]
        record_path = os.path.join(temp_dir, record_name)
        
        # D. Preprocessing Pipeline (Sama persis dengan training)
        try:
            # 1. Read Record
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
            
            # Cek FS
            if record.fs != FS:
                 raise HTTPException(status_code=400, detail=f"Sampling rate harus {FS}Hz, terdeteksi {record.fs}Hz")

            # 2. Bandpass Filter
            filtered = bandpass_filter(sig, LOWCUT, HIGHCUT, FS)

            # 3. R-Peak Detection (Lead II)
            lead_ii = filtered[:, 1]
            der = derivative_filter(lead_ii)
            sqr = der**2
            win = moving_window(sqr, FS)
            r_peaks = manual_Rpeaks(win, FS)

            # === DEBUG PRINT  ===
            print(f"DEBUG INFO:")
            print(f" - Panjang Sinyal: {len(filtered)} samples")
            print(f" - Jumlah R-Peaks: {len(r_peaks)}")
            print(f" - Lokasi R-Peaks: {r_peaks}")
            # ============================================

            if len(r_peaks) == 0:
                raise HTTPException(status_code=400, detail="Tidak ada R-Peak terdeteksi (Sinyal buruk/Flat)")

            # 4. Segmentation (Sekarang pakai fungsi baru yang ada padding)
            beats_raw = extract_beats_12lead(filtered, r_peaks, FS, BEFORE, AFTER)
            
            print(f" - Beats Extracted: {len(beats_raw)}") # Debug lagi
            
            if len(beats_raw) == 0:
                 raise HTTPException(status_code=400, detail=f"Gagal ekstrak. Sinyal: {len(filtered)}, Peaks: {r_peaks}")

            # 5. Feature Extraction (SWT)
            beats_swt = []
            for beat in beats_raw:
                beats_swt.append(apply_swt_multilevel(beat, WAVELET, LEVEL))
            
            X_input = np.stack(beats_swt).astype(np.float32) # Shape: (N_beats, 160, 12)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saat preprocessing: {str(e)}")

        # E. Prediksi Menggunakan Model
        if model is None:
            raise HTTPException(status_code=500, detail="Model belum dimuat")
            
        # Prediksi per beat (Batch Prediction)
        predictions = model.predict(X_input) # Output: (N_beats, 5) probabilitas
        
        # F. Agregasi Hasil (Voting / Averaging)
        # Kita ambil rata-rata probabilitas dari semua detak
        avg_prediction = np.mean(predictions, axis=0)
        predicted_index = np.argmax(avg_prediction)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(avg_prediction[predicted_index])
        signal_for_chart = filtered.T.tolist()
        record = wfdb.rdrecord(record_path) 
        # Opsi lain: Majority Voting (Detak terbanyak menang)
        # beat_classes = np.argmax(predictions, axis=1)
        # counts = np.bincount(beat_classes)
        # predicted_index = np.argmax(counts)

        return {
            "status": "success",
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
        # G. Bersih-bersih folder temp (Wajib agar server tidak penuh)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)