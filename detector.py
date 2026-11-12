"""detector.py - memuat model YOLO dan menjalankan deteksi APD.

Aplikasi (app.py) hanya memanggil fungsi analisis(); semua logika model ada di sini.
"""
import time
from collections import Counter

from ultralytics import YOLO

# APD yang WAJIB ada agar dinyatakan "PATUH".
# GANTI sesuai aturan cek_apd di notebook Colab kalian.
# Nama harus sama dengan nama kelas di model (cek lewat model.names).
APD_WAJIB = ["helmet", "vest"]


def muat_model(path="best.pt"):
    return YOLO(path)


def cek_kepatuhan(terdeteksi, wajib=APD_WAJIB):
    """Mengembalikan (status, daftar APD wajib yang tidak terdeteksi)."""
    kurang = [apd for apd in wajib if apd not in terdeteksi]
    return ("PATUH" if not kurang else "TIDAK PATUH"), kurang


def analisis(model, gambar_bgr, conf=0.25, imgsz=640):
    """Jalankan deteksi pada satu gambar (array BGR dari OpenCV).

    Return dict: gambar_hasil (RGB), jumlah (per kelas), deteksi (list),
    status, kurang, waktu_ms.
    """
    t0 = time.time()
    hasil = model.predict(gambar_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    waktu_ms = (time.time() - t0) * 1000

    deteksi = []
    for kotak in hasil.boxes:
        nama = model.names[int(kotak.cls)].lower()
        deteksi.append({"kelas": nama, "confidence": round(float(kotak.conf), 3)})

    jumlah = dict(Counter(d["kelas"] for d in deteksi))
    status, kurang = cek_kepatuhan(set(jumlah))

    return {
        "gambar_hasil": hasil.plot()[:, :, ::-1],  # BGR -> RGB untuk Streamlit
        "jumlah": jumlah,
        "deteksi": deteksi,
        "status": status,
        "kurang": kurang,
        "waktu_ms": round(waktu_ms, 1),
    }
