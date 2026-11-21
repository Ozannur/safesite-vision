"""detector.py - memuat model YOLO dan menjalankan deteksi APD.
 
Aplikasi (app.py) hanya memanggil fungsi analisis(); semua logika model ada di sini.
"""
import time
from collections import Counter, deque
 
import cv2
from ultralytics import YOLO
 
# APD yang WAJIB ada agar dinyatakan "PATUH".
# Nama harus sama dengan nama kelas di model (cek lewat model.names).
APD_WAJIB = ["helmet", "vest"]
 
WARNA_BGR = {"helmet": (0, 169, 242), "vest": (44, 107, 255), "glasses": (255, 162, 63),
             "glove": (255, 124, 176), "boot": (176, 214, 30)}
 
 
def muat_model(path="best.pt"):
    return YOLO(path)
 
 
def muat_model_orang():
    """Model COCO bawaan untuk mendeteksi orang (diunduh otomatis saat pertama dipakai)."""
    return YOLO("yolov8n.pt")
 
 
def cek_kepatuhan(terdeteksi, wajib=APD_WAJIB):
    """Mengembalikan (status, daftar APD wajib yang tidak terdeteksi)."""
    kurang = [apd for apd in wajib if apd not in terdeteksi]
    return ("PATUH" if not kurang else "TIDAK PATUH"), kurang
 
 
class Penstabil:
    """Konfirmasi temporal untuk kamera live: sebuah kelas dianggap terdeteksi hanya jika
    muncul di minimal `min_hits` dari `jendela` frame terakhir. Deteksi acak sesaat dibuang."""
 
    def __init__(self, jendela=8, min_hits=5):
        self.buf = deque(maxlen=jendela)
        self.min_hits = min_hits
 
    def perbarui(self, kelas_frame):
        self.buf.append(set(kelas_frame))
        semua = set().union(*self.buf)
        return {k for k in semua if sum(k in s for s in self.buf) >= self.min_hits}
 
 
def _kotak_orang(model_orang, gambar, imgsz):
    r = model_orang.predict(gambar, conf=0.4, imgsz=imgsz, classes=[0], verbose=False)[0]
    return [[float(v) for v in b.xyxy[0]] for b in r.boxes]
 
 
def _di_dalam(cx, cy, orang):
    for x1, y1, x2, y2 in orang:
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
            return True
    return False
 
 
def analisis(model, gambar_bgr, conf=0.5, imgsz=640, model_orang=None, penstabil=None):
    """Jalankan deteksi pada satu gambar (array BGR dari OpenCV).
 
    Jika model_orang diberikan, hanya APD yang berada di dalam kotak orang yang dihitung
    (mengurangi deteksi palsu di latar belakang). agnostic_nms mencegah satu kotak
    diberi dua nama kelas sekaligus.
    Return dict: gambar_hasil (RGB), jumlah, deteksi, jumlah_orang (None jika filter mati), waktu_ms.
    """
    t0 = time.time()
    hasil = model.predict(gambar_bgr, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
    orang = _kotak_orang(model_orang, gambar_bgr, imgsz) if model_orang is not None else None
    waktu_ms = (time.time() - t0) * 1000
 
    out = gambar_bgr.copy()
    if orang is not None:
        for x1, y1, x2, y2 in orang:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
 
    kandidat = []
    for kotak in hasil.boxes:
        x1, y1, x2, y2 = [float(v) for v in kotak.xyxy[0]]
        nama = model.names[int(kotak.cls)].lower()
        if orang is not None and not _di_dalam((x1 + x2) / 2, (y1 + y2) / 2, orang):
            continue
        kandidat.append((nama, float(kotak.conf), x1, y1, x2, y2))
    if penstabil is not None:  # hanya kelas yang konsisten muncul beberapa frame yang dipakai
        terkonfirmasi = penstabil.perbarui({k[0] for k in kandidat})
        kandidat = [k for k in kandidat if k[0] in terkonfirmasi]
 
    deteksi = []
    for nama, skor, x1, y1, x2, y2 in kandidat:
        deteksi.append({"kelas": nama, "confidence": round(skor, 3)})
        warna = WARNA_BGR.get(nama, (255, 255, 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
        teks = f"{nama} {skor:.2f}"
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(int(y1) - th - 6, 0)
        cv2.rectangle(out, (int(x1), ytop), (int(x1) + tw + 6, ytop + th + 6), warna, -1)
        cv2.putText(out, teks, (int(x1) + 3, ytop + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
 
    jumlah = dict(Counter(d["kelas"] for d in deteksi))
    status, kurang = cek_kepatuhan(set(jumlah))
    return {
        "gambar_hasil": out[:, :, ::-1],  # BGR -> RGB untuk Streamlit
        "jumlah": jumlah,
        "deteksi": deteksi,
        "jumlah_orang": None if orang is None else len(orang),
        "status": status,
        "kurang": kurang,
        "waktu_ms": round(waktu_ms, 1),
    }
 