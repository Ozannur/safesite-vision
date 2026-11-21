"""ambil_foto_negatif.py - ambil foto otomatis dari webcam sebagai data NEGATIF.

Foto langsung disimpan ke Dataset Utama/train/images dengan label kosong
(artinya: "di foto ini tidak ada APD").

PENTING sebelum menjalankan:
- Lepas kacamata (kacamata itu salah satu kelas APD, kalau dipakai di foto
  dengan label kosong, model belajar bahwa kacamata bukan kacamata).
- Jangan pakai rompi, helm, sarung tangan, atau sepatu APD.
- Tutup Google Meet/Zoom dan aplikasi lain yang memakai kamera.
"""
import os
import time

import cv2

# ========== UBAH BAGIAN INI KALAU PERLU ==========
DST = "Dataset Utama"   # folder dataset utama
JUMLAH = 100            # jumlah foto yang diambil
JEDA = 0.8              # detik antar foto
TUNDA = 8               # hitung mundur sebelum mulai (detik)
# =================================================

img_dir = os.path.join(DST, "train", "images")
lbl_dir = os.path.join(DST, "train", "labels")
assert os.path.isdir(img_dir) and os.path.isdir(lbl_dir), (
    f"Folder {img_dir} atau {lbl_dir} tidak ada. Cek nama folder DST.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Kamera tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, lalu coba lagi.")

for _ in range(15):  # pemanasan kamera (supaya gambar tidak gelap/blur di awal)
    cap.read()

print("Lepas kacamata. Ubah posisi, jarak, sudut, dan baju selama pengambilan foto.")
for s in range(TUNDA, 0, -1):
    print(f"Mulai dalam {s} detik...", end="\r")
    time.sleep(1)
print(" " * 40)

tag = time.strftime("%H%M%S")
tersimpan = 0
for i in range(JUMLAH):
    for _ in range(4):  # buang frame lama di buffer kamera
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        print("\nGagal membaca kamera, berhenti.")
        break
    nama = f"neg_cam_{tag}_{i:03d}"
    cv2.imwrite(os.path.join(img_dir, nama + ".jpg"), frame)
    open(os.path.join(lbl_dir, nama + ".txt"), "w").close()  # label kosong
    tersimpan += 1
    print(f"Tersimpan {tersimpan}/{JUMLAH}", end="\r")
    time.sleep(JEDA)

cap.release()
print(f"\nSelesai. {tersimpan} foto negatif ditambahkan ke {img_dir}")
