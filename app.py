"""app.py - SafeSite Vision: dashboard deteksi APD (tampilan premium + animasi).
Jalankan:  streamlit run app.py
"""
import base64
import time
from collections import deque
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from detector import muat_model, analisis, cek_kepatuhan, APD_WAJIB

import pathlib
_cfg = pathlib.Path(".streamlit/config.toml")
if not _cfg.exists():
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.write_text('[theme]\nbase = "dark"\nprimaryColor = "#F2A900"\nbackgroundColor = "#0A1220"\n'
                    'secondaryBackgroundColor = "#16263D"\ntextColor = "#EAF0F7"\nfont = "sans serif"\n')

st.set_page_config(page_title="SafeSite Vision", page_icon="🦺", layout="wide")

WARNA = {"helmet": "#F2A900", "vest": "#FF6B2C", "glasses": "#3FA2FF", "glove": "#B07CFF", "boot": "#1ED6B0"}
IKON = {"helmet": "🪖", "vest": "🦺", "glasses": "👓", "glove": "🧤", "boot": "🥾"}
HIJAU, MERAH = "#22E58B", "#FF4D5E"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&family=Orbitron:wght@700;900&display=swap');
html, body, [class*="css"], .stApp {font-family:'Poppins',sans-serif;}
.stApp {background: radial-gradient(circle at 12% 8%, #1c3258 0%, transparent 42%),
        radial-gradient(circle at 88% 92%, #4d2310 0%, transparent 42%), #0a1220;}
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.4rem; max-width:1280px;}
section[data-testid="stSidebar"] {background:rgba(10,18,32,.88); backdrop-filter:blur(12px);
        border-right:1px solid rgba(255,255,255,.08);}
@property --n {syntax:'<integer>'; initial-value:0; inherits:false;}
@keyframes fadeUp {from{opacity:0; transform:translateY(26px)} to{opacity:1; transform:none}}
@keyframes shift {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes shine {from{transform:translateX(-130%) skewX(-20deg)} to{transform:translateX(420%) skewX(-20deg)}}
@keyframes pulse {0%{box-shadow:0 0 0 0 var(--glow)} 70%{box-shadow:0 0 0 24px transparent} 100%{box-shadow:0 0 0 0 transparent}}
@keyframes scan {from{top:0} to{top:calc(100% - 3px)}}
@keyframes grow {from{width:0}}
@keyframes count {from{--n:0}}
@keyframes float {0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)}}
.hero {position:relative; overflow:hidden; padding:32px 40px; border-radius:24px; margin-bottom:24px;
       background:linear-gradient(120deg,#ff6b2c,#f2a900,#ff9a4d,#ff6b2c); background-size:300% 300%;
       animation:shift 8s ease infinite, fadeUp .8s both; box-shadow:0 24px 60px rgba(255,107,44,.28);}
.hero:after {content:''; position:absolute; top:0; left:0; width:22%; height:100%;
       background:linear-gradient(100deg,transparent,rgba(255,255,255,.5),transparent); animation:shine 5s infinite;}
.hero h1 {margin:0; color:#0a1220; font-weight:800; font-size:2.4rem; letter-spacing:-.5px;}
.hero p {margin:6px 0 0; color:#0a1220; font-weight:500; opacity:.85;}
.hero .em {position:absolute; right:40px; top:18px; font-size:4.2rem; animation:float 3.2s ease-in-out infinite;}
.glass {background:rgba(255,255,255,.055); backdrop-filter:blur(14px); border:1px solid rgba(255,255,255,.11);
        border-radius:18px; padding:18px; animation:fadeUp .7s both; transition:all .3s ease;}
.glass:hover {transform:translateY(-7px); border-color:var(--c); box-shadow:0 16px 36px rgba(0,0,0,.45), 0 0 26px -6px var(--c);}
.stat {text-align:center;}
.stat .ic {font-size:2rem;} .stat .nm {color:#aab6c6; font-size:.8rem; letter-spacing:1.5px; text-transform:uppercase;}
.cnt {counter-reset:num var(--n); animation:count 1.4s ease-out; font-weight:800; font-size:2.4rem; color:var(--c);}
.cnt::after {content:counter(num);}
.badge {--glow:var(--g); padding:22px; border-radius:20px; text-align:center; font-size:1.9rem; font-weight:800;
        color:#fff; margin-bottom:14px; animation:pulse 1.9s infinite, fadeUp .6s both;}
.hud {position:relative; border-radius:20px; overflow:hidden; border:1px solid rgba(255,255,255,.18);
      box-shadow:0 18px 50px rgba(0,0,0,.5); animation:fadeUp .7s both;}
.hud img {width:100%; display:block;}
.hud:after {content:''; position:absolute; left:0; right:0; height:3px; top:0;
      background:linear-gradient(90deg,transparent,#22e58b,transparent); box-shadow:0 0 20px #22e58b;
      animation:scan 2.6s ease-in-out infinite alternate;}
.tag {position:absolute; top:12px; left:14px; z-index:2; background:rgba(10,18,32,.75); color:#22e58b;
      padding:4px 12px; border-radius:20px; font-size:.75rem; font-weight:600; letter-spacing:1px;}
.row {display:flex; align-items:center; gap:10px; margin:9px 0; color:#dfe7f1; font-size:.9rem;}
.bar {flex:1; height:9px; border-radius:8px; background:rgba(255,255,255,.1); overflow:hidden;}
.bar i {display:block; height:100%; border-radius:8px; animation:grow 1.1s ease-out;}
.sec {color:#fff; font-weight:600; font-size:1.15rem; margin:22px 0 8px; animation:fadeUp .6s both;}
button[data-baseweb="tab"] {font-weight:600; font-size:1rem;} button[data-baseweb="tab"]:hover {color:#f2a900;}
div[data-testid="stFileUploader"] section {border:2px dashed #f2a900; border-radius:18px; background:rgba(242,169,0,.06); transition:all .3s;}
div[data-testid="stFileUploader"] section:hover {box-shadow:0 0 28px rgba(242,169,0,.4); transform:scale(1.01);}

header[data-testid="stHeader"] {background:transparent;}
.stAppDeployButton {display:none;}
.block-container {padding-top:3.2rem;}
.hero {text-align:center; padding:30px 40px 28px;}
.hero .em {position:static; display:block; font-size:3.4rem; margin-bottom:2px;}
.ttl {font-family:'Orbitron',sans-serif; font-weight:900; font-size:2.9rem; letter-spacing:7px; color:#0a1220;
      text-shadow:0 3px 0 rgba(255,255,255,.35); animation:fadeUp .9s both; position:relative; z-index:1;}
.sub {font-weight:600; font-size:1.08rem; color:#1a1208; margin-top:6px; position:relative; z-index:1;}
.chip {display:inline-block; margin:14px 5px 0; padding:6px 16px; border-radius:22px; background:rgba(10,18,32,.88);
       color:#ffd36b; font-size:.8rem; font-weight:600; letter-spacing:.6px; position:relative; z-index:1; animation:fadeUp 1s both;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none;}
div[data-baseweb="tab-list"] {gap:12px; padding:8px 4px;}
button[data-baseweb="tab"] {border:1.5px solid rgba(255,255,255,.28); border-radius:14px; padding:10px 24px; height:auto;
       background:rgba(255,255,255,.07); color:#eaf0f7; transition:all .3s;}
button[data-baseweb="tab"]:hover {border-color:#f2a900; transform:translateY(-3px); box-shadow:0 8px 20px rgba(0,0,0,.4);}
button[data-baseweb="tab"][aria-selected="true"] {border-color:#f2a900; color:#fff; box-shadow:0 0 20px rgba(242,169,0,.45);
       background:linear-gradient(135deg,rgba(242,169,0,.38),rgba(255,107,44,.38));}
@keyframes slideIn {from{transform:translateX(-70px); opacity:0} to{transform:none; opacity:1}}
section[data-testid="stSidebar"] {animation:slideIn .8s cubic-bezier(.22,1,.36,1) both;
       transition:transform .55s cubic-bezier(.22,1,.36,1), width .55s cubic-bezier(.22,1,.36,1),
                  min-width .55s cubic-bezier(.22,1,.36,1), margin-left .55s cubic-bezier(.22,1,.36,1) !important;}
section[data-testid="stSidebar"][aria-expanded="true"] > div {animation:fadeUp .6s both;}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] button {
       background:linear-gradient(135deg,#ff6b2c,#f2a900) !important; color:#0a1220 !important; border-radius:50% !important;
       box-shadow:0 0 18px rgba(242,169,0,.65); transition:transform .3s;}
[data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stExpandSidebarButton"]:hover {transform:scale(1.18) rotate(10deg);}
.pill {display:inline-block; padding:6px 16px; border-radius:20px; font-weight:700; font-size:.85rem; letter-spacing:1px;}
.pill.on {background:rgba(34,229,139,.18); color:#22e58b; border:1px solid #22e58b; --glow:rgba(34,229,139,.5); animation:pulse 1.9s infinite;}
.pill.off {background:rgba(255,77,94,.15); color:#ff4d5e; border:1px solid #ff4d5e;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="em">🦺</div><div class="ttl">SAFESITE VISION</div>
<div class="sub">Deteksi Alat Pelindung Diri pada Pekerja Konstruksi dengan YOLO</div>
<span class="chip">⚡ Real-time compliance check</span><span class="chip">🧠 YOLOv8</span><span class="chip">🦺 5 kelas APD</span></div>""",
            unsafe_allow_html=True)


@st.cache_resource
def get_model(path):
    return muat_model(path)


def b64(rgb):
    _, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()


def kartu_stat(label, ikon, nilai, warna, delay=0):
    return (f'<div class="glass stat" style="--c:{warna}; animation-delay:{delay}s">'
            f'<div class="ic">{ikon}</div><div class="cnt" style="--n:{nilai}"></div>'
            f'<div class="nm">{label}</div></div>')


with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Klik tombol « di pojok panel untuk menutup")
    model_path = st.text_input("File model", "best.pt")
    conf = st.slider("Confidence minimum", 0.05, 0.90, 0.25, 0.05)
    wajib = st.multiselect("APD wajib", list(WARNA), default=APD_WAJIB)
    imgsz = st.select_slider("Ukuran proses kamera live (kecil = lebih cepat)", options=[320, 416, 512, 640], value=416)
    try:
        model = get_model(model_path)
        st.success("Model aktif: " + model_path)
    except Exception as e:
        st.error(f"Model gagal dimuat. Pastikan '{model_path}' ada di folder proyek.\n\n{e}")
        st.stop()

if "riwayat" not in st.session_state:
    st.session_state.riwayat = {}


def tampilkan(gambar, nama):
    with st.spinner("AI sedang menganalisis gambar..."):
        hasil = analisis(model, gambar, conf)
    status, kurang = cek_kepatuhan(set(hasil["jumlah"]), wajib)
    st.session_state.riwayat[f"{nama}|{conf}|{','.join(sorted(wajib))}"] = {
        "file": nama, "status": status, "jumlah APD": sum(hasil["jumlah"].values()),
        "waktu (ms)": hasil["waktu_ms"]}

    kiri, kanan = st.columns([3, 2], gap="large")
    with kiri:
        st.markdown(f'<div class="hud"><span class="tag">● AI VISION ACTIVE</span>'
                    f'<img src="data:image/jpeg;base64,{b64(hasil["gambar_hasil"])}"></div>', unsafe_allow_html=True)
    with kanan:
        ok = status == "PATUH"
        st.markdown(f'<div class="badge" style="background:linear-gradient(135deg,{HIJAU if ok else MERAH},'
                    f'{"#0fb36b" if ok else "#c81e3a"}); --g:{"rgba(34,229,139,.6)" if ok else "rgba(255,77,94,.6)"}">'
                    f'{"✅ PATUH" if ok else "⚠️ TIDAK PATUH"}</div>', unsafe_allow_html=True)
        if wajib:
            t = len(wajib) - len(kurang)
            st.markdown(f'<div class="glass" style="--c:#f2a900"><div style="color:#aab6c6;font-size:.85rem">'
                        f'APD wajib terpenuhi: <b style="color:#fff">{t}/{len(wajib)}</b></div>'
                        f'<div class="bar" style="margin-top:8px"><i style="width:{t / len(wajib) * 100}%;'
                        f'background:linear-gradient(90deg,#ff6b2c,#f2a900)"></i></div></div>', unsafe_allow_html=True)
        if kurang:
            st.markdown("**Belum terdeteksi:** " + ", ".join(f"{IKON.get(k, '')} {k}" for k in kurang))
        st.markdown('<div class="sec">Deteksi teratas</div>', unsafe_allow_html=True)
        for d in sorted(hasil["deteksi"], key=lambda x: -x["confidence"])[:6]:
            w = WARNA.get(d["kelas"], "#fff")
            st.markdown(f'<div class="row"><span style="width:90px">{IKON.get(d["kelas"], "")} {d["kelas"]}</span>'
                        f'<div class="bar"><i style="width:{d["confidence"] * 100}%;background:{w}"></i></div>'
                        f'<b>{d["confidence"]:.0%}</b></div>', unsafe_allow_html=True)
        st.caption(f"⚡ {hasil['waktu_ms']} ms  (~{1000 / max(hasil['waktu_ms'], 1):.1f} FPS)")

    st.markdown('<div class="sec">Jumlah per kelas APD</div>', unsafe_allow_html=True)
    kol = st.columns(5)
    for i, (k, c) in enumerate(zip(WARNA, kol)):
        c.markdown(kartu_stat(k, IKON[k], hasil["jumlah"].get(k, 0), WARNA[k], i * 0.12), unsafe_allow_html=True)
    return status


tab1, tab2, tab3, tab4 = st.tabs(["🔍 Deteksi", "📷 Kamera", "📊 Statistik", "ℹ️ Tentang"])

with tab1:
    files = st.file_uploader("Seret & lepas satu atau beberapa gambar pekerja di sini",
                             type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("Unggah satu atau beberapa gambar untuk memulai analisis (disarankan maksimal 10 sekaligus).")
    else:
        ringkas = st.container()
        hasil_status = []
        for i, file in enumerate(files):
            img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error(f"{file.name} tidak bisa dibaca.")
                continue
            with st.expander(f"🖼️ {i + 1}. {file.name}", expanded=(i == 0)):
                hasil_status.append(tampilkan(img, file.name))
        if hasil_status:
            p = hasil_status.count("PATUH")
            with ringkas:
                st.markdown('<div class="sec">Ringkasan seluruh gambar</div>', unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)
                r1.markdown(kartu_stat("Total gambar", "🖼️", len(hasil_status), "#3FA2FF"), unsafe_allow_html=True)
                r2.markdown(kartu_stat("Patuh", "✅", p, HIJAU, .12), unsafe_allow_html=True)
                r3.markdown(kartu_stat("Tidak patuh", "⚠️", len(hasil_status) - p, MERAH, .24), unsafe_allow_html=True)

with tab3:
    rw = pd.DataFrame(list(st.session_state.riwayat.values()))
    if rw.empty:
        st.info("Belum ada gambar yang diproses.")
    else:
        p = int((rw["status"] == "PATUH").sum())
        c1, c2, c3 = st.columns(3)
        c1.markdown(kartu_stat("Total gambar", "🖼️", len(rw), "#3FA2FF"), unsafe_allow_html=True)
        c2.markdown(kartu_stat("Patuh", "✅", p, HIJAU, .12), unsafe_allow_html=True)
        c3.markdown(kartu_stat("Tidak patuh", "⚠️", len(rw) - p, MERAH, .24), unsafe_allow_html=True)
        df = pd.DataFrame({"Status": ["PATUH", "TIDAK PATUH"], "Jumlah": [p, len(rw) - p]})
        donat = alt.Chart(df).mark_arc(innerRadius=70, cornerRadius=8).encode(
            theta="Jumlah", color=alt.Color("Status", scale=alt.Scale(domain=["PATUH", "TIDAK PATUH"], range=[HIJAU, MERAH]))
        ).properties(height=300).configure(background="transparent").configure_legend(
            labelColor="#cfd8e3", titleColor="#cfd8e3").configure_view(strokeWidth=0)
        st.altair_chart(donat, width="stretch")
        st.dataframe(rw, width="stretch")
        if st.button("🗑️ Hapus riwayat"):
            st.session_state.riwayat = {}
            st.rerun()

with tab4:
    st.markdown('<div class="glass" style="--c:#f2a900"><b>SafeSite Vision</b> mendeteksi 5 jenis APD '
                '(helm, rompi, sepatu, sarung tangan, kacamata) dengan YOLOv8 dan menilai kepatuhan berdasarkan '
                'APD wajib yang dipilih.<br><br><b>Dataset:</b> PPE_cdp (HCTML, Roboflow Universe, CC BY 4.0)<br>'
                '<b>Batasan:</b> penilaian per gambar, belum per pekerja.</div>', unsafe_allow_html=True)


# Tab kamera diletakkan paling akhir karena mode live berupa loop (tab lain tetap tampil lebih dulu)
with tab2:
    mode = st.radio("Mode kamera", ["🔴 Live real-time", "📸 Foto"], horizontal=True)
    k1, k2 = st.columns([2, 3])
    nyala = k1.toggle("📷 Nyalakan kamera", value=False, key="kamera_on")
    k2.markdown(f'<span class="pill {"on" if nyala else "off"}">● KAMERA {"AKTIF" if nyala else "MATI"}</span>',
                unsafe_allow_html=True)
    if not nyala:
        st.info("Kamera dimatikan agar laptop tetap ringan. Geser tombol di atas untuk menyalakan.")
    elif mode.startswith("📸"):
        foto = st.camera_input("Ambil foto dari kamera")
        if foto is not None:
            img = cv2.imdecode(np.frombuffer(foto.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            tampilkan(img, "kamera.jpg")
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Kamera tidak terdeteksi. Tutup aplikasi lain yang memakai kamera (Zoom/Meet), "
                     "cek izin kamera Windows, lalu coba lagi. Mode live hanya jalan saat web dijalankan di laptop yang punya kamera. Di web online, pakai mode 📸 Foto (memakai kamera perangkat kamu).")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            tampil, info = st.empty(), st.empty()
            riwayat_status, fps, t_prev = deque(maxlen=8), 0.0, time.time()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Gagal membaca gambar dari kamera.")
                        break
                    hasil = analisis(model, frame, conf, imgsz)
                    st_frame, kurang = cek_kepatuhan(set(hasil["jumlah"]), wajib)
                    riwayat_status.append(st_frame)
                    status = max(set(riwayat_status), key=list(riwayat_status).count)  # voting 8 frame, anti berkedip
                    now = time.time()
                    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                    t_prev = now
                    tampil.image(hasil["gambar_hasil"], channels="RGB", width="stretch", output_format="JPEG")
                    warna = HIJAU if status == "PATUH" else MERAH
                    ket = "" if status == "PATUH" else " | belum terdeteksi: " + ", ".join(kurang)
                    info.markdown(f'<div style="padding:12px 18px;border-radius:14px;background:{warna};color:#fff;'
                                  f'font-weight:700;font-size:1.2rem;text-align:center">'
                                  f'{"✅ PATUH" if status == "PATUH" else "⚠️ TIDAK PATUH"}{ket} &nbsp;•&nbsp; {fps:.1f} FPS</div>',
                                  unsafe_allow_html=True)
            finally:
                cap.release()
