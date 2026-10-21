import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="TI-RADS AI Platform", page_icon="🩺", layout="wide")

st.title("🩺 Ultrason Yapay Zeka & TI-RADS Karar Destek Sistemi")
st.sidebar.header("Kontrol Paneli")

camera_index = st.sidebar.selectbox("Görüntü Kaynağı (Capture Card)", [0, 1, 2], index=0)
run_camera = st.sidebar.checkbox("Canlı Akışı Başlat", value=False)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Canlı USG Ekranı")
    video_placeholder = st.empty()

with col2:
    st.subheader("TI-RADS & nnU-Net++ Analizi")
    nodule_type = st.selectbox(
        "Nodül Kompozisyonu",
        ["Seçiniz...", "Kistik / Tamamen Kistik", "Süngerimsi (Spongiform)", "Mix (Kistik/Solid)", "Solid"]
    )
    echogenicity = st.selectbox(
        "Ekojenite",
        ["Seçiniz...", "Aneusöik", "İzookoik / Hiperöik", "Hipoöik", "Belirgin Hipoöik"]
    )
    
    if st.button("nnU-Net++ Segmentasyonunu Çalıştır"):
        st.info("Model ağırlıkları yükleniyor ve ROI analizi yapılıyor...")

if run_camera:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Capture card görüntüsü alınamadı! Bağlantıyı veya indeks numarasını kontrol edin.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()
else:
    video_placeholder.info("Canlı akışı başlatmak için sol menüdeki kutucuğu işaretleyin.")
