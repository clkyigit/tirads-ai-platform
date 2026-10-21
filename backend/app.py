import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="TI-RADS AI Platform", page_icon="🩺", layout="wide"
)

st.title("🩺 Ultrason Yapay Zeka & TI-RADS Karar Destek Sistemi")
st.sidebar.header("Kontrol Paneli")

# Görüntü Kaynağı Seçimi
camera_index = st.sidebar.selectbox(
    "Görüntü Kaynağı (Capture Card)", [0, 1, 2], index=0
)
run_camera = st.sidebar.checkbox("Canlı Akışı Başlat", value=False)

# Arayüz Düzeni: Sol taraf Video, Sağ taraf Analiz ve Skorlama
col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("Canlı USG Ekranı (GE LOGIQ P9)")
  video_placeholder = st.empty()

with col2:
  st.subheader("ACR TI-RADS & nnU-Net++ Analizi")

  # 1. Kompozisyon Puanlama
  composition = st.selectbox(
      "1. Kompozisyon",
      [
          ("Kistik veya neredeyse tamamen kistik (0 Puan)", 0),
          ("Spongiform / Süngerimsi (0 Puan)", 0),
          ("Karışık kistik ve solid (1 Puan)", 1),
          ("Solid veya neredeyse tamamen solid (2 Puan)", 2),
      ],
      format_func=lambda x: x[0],
  )

  # 2. Ekojenite Puanlama
  echogenicity = st.selectbox(
      "2. Ekojenite",
      [
          ("Aneusöik (0 Puan)", 0),
          ("İzookoik veya Hiperöik (1 Puan)", 1),
          ("Hipoöik (2 Puan)", 2),
          ("Çok Belirgin Hipoöik (3 Puan)", 3),
      ],
      format_func=lambda x: x[0],
  )

  # 3. Şekil Puanlama
  shape = st.selectbox(
      "3. Şekil",
      [
          ("Genişliği boyundan uzun / Enine (0 Puan)", 0),
          ("Boyu genişliğinden uzun / Dik (3 Puan)", 3),
      ],
      format_func=lambda x: x[0],
  )

  # 4. Kenar (Margin) Puanlama
  margin = st.selectbox(
      "4. Kenar Özelliği",
      [
          ("Düz / Pürüzsüz (0 Puan)", 0),
          ("İyi sınırlı ancak lobüle / irregüler (2 Puan)", 2),
          ("Makrokalsifikasyon içeriyor (2 Puan)", 2),
          ("Ekstratiroidal yayılım / Spiküle kenar (3 Puan)", 3),
      ],
      format_func=lambda x: x[0],
  )

  # 5. Ekojenik Odaklar (Foci) Puanlama
  foci = st.selectbox(
      "5. Ekojenik Odaklar",
      [
          ("Yok veya Kuyruklu Yıldız Artefaktı (0 Puan)", 0),
          ("Makrokalsifikasyonlar (1 Puan)", 1),
          ("Periferal (kenar) kalsifikasyonlar (2 Puan)", 2),
          ("Nokta şeklinde ekojenik odaklar / Mikrokalsifikasyon (3 Puan)", 3),
      ],
      format_func=lambda x: x[0],
  )

  # Toplam Puan ve TI-RADS Kategorisi Hesaplama
  total_score = (
      composition[1]
      + echogenicity[1]
      + shape[1]
      + margin[1]
      + foci[1]
  )


  def get_tirads_category(score):
    if score == 0:
      return "TR1 (İyi huylu - Takip gerekmez)"
    elif score in [1, 2]:
      return "TR2 (Düşük şüpheli - Takip gerekmez)"
    elif score == 3:
      return "TR3 (Hafif şüpheli - İnceleme/İğne biyopsisi kriterlerine göre)"
    elif score in [4, 5, 6]:
      return "TR4 (Orta derecede şüpheli - Boyut eşiğine göre biyopsi)"
    else:
      return "TR5 (Yüksek şüpheli - Biyopsi önerilir)"


  st.markdown("---")
  st.metric(label="Hesaplanan Toplam TI-RADS Puanı", value=total_score)
  st.success(f"**Öneri Kategori:** {get_tirads_category(total_score)}")

  st.markdown("---")
  if st.button("🤖 nnU-Net++ Otomatik Segmentasyonu Çalıştır"):
    st.info(
        "Model ağırlıkları yükleniyor, ROI bölgesi segmentasyonu yapılıyor..."
    )
    # nnU-Net++ inference ve maske çizdirme kodu buraya entegre edilecek

# Canlı Video Akış Döngüsü
if run_camera:
  cap = cv2.VideoCapture(camera_index)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  while run_camera:
    ret, frame = cap.read()
    if not ret:
      st.error(
          "Capture card görüntüsü alınamadı! Bağlantıyı veya indeks numarasını"
          " kontrol edin."
      )
      break

    # OpenCV BGR -> RGB Dönüşümü
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(
        frame_rgb, channels="RGB", use_container_width=True
    )

  cap.release()
else:
  video_placeholder.info(
      "Canlı akışı başlatmak için sol menüdeki kutucuğu işaretleyin."
  )
