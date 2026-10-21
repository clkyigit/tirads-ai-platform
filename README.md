# 🩺 TI-RADS AI Platform

GE LOGIQ P9 (veya herhangi bir USG) görüntüsünü **USB capture card** ile alıp tarayıcıda gösteren, **UNet++** ile nodülü segmente eden, ölçen ve **ACR TI-RADS** puanını boyut eşikleriyle birlikte **radyolog onayına** sunan çapraz platform (Linux / Windows) karar destek yazılımı.

> ⚠️ **Araştırma amaçlıdır.** Tıbbi cihaz olarak onaylı değildir. Tüm kontur, ölçüm ve kategoriler radyolog tarafından kontrol edilip onaylanmadan kaydedilemez.

## Ne yapar?

| Adım | Açıklama |
|---|---|
| Canlı görüntü | Capture card → tarayıcı (MJPEG akış). Windows: DirectShow/MSMF, Linux: V4L2 |
| Dondur | Kayıpsız (PNG) tam çözünürlüklü kare alınır; gizlilik maskeleri uygulanır |
| Segmentasyon | Nodülün üstüne tıkla → UNet++ (.pth veya .onnx) → tıklanan nodülün konturu |
| Ölçüm | Transvers genişlik + AP (transvers görüntüden), uzunluk (longitudinal görüntüden); kumpasla düzeltilebilir |
| Kalibrasyon | Ekrandaki derinlik skalasından iki noktayla mm/piksel; derinlik ayarlarına göre ön ayar kaydı. DICOM yüklenirse piksel aralığı otomatik önerilir |
| TI-RADS | 5 kategori (ekojenik odaklar çoklu seçim, toplanır); AP/genişlik oranından şekil önerisi |
| Öneri | ACR eşikleri: TR3 İİAB ≥2,5 cm / takip ≥1,5 cm · TR4 ≥1,5 / ≥1 cm · TR5 ≥1 / ≥0,5 cm |
| Kayıt | Radyolog onayıyla `data/cases/<vaka>/`: görüntü, maske, overlay, `case.json`, `report.txt` |

## Kurulum ve çalıştırma

Gerekli: **Python 3.10+**. İlk çalıştırmada sanal ortam (`.venv`) otomatik kurulur.

**Windows**
```bat
run.bat            :: normal
run.bat --demo     :: donanım ve model olmadan deneme
run.bat --ml       :: PyTorch (.pth) desteğini de kur
```

**Linux**
```bash
./run.sh           # normal
./run.sh --demo    # donanım ve model olmadan deneme
./run.sh --ml      # PyTorch (.pth) desteğini de kur
```
Linux'ta kullanıcı `video` grubunda olmalı: `sudo usermod -aG video $USER` (sonra oturumu yeniden açın).

Tarayıcı otomatik açılır: <http://127.0.0.1:8765>. Varsayılan olarak yalnızca bu bilgisayardan erişilir. Ağdan erişim için `--host 0.0.0.0` kullanın; bu durumda KVKK ve hastane ağ politikasını göz önünde bulundurun.

**GPU:** NVIDIA kart kullanacaksanız `--ml`'den önce torch'u <https://pytorch.org> adresindeki CUDA komutuyla kurun. ONNX modelleri için GPU'da `onnxruntime-gpu` (NVIDIA) veya `onnxruntime-directml` (Windows) kurulabilir.

## İlk kurulum (bir kez)

1. **Ayarlar → Görüntü kaynağı:** "Capture card" seçin, **Tara** ile cihaz numarasını bulun. Format olarak **YUY2/NV12** (sıkıştırmasız) seçin. MJPG, ince ekojenik odakları bozabilir.
2. Görüntüyü **dondurun** (Space).
3. **Ayarlar → Ekran alanları:**
   - *Görüntü alanını (ROI) çiz:* yalnızca USG görüntüsünü kapsayın. Menüler ve yazılar dışarıda kalsın; model sadece bu alanı görür.
   - *Gizlilik maskesi ekle:* hasta adı ve ID şeridini seçin. Bu alan kayıtlarda ve akışta siyaha boyanır.
4. **Ayarlar → Kalibrasyon:** "Skaladan kalibre et" → derinlik skalasında iki çentiğe tıklayın → mesafeyi (mm) girin → ön ayar olarak kaydedin (ör. "4 cm derinlik"). **Cihazda derinlik veya zoom değişince ilgili ön ayarı seçin.**
5. **Ayarlar → Segmentasyon modeli:** ağırlık dosyasını `models/` altına koyun ve parametreleri eğitimdekiyle aynı yapın.

## Muayene akışı

1. Transvers kesitte nodülü bulun → **Space** (dondur) → nodüle tıklayın (AI). Kontur, genişlik (G) ve AP otomatik çizilir.
2. **2** → longitudinal sekmesi → Space → nodüle tıklayın → uzunluk (U) çizilir.
3. Konturu beğenmezseniz **D** ile elle çizin, ölçümleri **W / H / L** kumpaslarıyla düzeltin ya da mm değerlerini yazın.
4. TI-RADS formunu doldurun. Şekil, AP/G oranından önerilir; değiştirebilirsiniz.
5. Onay kutusunu işaretleyip **Onayla ve kaydet**.

| Kısayol | İşlev |
|---|---|
| Space | Dondur / canlıya dön |
| 1 / 2 | Transvers / Longitudinal |
| A | AI segmentasyon (nodüle tıkla) |
| W / H / L | Genişlik / AP / Uzunluk kumpası |
| D | Elle kontur (Enter veya ilk noktaya tıklayarak kapat) |
| Esc | Aracı bırak |

## Model (UNet++)

Desteklenen formatlar:

- **`.pth` (PyTorch):** `segmentation_models_pytorch.UnetPlusPlus` için `state_dict`. Ayarlarda `encoder`, `in_channels`, `classes` değerleri eğitimdekiyle aynı olmalıdır. `module.` önekleri ve `{"state_dict": ...}` sarmalı otomatik açılır. Güvenlik için yalnızca ağırlıklar yüklenir (`weights_only=True`); `torch.save(model)` ile kaydedilmiş tam modeller yerine `torch.save(model.state_dict(), "unetpp.pth")` kullanın.
- **`.onnx`:** Giriş `N×C×H×W` float32, çıkış `N×K×H×W` (logit veya olasılık). Giriş boyutu ve kanal sayısı modelden okunur. Torch kurulmadan çalışır ve kurulumu en kolay seçenektir:

```python
import torch
model.eval()
torch.onnx.export(model, torch.randn(1, 3, 512, 512), "models/unetpp.onnx",
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "N"}, "logits": {0: "N"}}, opset_version=17)
```

Ön işleme ayarları (`normalize`: imagenet / zero_one / zscore, `input_size`, `threshold`) **eğitimdeki ön işlemeyle birebir aynı** olmalıdır; aksi halde maskeler bozulur. `classes > 1` ise `nodule_class` ile nodül kanalını seçin.

Model dosyası yoksa uygulama yine çalışır; konturlar elle çizilir. `--demo` modundaki segmentasyon basit bir eşiklemedir ve **klinik değildir**.

## Klasör yapısı

```
backend/
  server.py        HTTP API + statik dosyalar (Starlette/uvicorn)
  capture.py       Capture card okuma thread'i, cihaz tarama, demo kaynak
  segmentation.py  UNet++ (.pth / .onnx) çıkarımı, ROI kırpma
  measure.py       Maske → kontur, G/AP/en uzun çap, kalibrasyon
  tirads.py        ACR TI-RADS puanlama ve boyut eşikleri (tek doğruluk kaynağı)
  storage.py       Vaka kaydı ve rapor
  config.py        data/config.json
frontend/          index.html, app.js, style.css (internet gerektirmez)
models/            Model ağırlıkları (git'e eklenmez)
data/              Ayarlar ve vakalar (git'e eklenmez)
tests/             python -m unittest discover -s tests -t .
```

## Veri ve KVKK

- Hasta adı veya TC kimlik no girilmez; yalnızca çalışma/protokol no (takma ad) tutulur.
- Kaydedilen görüntülerde gizlilik maskeleri uygulanmıştır. Maskelerin doğru konumda olduğunu ilk kurulumda kontrol edin.
- Etik kurul onayı olmadan veri toplamayın. Klinik kullanım için tıbbi cihaz yazılımı mevzuatı (MDR / TİTCK) gerekir.

## Test

```bash
python -m unittest discover -s tests -t . -v
```
TI-RADS puan ve eşikleri, ölçüm geometrisi ve uçtan uca API akışı (demo kaynakla dondur → segmente et → kaydet) test edilir.
