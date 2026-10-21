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
Linux'ta capture card'a erişim: masaüstü oturumu açık kullanıcı çoğu dağıtımda (Ubuntu, Pardus) cihaza otomatik erişir. "açılamadı" hatası alırsanız kullanıcıyı `video` grubuna ekleyin: `sudo usermod -aG video $USER` (sonra oturumu yeniden açın).

**Masaüstü kısayolu** (bir kez):
```bash
./run.sh --kisayol                     # Linux: masaüstü + uygulama menüsü
./run.sh --kisayol --otomatik-baslat   # ayrıca oturum açılınca kendiliğinden başlasın
run.bat --kisayol [--otomatik-baslat]  # Windows
```
Sonrasında masaüstündeki **TI-RADS AI** simgesine çift tıklamanız yeterli. Program zaten açıksa ikinci tıklama yalnızca tarayıcıyı açar. Kaldırmak için: `--kisayol-kaldir`.

Tarayıcı otomatik açılır: <http://127.0.0.1:8765>. Varsayılan olarak yalnızca bu bilgisayardan erişilir. Ağdan erişim için `--host 0.0.0.0` kullanın; bu durumda KVKK ve hastane ağ politikasını göz önünde bulundurun.

**GPU:** NVIDIA kart kullanacaksanız `--ml`'den önce torch'u <https://pytorch.org> adresindeki CUDA komutuyla kurun. ONNX modelleri için GPU'da `onnxruntime-gpu` (NVIDIA) veya `onnxruntime-directml` (Windows) kurulabilir.

## İlk kurulum (bir kez)

1. **Capture card'ı takın; ayar gerekmez.** Program kartı adından tanır (dahili/USB webcam'leri seçmez). En iyi formatı kendisi dener: önce sıkıştırmasız YUY2/NV12, yeterli kare hızı yoksa MJPG; MJPG'ye düşerse üst şeritte sarı uyarı çıkar. Bulduğu ayarı hatırlar, sonraki takışta beklemeden açar. Kart çıkarılınca "Capture card bekleniyor" ekranı görünür, yeniden takılınca görüntü kendiliğinden gelir. USG'den görüntü gelmiyorsa "Sinyal yok" uyarısı çıkar.
   Yanlış cihaz seçilirse: **Ayarlar → Bulunan video cihazları** → doğru cihazda **Bunu kullan**, istenmeyende **Yok say**.
2. Görüntüyü **dondurun** (Space).
3. **Ayarlar → Ekran alanları:**
   - *Görüntü alanını (ROI) çiz:* yalnızca USG görüntüsünü kapsayın. Menüler ve yazılar dışarıda kalsın; model sadece bu alanı görür.
   - *Gizlilik maskesi ekle:* hasta adı ve ID şeridini seçin. Bu alan kayıtlarda ve akışta siyaha boyanır.
4. **Ayarlar → Kalibrasyon:** "Skaladan kalibre et" → derinlik skalasında iki çentiğe tıklayın → mesafeyi (mm) girin → ön ayar olarak kaydedin (ör. "4 cm derinlik"). **Cihazda derinlik veya zoom değişince ilgili ön ayarı seçin.**
5. **Ayarlar → Segmentasyon modeli:** ağırlık dosyasını `models/` altına koyun ve parametreleri eğitimdekiyle aynı yapın.

## Muayene akışı

1. Transvers kesitte nodülü bulun → **Space** (dondur) → nodüle tıklayın (AI). Kontur, genişlik (G) ve AP otomatik çizilir.
2. **AI konturunu kontrol edin.** AI'ın çizdiği kontur **kesikli mavi** görünür ve onaylanana kadar *öneri* sayılır. Görüntünün üstündeki çubuktan:
   - **Düzenle (E):** konturu tutup sürükleyin; daire içindeki noktalar birlikte hareket eder. Daireyi fare tekerleği veya `[` `]` ile büyütüp küçültebilirsiniz.
   - **Kısmi düzelt (K):** konturun üzerinden başlayıp yeni sınırı çizin, yine konturun üzerinde bitirin; yalnızca aradaki parça değişir.
   - **Yeniden çiz (D):** konturu baştan kendiniz çizin.
   - **AI tekrar (A):** nodülün başka bir noktasına tıklayarak modeli yeniden çalıştırın.
   - **Geri al / Yinele (Ctrl+Z / Ctrl+Y)**, **Sil (Del)**. Geri al ile her zaman AI'ın ilk önerisine dönebilirsiniz.
   - Uygunsa **Onayla (Enter)**. Onaylanan kontur **düz sarı** olur. Onaylanmamış AI konturu varken kayıt yapılamaz.
3. **2** → longitudinal sekmesi → Space → nodüle tıklayın → uzunluk (U) çizilir → kontrol edip onaylayın.
4. Gerekirse ölçümleri **W / H / L** kumpaslarıyla düzeltin ya da mm değerlerini yazın.
5. TI-RADS formunu doldurun. Şekil, AP/G oranından önerilir; değiştirebilirsiniz.
6. Onay kutusunu işaretleyip **Onayla ve kaydet**.

Kayıtta AI'ın ilk konturu (`*_mask_ai.png`) ve radyoloğun son konturu (`*_mask.png`) ayrı saklanır. Raporda konturun AI'dan değiştirilmeden mi, düzeltilerek mi yoksa elle yeniden çizilerek mi onaylandığı ve AI–radyolog uyumu (Dice) yazılır.

| Kısayol | İşlev |
|---|---|
| Space | Dondur / canlıya dön |
| 1 / 2 | Transvers / Longitudinal |
| A | AI segmentasyon (nodüle tıkla) |
| W / H / L | Genişlik / AP / Uzunluk kumpası |
| D | Elle kontur (basılı tutup çiz ya da tıkla; Enter/ilk nokta ile kapat) |
| E | Konturu sürükleyerek düzelt (tekerlek / [ ] fırça boyutu) |
| K | Konturun bir bölümünü yeniden çiz (kısmi düzeltme) |
| Enter | AI konturunu onayla |
| Ctrl+Z / Ctrl+Y | Geri al / Yinele |
| Del | Konturu sil |
| T / G | Biyopsi: hedefi / giriş noktasını seç |
| F | Biyopsi: iğne konumunu kaydet |
| S | Etiketi kaydet (etiketleme modunda) |
| Esc | Aracı bırak |

## Biyopsi rehberi (canlı iğne takibi)

Üst menüden **Biyopsi** moduna geçin. Canlı USG görüntüsünün üzerine bir rehber katmanı çizilir; kayıt dışında tüm işlemler tarayıcıda, canlı görüntü üzerinde yapılır.

1. **Hedefi seçin (T):** nodülde iğnenin ulaşmasını istediğiniz noktaya tıklayın. Hedef, görüntü parçasıyla takip edilir; solunum ve hafif prob hareketinde nodülle birlikte kayar. İğne hedefe yaklaşınca bu takip dondurulur, çünkü parlak iğne takibi yanıltır.
2. **Giriş noktasını seçin (G):** iğnenin görüntüye gireceği yere (prob kenarı) tıklayın. Bu adım isteğe bağlıdır (Esc ile atlanır), ama algılamayı belirgin şekilde güçlendirir. Planlanan yol kesikli mavi çizgiyle, hedefe kalan santimetre işaretleriyle ve hedef yarıçapı genişliğinde bir koridorla gösterilir.
3. **İğneyi ilerletin.** İğne otomatik algılanır; uç noktası ve iğne bu açıyla devam ederse izleyeceği yol (kesikli çizgi) gösterilir. Paneldeki durum:
   - **Rotada** (yeşil): tahmini yol hedef dairesinden geçiyor; hedefe kalan mesafe yazılır.
   - **Sapma var** (turuncu): tahmini yol hedefi kaç mm ıskalıyor ve hedef iğne hattının üstünde mi, altında mı.
   - **Uç hedefte** (parlak yeşil): uç, hedef dairesinin içinde.
   - **Uç hedefi geçti** (kırmızı).
   - Ayrıca: uç–hedef mesafesi, kalan yol, iğne açısı, plana göre açı farkı ve uç derinliği.
4. **İğne konumunu kaydet (F):** o anki görüntü, rehber çizimi ve ölçümler `data/biopsy/<tarih>/` altına kaydedilir (`frame.png`, `overlay.jpg`, `state.json`). Belgeleme için kullanılabilir.

Nasıl çalışır: Eğitilmiş bir model gerekmez. İğne, görüntüdeki ince, parlak, eğik çizgi olarak bulunur (arka plan çıkarma, Hough, çizgi uydurma). Aşağıdaki ön bilgiler kullanılır:
- Planlanan yol.
- Önceki karedeki konum.
- İlerleyen uçtaki hareket.
- Görüntüyü iki yandan kesen fasya ve kapsül benzeri çizgilerin elenmesi.

Uç, çizgi boyunca parlaklık profilinden belirlenir. Kare başına işlem yaklaşık 10–30 ms sürer.

Sınırlar:
- Yalnızca **in-plane** teknikte çalışır; out-of-plane iğne yalnızca bir nokta olarak görünür.
- Çok soluk iğnelerde, iğne görüntü düzleminden çıktığında ya da güçlü artefaktlarda algılama kaybolabilir veya gecikebilir.
- Uç, kapsül veya fasyaya değdiği anda birkaç mm hatalı görünebilir.
- Rehber yalnızca yardımcıdır; **iğneyi her zaman görüntüde gözle takip edin.**
- Kalibrasyon yoksa mesafeler piksel cinsinden gösterilir.
- Sentetik görüntülerde test edilmiştir; gerçek iğne görüntüleriyle doğrulanmadan klinik kararda kullanılmamalıdır.

İleride gerçek iğne görüntüleri toplandığında bir iğne segmentasyon modeli eğitilip bu algılayıcının yerine konulabilir.

## Veri toplama ve model eğitimi

Model henüz yoksa önce veri toplanır. Etiketleme de bu uygulamada yapılır. Böylece eğitim görüntüleri, modelin kullanımda göreceği capture görüntüleriyle aynı yoldan gelir (aynı capture card, aynı format, aynı ROI).

### 1. Etiketleme (üst menü → **Etiketleme**)

1. **Hasta takma kodu** (ör. `P0001`) ve **Etiketleyen** girin, Enter'a basın. Hasta adı ya da TC kimlik no girmeyin. Takma kod ile gerçek hasta eşleşmesini ayrı ve güvenli bir listede tutun.
2. Nodülü ekrana getirin → **Space** (dondur).
3. **D** ile konturu çizin: fareyi basılı tutup sınır boyunca dolaşın, bırakınca kontur kapanır. İsterseniz nokta nokta tıklayarak da çizebilirsiniz.
4. Gerekirse **E** ile düzeltin: konturu tutup sürükleyin, yakın noktalar yumuşakça birlikte hareket eder.
5. **S** ile kaydedin. Görüntü otomatik olarak canlıya döner, aynı nodülün bir sonraki karesine geçebilirsiniz.
6. **2** ile longitudinal kesite geçip aynı adımları tekrarlayın. Aynı hastadaki diğer nodül için **+** (N2, N3...) kullanın.
7. Nodül olmayan kareleri **"nodül yok (negatif örnek)"** ile kaydedin. Bu, modelin yanlış pozitiflerini azaltır.

TI-RADS formu isteğe bağlıdır; doldurulursa etiketle birlikte saklanır ve ileride özellik sınıflayıcıları için kullanılır.

İpuçları:
- Aynı nodülden farklı açılardan birkaç kare kaydedin; çeşitlilik modeli güçlendirir. Bölme hasta düzeyinde yapıldığı için bu durum sızıntıya yol açmaz.
- Cihaz ayarlarını (preset, SRI/compound, frekans) standart tutun ya da bilinçli olarak çeşitlendirin, ama kaydını tutun.
- Bir alt kümeyi ikinci bir radyolog bağımsız etiketlesin. Gözlemciler arası uyum, modelin ulaşabileceği tavanı gösterir.
- Model devreye girdikten sonra da etiketlemeye devam edin. AI konturu düzelttiğinizde hem düzeltilmemiş AI maskesi (`mask_ai.png`) hem son maske saklanır, panelde AI–radyolog Dice ortalaması görünür.

Kayıtlar `data/dataset/<hasta>/<nodül>/<etiket>/` altında tutulur: `image.png`, `mask.png`, `mask_ai.png` (varsa) ve `meta.json` (ROI, mm/px, cihaz bilgisi, TI-RADS, etiketleyen). Yanlış bir kayıt "Son kayıtlar" listesinden silinebilir; dosya silinmez, işaretlenip eğitimden çıkarılır.

### 2. Dışa aktarma

Paneldeki **Eğitim için dışa aktar** düğmesi ya da:
```bash
python -m tools.export_dataset             # data/dataset -> data/export
python -m tools.export_dataset --nnunet 501   # + nnU-Net v2 klasörü
```
- Görüntüler ROI'ye kırpılır (uygulama çıkarımda da yalnızca ROI'yi kullanır).
- Bölme **hasta düzeyindedir**: %70 eğitim, %15 doğrulama, %15 test.
- Sonraki aktarımlarda önceki bölme korunur; yeni hastalar eğitime eklenir, **test seti değişmez**.

### 3. Eğitim (GPU'lu makinede)

```bash
pip install -r requirements-ml.txt      # CUDA için önce torch'u pytorch.org'dan kurun
python -m tools.train_unetpp --data data/export --out models
```
- UNet++ (ResNet34 encoder, ImageNet ön eğitimli), Dice + BCE kaybı. Artırmalar: çevirme, ±10° dönme, ölçek, parlaklık/kontrast/gama, benek gürültüsü. Dikey çevirme yapılmaz, çünkü derinlik yönü anlamlıdır.
- Her epoch'ta doğrulama setinde, uygulamadaki gibi "nodüle tıklama" ile **Dice** ve **mm cinsinden boyut hatası** ölçülür. En iyi model kaydedilir; iyileşme durunca eğitim kendiliğinden biter.
- Çıktılar: `models/unetpp.pth`, `models/unetpp.onnx` ve `models/unetpp.json`. JSON dosyası ön işleme ayarlarını içerir ve uygulama tarafından otomatik okunur; elle ayar girmeye gerek kalmaz.
- Uygulamada **Ayarlar → Segmentasyon modeli → yol: `models/unetpp.onnx` → Modeli yükle**.
- Test setine yalnızca en sonda, bir kez bakın:
  ```bash
  python -m tools.train_unetpp --data data/export --out models --eval-test
  ```
- İnternetsiz makinede `--encoder-weights none` kullanın ya da ağırlıkları önceden indirin.

Kabaca hedefler: ilk deneme için 100–150 hastada 300+ görüntü; boyut hatası 1–2 mm'nin altına indiğinde biyopsi eşikleri açısından anlamlı hale gelir.

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
  capture.py       Okuma thread'i, tak-çıkar, sinyal yok tespiti, demo kaynak
  devices.py       Cihaz adları (Linux sysfs / Windows DirectShow), capture card
                   seçimi, otomatik format/çözünürlük
  shortcuts.py     Masaüstü kısayolu ve otomatik başlatma
  needle.py        İğne algılama, hedef takibi, rehber geometrisi
  biopsy.py        Canlı biyopsi takibi (arka plan), anlık görüntü kaydı
  segmentation.py  UNet++ (.pth / .onnx) çıkarımı, ROI kırpma
  measure.py       Maske → kontur, G/AP/en uzun çap, kalibrasyon
  tirads.py        ACR TI-RADS puanlama ve boyut eşikleri (tek doğruluk kaynağı)
  storage.py       Vaka kaydı ve rapor
  config.py        data/config.json
  dataset.py       Etiket kaydı, istatistik, hasta düzeyinde dışa aktarma
frontend/          index.html, app.js, style.css (internet gerektirmez)
tools/
  export_dataset.py  Etiketleri eğitim formatına aktarır (+ nnU-Net v2)
  train_unetpp.py    UNet++ eğitimi, değerlendirme, ONNX + json çıktısı
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
