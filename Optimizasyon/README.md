# TEKNOFEST 2026 Hepsiburada — Mid-Mile Linehaul Optimizasyonu

Türkiye'deki 18 kargo transfer merkezi arasındaki haftalık yükü, **toplam maliyeti
(araç + SLA cezası) en aza indirecek** şekilde araçlara, rotalara ve saatlere
dağıtan optimizasyon çözümü. Gelişmiş Çözüm Aşaması teslimi.

---

## Teslim edilen sonuç

| | Değer |
|---|---|
| **Toplam maliyet** | **11.001.468,07 TL** |
| Araç maliyeti | 10.936.761,67 TL |
| SLA cezası | 64.706,40 TL (toplamın %0,59'u) |
| Planlanan talep | **4.046 / 4.046** (tamamı) |
| Taşınan desi | 5.418.736 |
| Fiziksel araç | 523 |
| Plan satırı | 13.982 |
| Tarih aralığı | 29 Haziran – 7 Temmuz 2026 |

Teslim dosyaları [`TESLIM/`](TESLIM) klasöründedir:

| Dosya | İçerik |
|---|---|
| `TESLIM/Talep-tahmini.xlsx` | Talep tahmini (6 sütun, resmî format) |
| `TESLIM/Tasima-plani.xlsx` | Taşıma planı (16 sütun, resmî format) |

**İki dosya birbirine bağlıdır.** Plandaki her `Talep ID`, tahmin dosyasındaki bir
talebin kendisi ya da `-1`, `-2` ekli parçasıdır; desi toplamları birebir tutar.
Doğrulandı: uydurma ID **0**, planda yer almayan talep **0**, desi sapması
**0,000000**.

---

## Tek tıkla çalıştırma

**[`CALISTIR.bat`](CALISTIR.bat) dosyasına çift tıklayın.**

Betik sırayla şunları yapar:

1. Python ve bağımlılıkları kontrol eder (`pip install -r requirements.txt`)
2. **Dört tohumu paralel** başlatır (42, 7, 123, 999) — her biri ayrı süreç
3. Hepsi bittiğinde en ucuz **geçerli** planı `outputs/TAŞIMA PLANI.xlsx`'e yazar

**Süre:** tohum başına ~70 dakika; dördü paralel koştuğu için duvar saati de
**~70 dakika**. Intel i7-10750H (6 çekirdek) üzerinde ölçüldü.

> **Neden dört tohum?** Arama sonucu rastgele tohuma göre yüz binlerce TL oynuyor
> — ölçülen yayılım bu koşuda 199.626 TL. Tek tohum çalıştırmak, hangi tohumu
> çektiğinize bağlı bir kumardır. Dördünü paralel koşup en ucuzunu seçmek hiçbir
> algoritma değişikliği olmadan doğrudan kazanç sağlar.

### Elle çalıştırma

Tek tohum:

```bash
python -m src.planning.run_alns --iterations 1500 --time-limit 9000 --seed 123 --forecast data/interim/talep_tahmini_ml.xlsx
```

Üretilmiş tohum çıktılarından en ucuzunu seçmek:

```bash
python -m src.planning.select_best_plan
```

Testler (237 test, ~8 dakika):

```bash
python -m unittest discover -s tests
```

---

## Gereksinimler

- **Python 3.11+**
- `pip install -r requirements.txt` → `pandas`, `openpyxl`, `ortools`

`ortools` yalnızca opsiyonel bir son iyileştirme adımı için gerekir; kurulu
olmasa da plan üretimi çalışır.

---

## Klasör yapısı

```
Optimizasyon/
├── CALISTIR.bat            Tek tıkla plan üretimi
├── README.md               Bu dosya
├── MIMARI.md               Ne nedir, ne işe yarar, veri nereden nereye akar
├── requirements.txt
│
├── TESLIM/                 ► TESLİM EDİLEN İKİ DOSYA
│   ├── Talep-tahmini.xlsx
│   └── Tasima-plani.xlsx
│
├── src/                    Kaynak kod (ayrıntı: MIMARI.md)
│   ├── domain/             Veri modelleri
│   ├── io/                 Excel okuma/yazma, şema doğrulama
│   ├── evaluation/         Maliyet, bağımsız audit, validator, kural politikası
│   ├── planning/           Greedy plan kurucular + CLI'lar
│   ├── alns/               ALNS arama motoru
│   └── forecasting/        Talep tahmini yardımcıları
│
├── config/rules.yaml       Dosya/sütun eşlemesi + kural yorumları
├── raw/current/            Yarışmanın verdiği ham Excel girdileri
├── data/interim/           Talep tahminleri (aşağıya bakın)
├── outputs/                Üretilen plan
└── tests/                  237 birim testi
```

### `data/interim/` içindeki üç dosya

| Dosya | Ne işe yarar |
|---|---|
| **`talep_tahmini_ml.xlsx`** | **Teslim edilen tahmin.** Makine öğrenmesi modellerinin çıktısı. Plan bu dosyadan üretildi. |
| `development_forecast.xlsx` | Geçmiş verinin aritmetik ortalamasından üretilen **deterministik yedek** tahmin. Test paketi bunu gerçekçi bir veri seti olarak kullanır; `python -m src.forecasting.generate_development_forecast` ile yeniden üretilebilir. |
| `rota_id_eslesme.xlsx` | Model çıktılarındaki sayısal `rota_id` → çıkış/varış merkezi eşlemesi. `convert_ml_forecast` bunu kullanır. |

Plan üretimi **her zaman** `--forecast` ile hangi dosyanın kullanılacağını açıkça
alır; `CALISTIR.bat` teslim edilen `talep_tahmini_ml.xlsx`'i geçirir.

---

## Çözümün özeti

Problem üç katmanda çözülür:

1. **Greedy kurucu** — talepleri atomik parçalara böler, zorunlu kiralık araçları
   doldurur, aynı çıkışlı yükleri **20 durağa kadar milk-run** rotalarında
   birleştirir. Her adım yalnız gerçek TL'yi düşürüyorsa kabul edilir.
2. **ALNS** (Adaptive Large Neighborhood Search) — bu planı yıkıp yeniden kurarak
   iyileştirir. Bu koşuda **13.727.910 → 11.001.468 TL**, yani **%19,9** kazanç.
3. **Doğrulama** — her aday plan, maliyeti bağımsız olarak yeniden hesaplayan bir
   auditten ve tüm yarışma kısıtlarını denetleyen bir validator'dan geçer. Geçmeyen
   plan asla kabul edilmez, Excel'e yazılamaz.

Ayrıntılı mimari, katman katman akış ve her dosyanın ne yaptığı:
**[MIMARI.md](MIMARI.md)**

---

## Doğruluğa dair üç güvence

Bu çözümde optimizasyonun kendi sonucunu onaylamasına izin verilmez:

1. **Bağımsız maliyet auditi.** `evaluation/cost_audit.py`, `evaluation/costs.py`'yi
   *import etmez*; bütün formülleri sıfırdan yazar. İki hesap ayrışırsa validator
   hata üretir. Kod tekrarı kasıtlıdır.
2. **Tam kısıt denetimi.** `evaluation/validator.py` elleçleme kapasitesini, tır
   ziyaret kotasını, zorunlu kiralık takvimini, SLA'yı, yük akışının sürekliliğini
   ve zaman tutarlılığını ayrı ayrı denetler; hata fırlatmaz, **rapor** döner —
   böylece bir planın 15 sorunu varsa 15'i birden görülür.
3. **Excel round-trip.** Yazılan dosya geri okunup satır, kimlik, zaman, desi, süre
   ve TL açısından yeniden karşılaştırılır. Serileştirmede sessiz veri kaybı
   olamaz.

Ayrıca **determinizm**: aynı tohum ve aynı girdi her zaman aynı planı üretir.
Süreçler arası kararlılık için sıralamalar açıkça tanımlıdır (Python `set`
sıralaması ve `hash()` süreçler arası kararlı değildir).

---

## Yeni bir talep tahminiyle çalıştırma

Tahmin dosyası değişirse plan **baştan üretilmelidir** — yarışma talep ID'lerinin
iki dosyada birebir eşleşmesini şart koşar, mevcut planı yamamak geçersizdir.

```bash
python -m src.planning.run_alns --iterations 1500 --time-limit 9000 --seed 123 --forecast <yeni_tahmin.xlsx>
```

Yeni tahmin dosyası şu şemayı **birebir** taşımalıdır (sayfa adı `Sheet1`):

| Sütun | Başlık |
|---|---|
| 1 | `Talep ID` |
| 2 | `Tarih` |
| 3 | `Talep Tamamlama Saati` |
| 4 | `Çıkış Transfer Merkezi` |
| 5 | `Varış Transfer Merkezi` |
| 6 | `Tahmin Edilen Desi` |

Şema `config/rules.yaml` → `forecast_demand_template` içinde tanımlıdır; kod
hiçbir yerde sütun adı sabit yazmaz.
