# Mimari — ne nedir, ne işe yarar

Bu belge kodun nasıl kurgulandığını, verinin nereden nereye aktığını ve her
dosyanın ne yaptığını anlatır. Çalıştırma talimatı için [README.md](README.md).

---

## 1. Problem tek paragrafta

18 transfer merkezi arasında bir haftalık kargo talebi var (4.046 talep,
5.418.736 desi). Bu yükü araçlara, rotalara ve saatlere dağıtmamız gerekiyor.
Amaç **toplam maliyeti** en aza indirmek:

```
Toplam Maliyet = Araç Maliyeti + SLA Cezası

Araç Maliyeti = (Saatlik Kira × Kullanım Süresi) + (Mesafe × Km Başı Maliyet)
SLA Cezası    = Geciken Desi × Gecikme Saati (yukarı yuvarlanmış) × 0,40 TL
```

**Kullanım süresi** yolculuk + elleçleme + bekleme sürelerinin tamamıdır.
Elleçleme desi başına 0,01 dakikadır ve araç yüklenirken de boşaltılırken de
işler — yani büyük yük hem taşımayı hem aracın kirasını pahalılaştırır.

Zorlayıcı kısıtlar:

| Kısıt | Anlamı |
|---|---|
| **Elleçleme kapasitesi** | Her merkezin günlük desi limiti; giren + çıkan + aktarma toplamı. Aşılamaz. |
| **Tır ziyaret kotası** | Merkez başına günlük tır sayısı. Yalnız Tır için; diğer araç tiplerinde limit yok. |
| **Zorunlu kiralık filo** | Her gün 14 kiralık araç, sabit rotalarda, **talep olmasa bile** çıkmak zorunda. Sapamaz, dönüş yapamaz. |
| **SLA** | Hat başına 1 veya 2 gün. Talep tamamlanma anında başlar, **nihai varışta elleçleme bitince** durur. |
| **Konsolidasyon** | Serbest. Bir araca farklı varışlara giden yükler konabilir; ara merkezde aktarma yapılabilir (elleçleme iki kez sayılır). |

---

## 2. Katmanlar

Bağımlılık **hep aşağıdan yukarı** akar. Alt katman üst katmanı tanımaz.

```
┌──────────────────────────────────────────────────────────────┐
│  CLI            run_alns · select_best_plan · export_* ...   │
├──────────────────────────────────────────────────────────────┤
│  ALNS           engine · destroy · repair · evaluator        │
│  (arama)        materializer · state · invariants            │
├──────────────────────────────────────────────────────────────┤
│  Planning       baseline · consolidated · mandatory_filled   │
│  (greedy kurucu) two_stop · multi_stop · partitioning        │
│                 handling_ledger · truck_ledger               │
├──────────────────────────────────────────────────────────────┤
│  Evaluation     costs · cost_audit · validator · policy      │
│  (doğruluk)                                                  │
├──────────────────────────────────────────────────────────────┤
│  Domain         models · solution_models                     │
│  (veri modelleri)                                            │
├──────────────────────────────────────────────────────────────┤
│  IO             manifest · excel_loader · validation         │
│                 transport_plan_workbook                      │
├──────────────────────────────────────────────────────────────┤
│  Yapılandırma   config/rules.yaml  ·  raw/current/*.xlsx     │
└──────────────────────────────────────────────────────────────┘
```

**Domain katmanı hiçbir şeye bağımlı değildir** — yalnız Python standart
kütüphanesini kullanır. Excel okumaz, rota üretmez, maliyet hesaplamaz. Bu sayede
test edilmesi ve değiştirilmesi kolaydır.

---

## 3. Uçtan uca veri akışı

```
config/rules.yaml ─┐
raw/current/*.xlsx ┤
                   ▼
            io/manifest.py            mantıksal ad → dosya, sayfa, sütun
                   ▼
            io/excel_loader.py ◄──── io/validation.py  (şema kontrolü)
                   ▼
        ┌──────────────────────────┐
        │ StaticProblemData        │  ← merkezler, mesafeler, araçlar,
        │ DemandInstance[]         │    kapasiteler, kotalar, kiralık filo
        │ ProblemData              │  ← talep (tek değişen girdi)
        └──────────────────────────┘
                   ▼
        evaluation/policy.py          config yorumlarını tip güvenli hale getirir
                   ▼
   ┌───────────────────────────────────────────────────────┐
   │  GREEDY KURUCU ZİNCİRİ                                │
   │                                                        │
   │  partitioning     talep → atomik parçalar (≤12.000)   │
   │  handling_ledger  merkez/gün elleçleme defteri         │
   │  truck_ledger     merkez/gün tır ziyaret defteri       │
   │                                                        │
   │  baseline → consolidated → mandatory_filled            │
   │  (fizibil)   (aynı O-D)     (kiralıkları doldur)      │
   │                    ↓                                   │
   │              two_stop → multi_stop                     │
   │              (iki durak) (uzun milk-run)               │
   └───────────────────────────────────────────────────────┘
                   ▼
            başlangıç TransportPlan
                   ▼
   ┌───────────────────────────────────────────────────────┐
   │  ALNS ARAMA                                            │
   │                                                        │
   │  materializer.extract_state()  plan → hafif durum      │
   │                   ▼                                    │
   │  engine.run()  ── döngü ──┐                            │
   │     yık → onar → proxy maliyet → SA kabul              │
   │                            │                           │
   │     kesin denetim: materialize → validate → hesapla    │
   │                            ▼                           │
   │                     en iyi plan                        │
   └───────────────────────────────────────────────────────┘
                   ▼
        io/transport_plan_workbook.py
                   ▼
        outputs/TAŞIMA PLANI.xlsx
                   ▼
        geri okunup round-trip doğrulaması yapılır
```

---

## 4. Dosya dosya ne yapar

### `src/domain/` — veri modelleri (3 dosya)

| Dosya | İşi |
|---|---|
| `models.py` | **Girdi** modelleri: `Center`, `TravelArc`, `VehicleType`, `MandatoryRentalRoute`, `HandlingCapacity`, `DailyTruckCapacity`, `DemandInstance`. Hepsi `frozen=True` — bir kez kurulduktan sonra sessizce değişemez. |
| `solution_models.py` | **Çözüm** modelleri: `DemandPart` (atomik izlenebilir yük), `DemandAssignment`, `VehicleLeg`, `HandlingOperation`, `VehicleTrip`, `TransportPlan`, `CostBreakdown`, `ValidationReport`. |

Girdi ve çözüm modellerinin ayrı olması bilinçli: "ne çözülecek" ile "nasıl
çözüldü" karışmaz.

### `src/io/` — Excel sınırı (6 dosya)

| Dosya | İşi |
|---|---|
| `manifest.py` | `config/rules.yaml`'ı okur; mantıksal ad → fiziksel dosya/sayfa/sütun eşlemesi. **Kod hiçbir yerde dosya veya sütun adı sabit yazmaz.** |
| `excel_loader.py` | Ham Excel'leri okuyup domain nesnelerine çevirir. `Talep ID` dahil her alan dosyadan gelir; kod ID üretmez. |
| `validation.py` | Şema kontrolü: beklenen sütunlar var mı, tipler doğru mu. |
| `transport_plan_workbook.py` | Planı resmî 16 sütunlu formata yazar ve **geri okuyup karşılaştırır**. Geçersiz plan yazılamaz. |
| `inspect_data.py` | Veriyi yükleyip özet basar (teşhis aracı). |

### `src/evaluation/` — doğruluk katmanı (5 dosya)

| Dosya | İşi |
|---|---|
| `policy.py` | `rules.yaml`'daki kural yorumlarını tip güvenli `EvaluationPolicy`'ye çevirir ve **kesin kuralla çelişen config'i reddeder** (45 alan kilitli). |
| `costs.py` | Resmî maliyet hesabı: kullanım süresi (elleçleme + bekleme + seyir), saat tavana yuvarlama, mesafe maliyeti, SLA cezası. |
| `cost_audit.py` | **Bağımsız** ikinci maliyet hesabı. `costs.py`'yi *import etmez*, formülleri sıfırdan yazar. |
| `validator.py` | Tüm yarışma kısıtlarını denetler: elleçleme kapasitesi (gece yarısı oransal bölme dahil), tır ziyaret kotası, zorunlu kiralık takvimi, SLA, yük akışı sürekliliği, zaman tutarlılığı, kapasite. Hata **fırlatmaz**, yapılandırılmış rapor döner. |

### `src/planning/` — greedy kurucular ve CLI'lar (17 dosya)

| Dosya | İşi |
|---|---|
| `partitioning.py` | Talebi atomik parçalara böler (`D00001` → `D00001-1`, `D00001-2`). Toplamı **birebir korur**, ondalık yuvarlama yapmaz. |
| `handling_ledger.py` | Merkez/gün elleçleme kapasitesi defteri. `fork()`/`commit_from()` ile **işlemsel**: reddedilen deneme kapasite tüketmez. |
| `truck_ledger.py` | Merkez/gün tır ziyaret kotası defteri; zorunlu kiralık ziyaretleri sabit taban olarak önceden yüklenir. |
| `baseline.py` | Fizibil taban: her parçaya ayrı araç, direkt taşıma. Optimizasyon yapmaz, zincirin çalıştığını kanıtlar. |
| `consolidated.py` | Aynı tarih + aynı O-D parçalarını FFD ile kutulara paketler, kutuya tek araç verir. |
| `mandatory_filled.py` | Uygun yükü **önce zorunlu kiralıklara** koyar — o araçlar zaten çıkacağı için o kapasite bedavadır. |
| `two_stop.py` | Aynı çıkışlı, iki farklı varışlı kutuları tek araçta birleştirir. İki durak sırasını da dener, **yalnız gerçek TL'yi düşüreni** kabul eder. |
| `multi_stop.py` | Rotaları **20 durağa kadar** büyütür (cheapest insertion). Her ek durak gerçek maliyeti düşürmek zorundadır. |
| `run_alns.py` | **Ana komut.** Greedy tohumu kurar → ALNS çalıştırır → Excel yazar. |
| `select_best_plan.py` | Tohum başına üretilmiş planlardan en ucuz **geçerli** olanı resmî çıktıya kopyalar. |
| diğer `export_*`, `compare_*` | Ara katmanları tek tek üretip karşılaştıran teşhis CLI'ları. |

### `src/alns/` — arama motoru (11 dosya)

| Dosya | İşi |
|---|---|
| `state.py` | Hafif çözüm temsili: `ALNSSolutionState` (parçalar + donmuş kiralık seferler + oynanabilir spot rotalar). Tam `TransportPlan` binlerce iterasyon için çok ağır. |
| `invariants.py` | Bir durumun yapısal geçerliliğini denetler: kapasite, yük sürekliliği, her parça tam bir kez alınıp bırakılmış mı. |
| `evaluator.py` | **Proxy maliyet** — materialize etmeden hızlı tahmin, önbellekli. Ayrıca tır kotası ihlalini durum düzeyinde denetler ve çok günlü rotalarda durakların gerçekten hangi güne düştüğünü hesaplar. |
| `destroy.py` | Yıkım operatörleri: `random_removal`, `worst_removal`, `shaw_removal` (ilişkili yükleri birlikte söker). |
| `repair.py` | Onarım operatörleri: `greedy_insertion`, `regret_2_insertion`. Parçayı **2 gün sonrasına kadarki** rotalara da yerleştirebilir — ince hatlarda "beklet, dolu araçla gönder" stratejisi. |
| `route_index.py` | (gün, merkez) → rota indeksi; onarımda tüm rotaları taramayı önler. |
| `materializer.py` | Hafif durumu tekrar tam `TransportPlan`'a çevirir; saatleri elleçleme defterine uyacak şekilde yeniden zamanlar ve plan uzarsa zorunlu takvimi genişletir. |
| `engine.py` | Ana döngü: yık → onar → proxy → simulated annealing kabul → **kesin denetim**. Operatör ağırlıkları performansa göre uyarlanır. Ret sebeplerini hata koduna göre sayıp koşu sonunda raporlar. |
| `candidate_archive.py` | En iyi adayları saklar; periyodik kesin denetime sokulur. |
| `set_partition.py` | Opsiyonel son adım: toplanan rotaları CP-SAT ile yeniden birleştirmeyi dener. **Varsayılan kapalı** (ölçüldü, kazanç üretmedi). |

### `src/forecasting/` — talep tahmini yardımcıları (5 dosya)

| Dosya | İşi |
|---|---|
| `providers.py` | Geçmiş verinin aritmetik ortalamasından deterministik geliştirme tahmini üretir (ML tahmini yoksa yedek). |
| `workbook.py` | Tahmini resmî 6 sütunlu formata yazar; başlıkları manifestten alır. |
| `convert_ml_forecast.py` | ML modellerinin vardiya bazlı çıktılarını resmî formata çevirir ve yarışma kurallarını kapıda denetler. |
| `generate_development_forecast.py` | Geliştirme tahmini CLI'ı. |

---

## 5. Neden iki ayrı maliyet hesabı var?

`costs.py` ve `cost_audit.py` **aynı formülleri iki kez** uygular. Bu kod tekrarı
**kasıtlıdır ve temizlenmemelidir.**

```
CostCalculator ──► planın raporlanan maliyeti
                          ▼
                 TransportPlanValidator
                          ▼
                 cost_audit.py ──► bağımsız beklenen maliyet
                          ▼
                  iki sonuç karşılaştırılır
```

Sebep basit: **bir doğrulayıcı, doğruladığı şeyin kendisini referans alırsa hiçbir
şey doğrulamış olmaz.** `cost_audit.py`, `costs.py`'yi import etmez. İkisi
ayrışırsa validator hata üretir.

Aynı felsefe `transport_plan_workbook.py` içinde de var: yazılan Excel geri
okunup satır, kimlik, zaman, desi, süre ve TL açısından yeniden karşılaştırılır
(**round-trip doğrulaması**).

---

## 6. Yapılandırma sınırı: `config/rules.yaml`

İki iş yapar:

1. **Manifest** — mantıksal dosya adı → fiziksel dosya, sayfa ve sütun eşlemesi.
2. **Kural yorumları** — yarışmanın net cevaplamadığı noktalarda seçilen varsayım
   (saat yuvarlama yönü, gece yarısı kapasite dağıtımı, toleranslar…).

Bu yüzden **kodda magic number yoktur.** Bir kural yorumu değişecekse `rules.yaml`
güncellenir, kod değişmez. `evaluation/policy.py` bu değerleri okurken kesin
kurallarla çelişip çelişmediğini de denetler.

---

## 7. Tasarım kararları ve gerekçeleri

| Karar | Gerekçe |
|---|---|
| Domain modelleri `frozen=True` | Plan kurulduktan sonra sessizce değişemez; doğrulama ile maliyetleme aynı nesneyi görür |
| Girdi ve çözüm modelleri ayrı dosyalarda | "Ne çözülecek" ile "nasıl çözüldü" karışmaz |
| `StaticProblemData` / `DemandInstance` ayrımı | Sürpriz veri setinde **yalnız talep** değişir; statik veri sabittir |
| Validator exception değil **rapor** döner | Bir planın 15 sorunu varsa 15'i birden görülür, ilkinde durulmaz |
| İki bağımsız maliyet hesabı | Doğrulayıcı kendi kendini onaylamasın |
| Excel round-trip kontrolü | Serileştirmede sessiz veri kaybı olmasın |
| ALNS için ayrı hafif `state` | `TransportPlan` binlerce iterasyon için çok ağır |
| ALNS başlangıç planını daima saklar | Arama başarısız olsa bile geçerli bir plan teslim edilir |
| Ledger'da `fork` / `commit_from` | Reddedilen deneme hayalî kapasite tüketmesin |
| Her yerde açık sıralama | Aynı tohum → aynı sonuç, **süreçler arası dahil**. Python `set` sıralaması ve `hash()` süreçler arası kararlı değildir |
| Ret sebeplerinin sayılması | Sessizce atılan aday, teşhis edilemeyen hata demektir |

---

## 8. Performans ve ölçekleme

| Ölçüm | Değer |
|---|---|
| Talep / atomik parça | 4.046 / 4.140 |
| Greedy tohum kurulumu | ~60 saniye |
| ALNS 1500 iterasyon | ~70 dakika (tohum başına) |
| Dört tohum paralel | ~70 dakika duvar saati (6 çekirdek) |
| Bellek | süreç başına ~190 MB |
| Test paketi | 237 test, ~8 dakika |

Makine: Intel i7-10750H (6 fiziksel / 12 mantıksal çekirdek), 15,8 GB RAM.

ALNS'in bu koşudaki katkısı **13.727.910 → 11.001.468 TL (%19,9)**. Greedy tohum
tek başına iyi bir plan değildir; asıl kazanç aramadan gelir.
