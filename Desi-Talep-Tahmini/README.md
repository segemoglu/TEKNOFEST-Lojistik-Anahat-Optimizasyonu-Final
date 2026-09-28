# Lojistik Rota Hacim (Desi) Tahminleme Mimarisi

Bu proje, lojistik operasyonlarında Türkiye geneli rota bazlı günlük yük hacimlerini (desi) tahmin etmek için geliştirilmiş uçtan uca bir makine öğrenmesi boru hattıdır (pipeline). Sistem, operasyonel gerçeklikleri göz önünde bulundurarak **Sabah (09:00)** ve **Akşam (17:00)** vardiyaları için birbirinden izole iki ayrı model mimarisi üzerinde çalışır.

Zaman serisi dinamikleri, rotaların geçmiş hacim ortalamaları ve operasyonel standart sapmalar, XGBoost algoritması ve Target Encoder (Kategorik Dönüştürücü) kullanılarak modellenmiştir.

---

## 📂 Klasör Yapısı

Proje dosyaları, vardiyaların birbirine karışmasını engellemek ve düzeni sağlamak amacıyla aşağıdaki hiyerarşiye göre organize edilmiştir:

📦 **Desi-Talep-Tahmini**
* 📜 `Canliya_Alma_v2.ipynb` (Eğitim ve recursive tahmin notebook'u — Google Colab)
* 📜 `Master_Dataset_Tum_Turkiye.xlsx` (Model eğitimi için kullanılan ana tarihsel veri seti)
* 📜 `Master_Dataset_Tum_Özellikler_Hazir_V2.xlsx` (Gelecek tahminleri için dinamik özelliklerin bulunduğu veri seti)
* 📜 `Talep-tahmini.xlsx` (Sabah ve akşam tahminlerinin birleştirildiği teslim dosyası)
* 📂 **17.00_Aksam_Modeli**
  * 📜 `01_target_encoder.pkl` (Akşam rotaları için eğitilmiş Target Encoder nesnesi)
  * 📜 `xgboost_lojistik_modeli_AKSAM.pkl` (Hiperparametre optimizasyonu yapılmış Akşam XGBoost modeli)
  * 📜 `Test_Sonuclari_Gercek_Vs_Tahmin_AKSAM.xlsx` (Test seti gerçek / tahmin karşılaştırması)
  * 📜 `29Haziran_5Temmuz_AKSAM_Encoderli_Recursive_Rapor.xlsx` (Yarışma haftası akşam tahmini)
* 📂 **09.00_Sabah_Modeli**
  * 📜 `01_target_encoder_SABAH.pkl` (Sabah rotaları için eğitilmiş Target Encoder nesnesi)
  * 📜 `xgboost_lojistik_modeli_SABAH.pkl` (Hiperparametre optimizasyonu yapılmış Sabah XGBoost modeli)
  * 📜 `Test_Sonuclari_Gercek_Vs_Tahmin_SABAH.xlsx` (Test seti gerçek / tahmin karşılaştırması)
  * 📜 `29Haziran_5Temmuz_SABAH_Encoderli_Recursive_Rapor.xlsx` (Yarışma haftası sabah tahmini)

> **Not:** Notebook Google Colab için yazılmıştır; veri setlerini ve model
> dosyalarını `/content/` altında bekler.

---

## ⚙️ Teknik İşleyiş ve Modüller
## 1. Veri Hazırlığı ve Ayrıştırma
Zaman Serisi Bütünlüğü: Kök veri seti (Master_Dataset_Tum_Turkiye.xlsx), zaman serisi bütünlüğünü korumak adına öncelikle tarihe ve sabah_mi durumuna göre kronolojik olarak sıralanır.

Vardiya İzolasyonu: Veri, Akşam (sabah_mi == 0) ve Sabah (sabah_mi == 1) olmak üzere iki ana kola ayrılır.

Sızıntı (Leakage) Kontrolü: Modelin ezber yapmasını önlemek için tarih, talep kimlikleri, sentetik bayraklar ve geleceği doğrudan ele veren sızıntı yaratacak çöp sütunlar (cop_sutunlar) veriden temizlenir.

---

## 2. Target Encoding (Akıllı Kategorik Dönüşüm)
Karakteristik Dönüşüm: Yüzlerce farklı rota_id'nin barındırdığı tarihsel hacim karakteristiğini modelin anlayabileceği matematiksel bir forma çevirmek için category_encoders kütüphanesinden TargetEncoder kullanılır.

Eğitim ve Test Ayrımı: Veri sızıntısını engellemek için encoder sadece eğitim verisine (fit_transform) uygulanır ve ardından test/tahmin verisine (transform) aktarılır.

Kayıt:Eğitilen encoder nesneleri her vardiya için ayrı ayrı .pkl formatında kaydedilir.

---

## 3. Model Eğitimi ve Hiperparametre Optimizasyonu
Algoritma: Tahmin motoru olarak XGBRegressor (Xtreme Gradient Boosting) kullanılmıştır.

Optimizasyon: Modeller TimeSeriesSplit (Zaman Serisi Çapraz Doğrulama) ve RandomizedSearchCV kullanılarak geniş bir hiperparametre ağında (30 iterasyon) optimize edilir.

Performans Ölçümü: Başarı ölçümü için MAE (Ortalama Mutlak Hata), RMSE, R2 Skoru ve WAPE (Ağırlıklandırılmış Ortalama Yüzde Hata) metrikleri hesaplanır ve şampiyon modeller diskte saklanır.

---

## 4. Tahminleme Mimarisi 
Gelecek 7 günlük operasyonu tahmin etmek için yazılan ardışık (recursive) boru hattı şu adımları izler:

Dinamik Güncelleme: Tahmin edilecek her yeni gün için, son_7_gun_hareketli_ortalama, operasyonel_std_7 ve operasyonel_std_30 sütunları geçmişteki gerçek (veya önceki adımda tahmin edilmiş) desiler baz alınarak anlık hesaplanır.

Çıktı Raporlama: Target encoder ile işlenen günün verisi tahmin edilir, ana veri setine enjekte edilir ve döngü tamamlandığında çıkış ve varış bilgilerini (cikis, varis) içeren nihai bir Excel operasyon raporu oluşturulur.

## 🚀 Kurulum ve Gereksinimler

Projenin bağımlılıklarını sisteminize kurmak için aşağıdaki paketlerin yüklü olması gerekmektedir:

```bash
pip install pandas numpy xgboost scikit-learn category_encoders openpyxl matplotlib joblib
```
