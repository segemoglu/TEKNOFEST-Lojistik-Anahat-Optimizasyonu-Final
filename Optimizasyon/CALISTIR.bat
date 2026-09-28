@echo off
REM =====================================================================
REM  TEKNOFEST 2026 Hepsiburada - Mid-Mile Linehaul Optimizasyonu
REM  Tasima planini sifirdan yeniden uretir.
REM
REM  Tek tik : bu dosyaya cift tiklayin.
REM  Sure    : dort tohum paralel, ~70 dakika duvar saati.
REM  Cikti   : outputs\TASIMA PLANI.xlsx  (en ucuz gecerli plan)
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set TOHUMLAR=42 7 123 999

echo.
echo ==========================================================
echo  1/3  Python ve bagimliliklar
echo ==========================================================
python --version
if errorlevel 1 (
  echo.
  echo HATA: Python bulunamadi. Python 3.11+ kurup PATH'e ekleyin.
  pause
  exit /b 1
)
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo HATA: Bagimliliklar kurulamadi.
  pause
  exit /b 1
)

echo.
echo ==========================================================
echo  2/3  Dort tohum paralel calisiyor  (~70 dakika)
echo ==========================================================
echo  Arama sonucu rastgele tohuma gore yuz binlerce TL oynar,
echo  bu yuzden dort tohum ayni anda kosar ve en ucuzu secilir.
echo.

if not exist outputs mkdir outputs
del /q outputs\kosu_seed*.xlsx 2>nul
del /q outputs\kosu_seed*.log 2>nul
del /q outputs\bitti_*.flag 2>nul

REM Her tohum kendi surecinde kosar ve bitince bir bayrak dosyasi birakir.
REM Bayrak dosyasi sayarak bekliyoruz: makinede baska Python programlari
REM acik olsa bile bu betik yalnizca kendi islerini bekler.
for %%S in (%TOHUMLAR%) do (
  echo  -^> tohum %%S baslatildi
  start "ALNS tohum %%S" /min cmd /c "python -m src.planning.run_alns --iterations 1500 --time-limit 9000 --seed %%S --forecast data\interim\talep_tahmini_ml.xlsx --output outputs\kosu_seed%%S.xlsx > outputs\kosu_seed%%S.log 2>&1 & echo bitti > outputs\bitti_%%S.flag"
)

set /a HEDEF=0
for %%S in (%TOHUMLAR%) do set /a HEDEF+=1

echo.
echo  Bekleniyor... (ilerlemeyi outputs\kosu_seed*.log dosyalarindan izleyebilirsiniz)
:BEKLE
timeout /t 30 /nobreak >nul
set /a BITEN=0
for %%S in (%TOHUMLAR%) do if exist outputs\bitti_%%S.flag set /a BITEN+=1
echo    !BITEN!/!HEDEF! tohum tamamlandi
if !BITEN! LSS !HEDEF! goto BEKLE

del /q outputs\bitti_*.flag 2>nul

echo.
echo ==========================================================
echo  3/3  En ucuz gecerli plan seciliyor
echo ==========================================================
python -m src.planning.select_best_plan
if errorlevel 1 (
  echo.
  echo HATA: Plan secilemedi. Ayrinti icin outputs\kosu_seed*.log dosyalarina bakin.
  pause
  exit /b 1
)

echo.
echo ==========================================================
echo  TAMAMLANDI
echo ==========================================================
echo  Plan : outputs\TASIMA PLANI.xlsx
echo  Log  : outputs\kosu_seed*.log
echo.
pause
