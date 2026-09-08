# NEO — Kişisel Yapay Zeka Asistanı

## Proje Özeti
NEO, Windows masaüstü için PySide6 tabanlı kişisel AI asistanı. Ses kontrolü (uyandırma kelimesi, push-to-talk, TTS), araç çağrısı (dosya sistemi, takvim, Gmail, medya kontrolü, bilgisayar otomasyonu) ve tam ekran animasyonlu UI içeriyor.

## Geliştirme Ortamı
- **Python**: `.venv\Scripts\python.exe` (proje kökündeki venv)
- **Build**: `pyinstaller neo.spec -y` → `dist/NEO/NEO.exe`
- **Çalıştırma (kaynak)**: `python -m neo.main` veya test için `NEO_NO_HOTKEY=1 python -m neo.main`
- **Testler**: `pytest` (622 test, hepsi yeşil olmalı)
- **Platform**: Windows 11, PowerShell

## Kritik Mimari Notlar

### Async Döngü
`qasync` ile Qt event loop üzerinde asyncio çalışıyor. **Tüm UI güncellemeleri ve `asyncio.ensure_future` çağrıları Qt main thread'inde yapılmalı.** Background iş için `asyncio.to_thread` kullanılır.

### Render Pipeline (`neo/ui/neuro_visual.py`)
- 4000 parçacıklı küre, **sabit 400×400 iç çözünürlükte** render edilir
- Qt `SmoothTransformation` ile widget boyutuna ölçeklenir (ekran boyutundan bağımsız maliyet)
- Render ayrı bir `QThread`'de çalışır → UI bloke olmaz
- `_box_blur1`: tek-pass separable box blur (cumsum trick, allocation-free)
- IDLE: ~10 fps, SPEAKING: ~25 fps hedefi

### Ses/Uyandırma
- Uyandırma kelimesi: `KeywordSpotter` + `WakeWordListener` (DTW tabanlı, kayıtlı şablonlarla karşılaştırma)
- Push-to-talk: `PushToTalkRecorder` → Whisper STT (CPU)
- TTS: Windows MCI (`winmm`) — gerçek zamanlı RMS erişimi yok, konuşma görselleştirmesi simüle edilir

### Güvenlik / Mod
- `AccessMode`: ASSISTANT (kısıtlı) / HELPER (parola ile açılır)
- `PermissionManager`: araç çağrılarını MEDIUM/HIGH risk için onay ister
- `AuditStore`: tüm araç çağrılarını kaydeder

### Single-Instance
`main.py`'da Windows named mutex (`Global\NEO_SingleInstance_v1`) ile ikinci başlatma engellenir.

## Önemli Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `neo/main.py` | Giriş noktası, tüm bileşenlerin wiring'i |
| `neo/ui/main_window.py` | Ana pencere, tam ekran layout |
| `neo/ui/neuro_visual.py` | Parçacık küresi animasyonu |
| `neo/ui/theme.py` | QSS tema (koyu yeşil) |
| `neo/core/agent.py` | LLM döngüsü, araç çağrısı |
| `neo/core/access_mode.py` | ASSISTANT/HELPER mod yönetimi |
| `neo/core/planner.py` | Görev planlayıcı (TaskPlanner) |
| `neo/config/version.py` | Sürüm (`__version__`) — build öncesi artır |
| `neo/config/settings.py` | Yapılandırma (model, dizinler vb.) |

## MainWindow Backend API
Backend, UI'a şu metodlarla bağlanır:
```python
window.set_access_mode(mode: AccessMode)           # mod rozeti
window.set_task_progress(task_name, step, total)   # ilerleme etiketi
window.show_proactive_notification(text, duration_ms=4000)  # toast
window.set_audit_store(audit_store: AuditStore)    # audit referansı
```

## Build & Release
1. `neo/config/version.py` içinde `__version__` artır
2. NEO.exe çalışıyorsa kapat (dist klasörü kilitlenir)
3. `pyinstaller neo.spec -y`
4. `git add neo/config/version.py && git commit && git push`

## Bilinen Kısıtlamalar
- Whisper her kullanımda yeniden yükler (60s idle sonra `_release_idle_resources` ile boşaltılır)
- TTS gerçek zamanlı ses seviyesi vermez → konuşma görselleştirmesi `sin(t)` formülü ile simüle edilir
- GPU istatistiği `nvidia-ml-py` gerektirir, yoksa "yok" gösterilir
- `_gauss_blur` (3-pass) eskiden 1080p'de ~1 sn/kare alıyordu — sabit-çözünürlük render bunu çözdü
