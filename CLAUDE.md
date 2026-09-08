# NEO — Claude Code çalışma notları

Windows için Türkçe konuşan, sesli masaüstü yapay zekâ asistanı.
Python + PySide6 + Claude API. Paketlenmiş `.exe` olarak dağıtılıyor.

## Komutlar

Sanal ortam `.venv`, **her zaman onun Python'ı kullanılmalı** (sistem
Python'ında bağımlılıklar yok):

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe run_neo.py
```

Paketleme (PyInstaller, çıktı `dist/NEO/NEO.exe`):

```bash
.venv/Scripts/python.exe -m PyInstaller neo.spec --noconfirm
```

**Build almadan önce çalışan NEO'yu kapat.** Açık uygulama `dist/` altındaki
dosyaları kilitler ve PyInstaller `PermissionError` ile düşer:

```bash
powershell -Command "Get-Process NEO -ErrorAction SilentlyContinue | Stop-Process -Force"
```

Sürüm numarası tek yerde: `neo/config/version.py`. Hakkında penceresi ve
güncelleme kontrolü ikisi de oradan okur, elle başka yere yazma.

## Dil

- **Kullanıcıya görünen her şey Türkçe**: arayüz metinleri, log mesajları,
  hata metinleri, commit mesajları.
- **Kod içi yorumlar İngilizce.**
- Commit mesajlarında ASCII kullan (Türkçe karakterler konsol kodlamasında
  bozuluyor); dosya içeriğinde Türkçe karakter serbest.

## Yapı

```
neo/
├── core/    Agent, state machine, updater, access_mode
├── tools/   Claude tool-use ile çağrılan araçlar (registry tabanlı)
├── voice/   STT / TTS / wake-word / keyword_spotter / chime
├── memory/  SQLite takvim & not deposu, audit store
├── ui/      PySide6 arayüzü
├── web/     Opt-in LAN paneli (token korumalı, varsayılan kapalı)
└── config/  settings, version, autostart
```

**Import yönü kuralı: `neo.voice` asla `neo.config`'i import etmez.**
Ters yön (`config` → `voice`) güvenli ve `settings.py` bunu kullanıyor.
Bu yönü bozarsan döngüsel import çıkar.

## Ses hattı — dikkat edilecekler

Uyandırma iki aşamalı: `KeywordSpotter` (MFCC + DTW, kullanıcının kendi
kayıtlarına karşı) ucuz ön filtre, ardından Whisper ile ifade doğrulaması.
Akustik eşleştirme tek başına bu kullanıcının sesinde ayrışmadı, o yüzden
asıl kararı transkript veriyor.

**Eşik tek kaynaktan gelir.** `settings.wake_sensitivity`,
`wake_word.WAKE_PHRASE_THRESHOLD`'dan türetilir. Bir dönem iki ayrı sayıydı
ve `main.py` ayarı geçirdiği için sabiti değiştirmek çalışma zamanında
hiçbir şeyi değiştirmiyordu — kullanıcı ifadeyi tekrar tekrar söylemek
zorunda kalıyordu. İkisini tekrar ayırma.

**Eşikleri tahminle değil, `logs/neo.log`'daki gerçek transkriptlerle ayarla.**
Whisper "Neo uyan"ı `'uyan.'`, `'Ne o ya?'`, `'Ne yok, uyan.'` diye
döndürebiliyor; temiz metinden türetilen eşikler gerçek denemelerin hepsini
reddediyor. `tests/test_wake_confirmation.py` bu ayrışmayı test ediyor.

**Komut yakalamanın her çıkışı arayüze haber vermeli.** Uyandırma anında
arayüz "Dinliyor"a geçer ve oradan yalnızca komut callback'i çıkarır. Sessizce
dönen bir yol (az konuşma, VAD boş, boş transkript, bozuk tanıyıcı, susturma)
arayüzü sonsuza kadar "Dinliyor"da asılı bırakır. `tests/test_wake_command_always_reports.py`
bu yolları koruyor.

Mikrofon, NEO konuşurken susturulur (`_speaking` bayrağı) — yoksa kendi sesini
komut olarak yakalar. Susturma kalktığında açık komut yakalaması **silinmemeli**.

## Arayüz

`neo/ui/neuro_visual.py` — merkezdeki parçacık küresi. Tamamen numpy +
QPainter, **OpenGL yok**.

- Kare `self._render_frame()` içinde **arka plan thread'inde** üretilir,
  ana thread'e `_frame_ready` sinyaliyle döner. Ana thread'de render etme:
  olay döngüsünü bloke eder ve ses tarafındaki `asyncio.sleep` zamanlamalarını
  bozar (uyandırma kaydı bu yüzden bozulmuştu).
- Blur maliyeti O(S²). `_RENDER_SIZE` sabit tutulur, Qt `SmoothTransformation`
  ile ölçekler — widget/ekran boyutuna göre büyütme yapılmaz.
- Blur **tek kanallı yoğunluk haritasında** çalışır, renk blur'dan sonra bir
  kez uygulanır. Üç kanalı ayrı blur'lamak kareyi 90 ms'ye çıkarıyordu.
- Parçacıklar `_splat()` ile **bilinear alt-piksel** dağıtılır. Tam piksele
  yuvarlamak noktaları sert kenarlı karelere çeviriyor ve küre dönerken
  zıplatıyor.
- Derinlik hissini **rim (Fresnel) ışığı** veriyor, sadece derinlik gölgelemesi
  küreyi düz bir disk gibi gösteriyor.
- NEO ömrünün çoğunu boşta geçirir; `_STATE_TICK_MS` boşta kasıtlı olarak
  düşük. Kare süresinden hızlı tick atmak boşuna CPU yakar.

Ölçüm için `scratchpad/bench_render.py` benzeri bir script yaz ve
`_render_frame`'i doğrudan çağır — tahmin etme, ölç.

## Test

75 dosya, ~639 test. `pytest -q` saniyeler sürüyor, her değişiklikten sonra
çalıştır.

Bu projede testler **regresyon kanıtı** olarak yazılıyor: docstring'inde
hatanın kullanıcıya nasıl göründüğü ve neden olduğu anlatılıyor. Yeni bir
davranış testi eklerken önce düzeltmeyi geçici geri alıp testin gerçekten
düştüğünü doğrula.

## Yorum üslubu

Bu kod tabanındaki yorumlar alışılmadık derecede kanıta dayalı: ölçülen
değerler, log zaman damgaları, reddedilen alternatifler yazılı. Bir sabiti
değiştirirken **neden o değer olduğunu** yaz, sadece ne yaptığını değil.
Çevredeki üslubu koru.

## Gizli veri

`credentials.json`, `token.json`, `.env` ve `data/` gitignore'da. Geniş bir
`git add` sonrası neyin stage'lendiğini kontrol et.
