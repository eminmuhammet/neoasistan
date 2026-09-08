# NEO — Claude Code çalışma notları

Windows için Türkçe konuşan, sesli masaüstü yapay zekâ asistanı.
Python + PySide6 + Claude API. Paketlenmiş `.exe` olarak dağıtılıyor.

## KRİTİK: Bu repo aynı anda birden fazla oturumda açık olabilir

Bu projede eş zamanlı olarak farklı Claude Code oturumları çalışabiliyor
(örn. biri backend/V2 özellikleri, biri arayüz yeniden tasarımı üzerinde —
bu dosyanın kendisi bile iki oturumun aynı anda yazmasıyla çakıştı).

- **Her zaman** commit atmadan önce `git status` / `git diff` ile neyin
  değiştiğini kontrol et — üzerinde çalışmadığın dosyalarda değişiklik
  görürsen bu başka bir oturumun işidir, dokunma, geri alma.
- **Asla `git add -A` kullanma.** Sadece kendi değiştirdiğin dosyaları açıkça
  isimlendirerek stage et.
- Bir dosya diğer oturum tarafından o an düzenleniyorsa geçici olarak
  syntax hatası verebilir (yarım kayıt) — birkaç saniye bekleyip tekrar dene,
  paniklemeyip dosyaya kendin dokunma.
- Bir dosyanın "sahibi" belli değilse (örn. `neo/ui/*` bir arayüz oturumunda
  aktif değişiyorsa), o dosyayı düzenlemek yerine gerekli değişikliği
  (metot imzası, veri şekli) tarif edip diğer oturuma
  `mcp__ccd_session_mgmt__send_message` ile ilet; `list_sessions` /
  `list_events` ile diğer oturumların durumunu görebilirsin.
- UI↔backend bağlantısı gerekiyorsa **post-hoc setter** deseni kullan
  (`permissions.set_confirm(...)`, `mode_manager.set_on_change(...)`,
  `planner.set_on_progress(...)`, `window.set_access_mode(...)` gibi) — bir
  tarafın constructor imzasına bağımlı olmadan, ikisi de kendi hızında
  ilerleyebilsin diye.
- Commit mesajı sonunda: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
  PR açıklaması sonunda: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

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

`core/` içindeki önemli parçalar: `Agent` (tool-use döngüsü),
`PermissionManager` + `AccessModeManager` (asistan/yardımcı modu),
`TaskPlanner` (planla→uygula→doğrula→**başarısızsa revize et**→devam et),
`Scheduler` + `DiskSpaceWatcher` (proaktif görevler), `local_commands.py`
(API'siz hızlı yol — saat/tarih gibi sık ve tek anlamlı sorular).

**Araç çağırma akışı ve risk seviyeleri:** LLM `ToolRegistry`'deki bir aracı
seçer → `PermissionManager.check` risk seviyesine göre onay ister/reddeder →
`ToolRegistry.execute` çalıştırır ve kayıtlı **audit hook** üzerinden otomatik
loglar. Loglama kasıtlı olarak Agent'ın kendi döngüsünde değil registry'de —
böylece `local_commands.py`'nin hızlı yolu dahil TÜM çağıranlar kapsanıyor
(bir dönem sadece Agent'ta loglanıyordu ve hızlı yoldan geçen "saat kaç?"
gibi sorular hiç kayda geçmiyordu).

- **LOW**: her zaman izinli.
- **MEDIUM**: yardımcı modunda otomatik onaylı, asistan modunda kullanıcı
  onayı ister (ör. ekrana bakma, belge okuma, fare/klavye kontrolü). Fare/
  klavye araçları (`computer_control.py`) kasıtlı olarak burada — her tek
  tıklama/tuş için ayrı onay istemek çok adımlı bir otomasyonu kullanılamaz
  hale getiriyordu; panik tuşu (Ctrl+Alt+Shift+Q) ve fareyi ekran köşesine
  götürme hâlâ her an devreye giriyor.
- **HIGH**: asistan modunda tamamen reddedilir; yardımcı modunda bile HER
  SEFERİNDE onay ister — ekranı izleyerek geri alınamayacak eylemler için
  (kapatma/yeniden başlatma, gerçek bir e-posta gönderme).

Yeni bir tool eklerken risk seviyesini "ne geri alınamaz / kimin gözünden
kaçabilir" sorusuna göre seç, sadece "ne kadar tehlikeli hissettiriyor"a göre
değil.

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

## Donanım kısıtları (yerel model/GPU işleri için önemli)

Geliştirme makinesi: GTX 1650 (4GB VRAM), Intel i5-10300H (4 çekirdek/8 iş
parçacığı), 15.8GB RAM. Yerel bir LLM/görsel model eklemeyi düşünürken bu
sınırları hesaba kat — 4GB VRAM'e sığan modeller (3-8B, 4-bit) pratik üst
sınır; büyük/kaliteli modeller bu donanımda gerçekçi değil. Whisper zaten
CPU'da çalışıyor (`whisper_device=cpu` varsayılan) — GPU'yu sürekli tüketen
tek şey render/composite tarafı, backend hiçbir şeyde sürekli GPU kullanmaz.

## Gizli veri

`credentials.json`, `token.json`, `.env` ve `data/` gitignore'da. Geniş bir
`git add` sonrası neyin stage'lendiğini kontrol et.
