# NEO

Windows için kişisel yapay zekâ masaüstü asistanı. Sesle veya yazıyla komut
verebileceğin, uygulama açabilen, sistem durumunu okuyabilen, internette
araştırma yapabilen ve doğal Türkçe konuşabilen bir AI Agent.

## Kurulum

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` dosyasını açıp `ANTHROPIC_API_KEY` değerini doldur.

## Çalıştırma

```bash
python -m neo.main
```

İlk açılışta yapman gerekenler:
1. Pencerede **"🎓 Neo'yu öğret"** butonuna bas, 3 kez "Neo" de (sesini
   öğrenmesi için — bkz. aşağıdaki not). Bu olmadan "Sürekli dinleme"
   kapalı kalır; push-to-talk (🎙 basılı tut) her zaman çalışır.
2. `.env` içine `ANTHROPIC_API_KEY` girmediysen veya Anthropic hesabında
   kredi yoksa, genel sohbet/araştırma çalışmaz ama saat/tarih/CPU/RAM/GPU/
   disk/hava durumu/takvim gibi temel işler API'siz de çalışır.

## Test

```bash
pytest
```

## Durum

### Phase 1 — Core ✅

- [x] Proje yapısı
- [x] Config (.env) yükleme
- [x] Loglama (secret redaction ile)
- [x] Claude API client + tool-use tabanlı Agent döngüsü
- [x] Risk seviyeli permission sistemi (LOW/MEDIUM/HIGH)
- [x] Koyu temalı, animasyonlu PySide6 GUI (bkz. "Arayüz" aşağıda)
- [x] İlk araçlar: `get_time`, `get_date`

### Phase 2 — Voice ✅

- [x] Push-to-talk mikrofon kaydı (`sounddevice`, 16kHz mono) — en güvenilir yöntem
- [x] Yerel/offline STT (`faster-whisper`, Türkçe) — model ilk kullanımda
      Hugging Face'ten indirilir (bkz. kurulum notu aşağıda)
- [x] TTS: önce Microsoft'un doğal nöral erkek sesi (`tr-TR-AhmetNeural`,
      online), yoksa otomatik olarak offline Windows SAPI ("Microsoft
      Tolga") sesine düşer
- [x] Kişiselleştirilmiş uyandırma kelimesi ("Neo"): genel amaçlı Whisper'ın
      "Neo"yu sürekli "Ne?"/"Ne o?" diye yanlış anlaması yüzünden, artık
      kullanıcının kendi ses örneklerini öğrenen bir akustik eşleştirici
      (MFCC + DTW, `neo/voice/keyword_spotter.py`) kullanılıyor
- [x] "Neo" algılanınca kısa bir bip sesi çalar (ekrana bakmadan da
      duyulduğunu anlaman için)
- [x] Konuşurken kendi sesini duyup kendine cevap vermesin diye, NEO
      konuşurken mikrofon otomatik susturuluyor; ayrıca yanlış tetiklenmeyi
      azaltmak için aktivasyon sonrası kısa bir "sağırlaşma" (cooldown) var
- [x] Komut yakalama push-to-talk gibi: sen susana kadar kaydedip öyle
      gönderiyor (sabit pencereyle erken kesmiyor)
- [x] Yazarak sorduğun şeyler de sesli okunuyor (yalnızca sesli girişe özel
      değil)
- [x] Kişilik: doğal/esprili sohbet, ismi "Neo" olduğu için çok nadir
      Matrix göndermeleri (sistem promptunda tanımlı — bu kısım Claude API
      gerektirir)
- [x] API'siz çalışan yerel katman: saat/tarih/CPU/RAM/GPU/disk/hava
      durumu/takvim soruları ve "nasılsın"/"neler yapabilirsin" gibi
      sorular Claude'a hiç gitmeden yanıtlanıyor (`neo/core/local_commands.py`)
- [x] GUI: 🎙 push-to-talk, ⏹ Durdur, "Neo'yu öğret" ve "Sürekli dinleme"
      anahtarı, canlı CPU/RAM/GPU göstergeleri, ℹ bilgi butonu

> **"Neo'yu öğret":** Sürekli dinleme anahtarı, sen "🎓 Neo'yu öğret"e basıp
> 3 kez "Neo" diyerek sesini kaydettirene kadar kapalı kalır. Kayıt çok
> sessiz/boşsa o örnek otomatik reddedilip yeniden istenir. Örnekler
> `data/wake_word_templates.npz` içinde makinende saklanır, git'e eklenmez.
> Yanlış tetiklenme yaşarsan "🎓 Neo'yu öğret"i tekrar çalıştırıp daha net
> örnekler vermek genelde düzeltir.
>
> STT modeli (`NEO_WHISPER_MODEL`, varsayılan `small`, ~500MB) ilk
> kullanımda Hugging Face'ten indirilir. Farklı bir boyut istersen `.env`
> içindeki `NEO_WHISPER_MODEL` değerini değiştir (`tiny`/`base` daha küçük
> ve hızlı, `medium`/`large-v3` daha doğru).

### Phase 3 — Windows Tools ✅

- [x] `open_application` / `open_website` — Start Menu kısayolları + registry
      "App Paths" + PATH taraması ile dinamik uygulama bulma (hard-code yok)
- [x] `get_system_info`, `get_cpu_usage`, `get_ram_usage`, `get_disk_usage` (`psutil`)
- [x] `get_gpu_status` — gerçek NVIDIA telemetrisi (`nvidia-ml-py`); GPU
      verisine erişilemiyorsa uydurmak yerine bunu açıkça söyler
- [x] `get_network_status`

### Phase 4 — Internet ✅

- [x] `get_weather` — Open-Meteo (API anahtarı gerekmez), bugün + yarın
      tahmini; "hava nasıl" gibi çıplak sorular API'siz de çalışır (şehir
      adı geçen sorular -- "Bursa'da hava nasıl" -- doğru şehri
      kullanabilmek için Claude'a gidiyor)
- [x] Web araştırma/arama: Claude'un yerleşik `web_search` aracı
      (ayrı bir arama API anahtarı gerekmez). Sistem promptu, "araştırma
      yap" gibi isteklerde KONU/ÖZET/BULGULAR/ANALİZ/SONUÇ/KAYNAKLAR
      formatında yapılandırılmış rapor üretmesini söylüyor

### Phase 5 — Memory (başlandı, kısmi) 🟡

- [x] Yerel takvim/not sistemi (SQLite, `neo/memory/calendar_store.py`):
      `add_calendar_note` / `get_calendar_notes` araçları
- [x] "Günaydın" dediğinde bugünün notlarını API'siz özetler
- [x] "Bugün/yarın boş muyum" gibi sorular API'siz cevaplanır (başka bir
      güne ait sorular -- "cuma boş muyum" -- tarih çözümlemesi için
      Claude'a gidiyor)
- [x] Not eklemek (`add_calendar_note`) şu an Claude gerektiriyor (doğru
      tarih/metin çıkarımı için) -- kredi yoksa çalışmaz
- [x] Kalıcı konuşma geçmişi (`neo/memory/conversation_store.py`): her
      konuşma SQLite'a yazılır, NEO yeniden başlatıldığında son 20 mesajı
      hatırlayarak devam eder
- [x] **Çoklu cihaz erişimi**: geçmiş, OneDrive klasörüne
      (`%OneDrive%\NEO`) yazılır — hem `conversations.db` hem de günlük
      okunabilir `konusmalar/YYYY-AA-GG.md` dosyaları olarak, böylece
      telefondan/başka bir bilgisayardan da okunabilir. `.env` içindeki
      `NEO_CONVERSATION_DIR` ile başka bir klasöre yönlendirilebilir
- [ ] Google Takvim senkronizasyonu: **kod hazır ama devreye almak senin
      elinde bir OAuth adımı gerektiriyor**, bkz. aşağıda
- [ ] Öğrenilen kullanıcı tercihleri ("adım şu", "şunu sevmem") henüz yok

> **Google Takvim'i açmak için** (bunu senin yapman gerekiyor, Google
> hesabı girişi içerdiği için NEO bu adımı senin adına yapamaz):
> 1. https://console.cloud.google.com/ adresinde bir proje aç (veya var
>    olanı kullan), "APIs & Services → Library" içinden **Google Calendar
>    API**'yi etkinleştir.
> 2. "APIs & Services → Credentials → Create Credentials → OAuth client
>    ID" ile **Desktop app** tipinde bir istemci oluştur.
> 3. İndirdiğin JSON dosyasını proje köküne `credentials.json` adıyla
>    koy (zaten `.gitignore`'da, git'e eklenmez).
> 4. NEO ilk kez bir not eklemeye çalıştığında bir tarayıcı penceresi
>    açılır, Google hesabınla giriş yapıp takvim izni verirsin; sonrasında
>    `data/google_token.json` içinde saklanır ve sessizce çalışır.
> `credentials.json` yoksa notlar sorunsuzca sadece yerel kalır, hata
> vermez.

### Phase 6 — Security (başlandı, kısmi) 🟡

- [x] Gerçek bir onay diyaloğu: `PermissionManager`'ın confirm callback'i
      artık GUI'ye bağlı (`MainWindow.confirm_action`) — MEDIUM/HIGH riskli
      bir araç çağrıldığında ekranda "Onay Gerekiyor" penceresi açılır;
      "Hayır" ya da pencereye hiç cevap verilmezse araç ÇALIŞMAZ
- [x] `lock_computer` (MEDIUM), `shutdown_computer` / `restart_computer`
      (HIGH) araçları eklendi — bunlar test ortamında YALNIZCA mock'lanmış
      testlerle doğrulandı, gerçek bir kapatma/yeniden başlatma hiç
      tetiklenmedi
- [ ] Dosya silme/değiştirme gibi diğer riskli işlemler henüz yok

> Onay penceresi gerçek kullanımda ilk kez senin gözünle görülecek — kod
> incelemesiyle mantığını doğruladım ama Qt penceresinin gerçekten doğru
> göründüğünü/davrandığını sen kontrol etmelisin.

### Araştırma modu 🔬

"**Neo, araştırma modu**" dediğinde açılır ("araştırma modunu kapat" ile
kapanır). Bu moddayken NEO her soruyu ciddi bir araştırma isteği sayar:
web_search ile birden fazla kaynağı karşılaştırır ve
KONU / KISA ÖZET / ANA BULGULAR / DETAYLI ANALİZ / SONUÇ / KAYNAKLAR
başlıklarıyla uzun bir rapor üretir (token limiti bu modda 4096'ya çıkar).

Rapor ekranda tam haliyle görünür ama **sesli olarak sadece en kritik 2-4
cümlelik özet okunur** — modelden cevabın ilk satırına `SESLİ ÖZET:` etiketi
eklemesi istenir, `extract_spoken_summary()` de TTS'e yalnızca o satırı verir.
Ekranda modun açık olduğunu gösteren bir "🔬 ARAŞTIRMA MODU" etiketi belirir.

### Phase 7 — Polish (başlandı) 🟡

- [x] `find_file` / `open_folder` araçları (sadece okuma/açma; silme yok)
- [x] Sistem tepsisi: pencereyi kapatmak uygulamayı kapatmaz, arka planda
      çalışmaya devam eder (tepsi menüsünden "Göster" / "Çıkış")
- [x] Windows açılışında otomatik başlatma anahtarı (HKCU\...\Run, yönetici
      hakkı gerektirmez, aynı düğmeyle kapatılabilir)
- [x] `run_neo.py` — çalışma dizininden bağımsız başlatıcı
- [x] PyInstaller paketleme (`neo.spec`) — bkz. aşağıdaki "EXE oluşturma"
- [ ] Hata yönetimi/performans son cilası

### EXE oluşturma

```bash
.venv\Scripts\pyinstaller neo.spec --noconfirm
```

Çıktı: `dist\NEO\NEO.exe`. `--onefile` yerine `--onedir` kullanılıyor: paket
PySide6 ve ctranslate2 native DLL'leri taşıyor, onefile bunları her açılışta
geçici klasöre çıkarmak zorunda kalır (daha yavaş açılış ve DLL yükleme
hatalarının sık kaynağı). Konuşma tanıma modeli pakete dahil DEĞİL; ilk
kullanımda Hugging Face önbelleğine indirilir.

## Arayüz

Koyu, yeşil ağırlıklı temalı, tam ekran açılan tek odaklı bir panel:
- Ortada, hacimli parçacık küresi (`NeuroVisual`): binlerce parçacıktan oluşan
  dönen bir kabuk. Silüetinde parlayan rim ışığı küreye derinlik verir; rengi,
  dönüş hızı ve nefes alışı duruma göre değişir
  (bekliyor/dinliyor/düşünüyor/konuşuyor/hata). Konuşurken ses seviyesiyle
  şişip parlar. Pencere gizlenince (tepsiye küçültme) animasyon durur.
  Tamamen numpy + QPainter ile çizilir, OpenGL gerekmez.
- Altta tam genişlikte kontrol çubuğu: 🎙 push-to-talk, ⏹ anında durdur,
  🎓 Neo'yu öğret, sürekli dinleme anahtarı ve canlı CPU/RAM/GPU çubukları
  (`StatsPanel`, 2.5 saniyede bir gerçek verilerle günceller — Claude'a
  gitmez, GPU yoksa "yok" yazar, asla sayı uydurmaz).
- 💬 butonu sohbet akışını ve giriş satırını sağdan açar (varsayılan gizli).
- ℹ butonuyla NEO hakkında kısa bilgi.

## Mimari

```
neo/
├── config/     Ayarlar ve .env yükleme
├── core/       Agent döngüsü, LLM client, context, permission, state,
│               local_commands (API'siz hızlı yol)
├── tools/      Claude tool-use ile çağrılabilen araçlar (registry tabanlı)
├── voice/      STT/TTS/wake-word/keyword_spotter (Phase 2)
├── memory/     SQLite tabanlı takvim/not deposu (Phase 5, kısmi)
└── ui/         PySide6 arayüzü (main_window, neuro_visual, stats_panel, theme)
```

Araç çağırma akışı klasik bir intent-router yerine Claude'un native
tool-use'unu kullanır: LLM hangi aracın gerektiğine karar verir, `PermissionManager`
riskli araçları onay akışından geçirir, `ToolRegistry` aracı çalıştırıp
yapılandırılmış sonucu geri LLM'e verir. `local_commands.py` bu akışın
ÖNÜNE geçen, sık kullanılan ve tek anlamlı sorular için API'ye hiç gitmeyen
bir kısayol katmanıdır.
