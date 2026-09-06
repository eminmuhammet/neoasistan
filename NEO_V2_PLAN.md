# NEO V2 — Geliştirme Planı

Mevcut mimari korunuyor. Hiçbir şey baştan yazılmıyor; 137 test ve çalışan
yapı üzerine modüler ekleme yapılıyor.

---

## Önce: senin listendeki 3 teknik düzeltme

Bunları planın içine gömmek yerine ayrıca yazıyorum çünkü işin süresini ve
riskini doğrudan değiştiriyorlar.

### 1. Ekran görüşü için OCR'a gerek yok
Sen "ekran görüntüsü al → OCR yap → Claude'a gönder" demişsin. **OCR adımını
tamamen atlayabiliriz:** Claude görüntüyü doğrudan okuyor ve bunu OCR'dan çok
daha iyi yapıyor (düzeni, hangi butonun nerede olduğunu, hata penceresinin
bağlamını da anlıyor — OCR sadece düz metin verir). Bu, özelliği hem
basitleştiriyor hem kalitesini artırıyor. Tesseract/OCR bağımlılığı yok.

**Tek maliyet:** her ekran görüntüsü ~1.000-1.500 token. Göndermeden önce
küçültmek gerekiyor, yoksa 5 dolar hızlı erir.

### 2. Mouse/klavye kontrolü listenin en riskli maddesi
Diğer her şey "yanlış yaparsa can sıkıcı" seviyesinde. Bu madde "yanlış
yaparsa gerçek zarar" seviyesinde: yanlış yere tıklamak dosya siler, yanlış
kişiye mesaj gönderir, yanlış butona basar. LLM'in ekranı %100 doğru
yorumlamadığı durumlar olacak.

O yüzden bunu "3. sıraya" değil, **planlayıcı ve doğrulama döngüsü
kurulduktan sonraya** koydum. Çünkü güvenli hali şu: ekran gör → eylemi
öner → kullanıcıya göster → onay → uygula → **tekrar ekran görüntüsü alıp
gerçekten olması gerekeni yaptı mı diye doğrula.** Bu döngü olmadan yapılan
mouse kontrolü, tehlikeli bir oyuncak olur.

### 3. Plugin sistemine şimdilik karşıyım
`ToolRegistry` zaten plugin sistemi: yeni yetenek = yeni `Tool` sınıfı,
ana koda dokunmadan. Üçüncü parti plugin yazan kimse yokken bir plugin
yükleyici eklemek, sadece araya bir katman daha koyar. Gerçekten ihtiyaç
doğduğunda (başkası NEO'ya eklenti yazmak istediğinde) yapalım.

Aynı şekilde **kodlama modu** için de dürüst olayım: şu an benimle
(Claude Code) yaptığın şeyin daha zayıf bir kopyası olur. NEO'nun kod
yazması yerine, NEO'nun *seni* Claude Code'a bağlaması daha mantıklı.

---

## Sıralama: neden senin sıralamandan farklı

Senin sıran: EXE → görme → kontrol → hafıza → telefon.

**İtirazım EXE'nin başta olmasına.** Her yeni özellik yeni bağımlılık
getiriyor (`mss`, `pyautogui`, `pycaw`, `pypdf`, `fastapi`...). Paketleme,
bağımlılık listesi oturduktan sonra bir kez düzgün yapılırsa bir günlük iş;
her özellikten sonra tekrar tekrar yapılırsa her seferinde yeni DLL/hidden
import sorunlarıyla uğraşırız.

Ama günlük kullanmak istemeni de anlıyorum. Ortası şu:

| Sıra | İş | Süre | Risk | Neden burada |
|------|-----|------|------|--------------|
| 1 | Kişisel hafıza | ~0.5 gün | Düşük | Listedeki en iyi emek/değer oranı, sıfır yeni bağımlılık |
| 2 | Ekran görüşü | ~1 gün | Düşük | NEO'yu bambaşka bir şey yapan özellik, OCR'sız basit |
| 3 | **Asistan / Yardımcı modu** | ~1 gün | Düşük | Sonraki her riskli özelliğin oturacağı çerçeve — önce bu kurulmalı |
| 4 | Görev planlayıcı | ~2-3 gün | Orta | **V2'nin kalbi** — senin bahsettiğin döngü |
| 5 | Proaktif NEO | ~1 gün | Düşük | Planlayıcı varken kolay, JARVIS hissini asıl bu verir |
| 6 | **EXE + kurulum** | ~1 gün | Orta | Bağımlılıklar oturdu, artık paketlemeye değer |
| 7 | Bilgisayar kontrolü | ~2-3 gün | **Yüksek** | Mod sistemi + doğrulama döngüsü hazır, güvenli yapılabilir |
| 8 | Dosya zekâsı | ~1-2 gün | Düşük | |
| 9 | Medya + güvenlik paneli | ~1 gün | Düşük | Küçük, hızlı kazanımlar |
| 10 | Telefon/web paneli | ~2-3 gün | Orta | |
| 11 | E-posta | ~1 gün | Düşük | Senin OAuth kurulumuna bağlı |

**EXE'yi hemen istiyorsan** 5. sırayı 1'e alabiliriz — ama o zaman her yeni
özellikten sonra yeniden paketleme işi çıkacağını bilerek yapalım.

---

## Adım adım: hangi dosyaya ne

### 1. Kişisel hafıza

**Yeni dosyalar**
- `neo/memory/preference_store.py` — SQLite (`key`, `value`, `learned_at`,
  `source`). Takvim/konuşma deposuyla aynı desende.
- `neo/tools/preferences.py` — `remember_preference`, `recall_preferences`,
  `forget_preference` araçları.

**Değişecekler**
- `neo/core/agent.py` — `_current_date_context()` gibi, her istekte bilinen
  tercihleri sistem promptuna enjekte eden `_preference_context()`.
- `neo/main.py` — deponun kurulup registry'ye bağlanması.

**Bağımlılık:** yok.

**Dikkat:** NEO'nun neyi hatırlayacağına Claude karar vermeli ("Bursa'da
oturuyorum" → hatırla; "bugün başım ağrıyor" → hatırlama). Sistem promptuna
bu ayrımı yazacağız. Ayrıca hassas bilgi (şifre, TC, kart) asla
kaydedilmemeli — buna açık kural koyacağız.

---

### 2. Ekran görüşü

**Yeni dosyalar**
- `neo/tools/screen.py` — `capture_screen(region=None)`; `mss` ile ekran
  görüntüsü, `Pillow` ile küçültme (uzun kenar ~1400px) ve PNG→base64.

**Değişecekler**
- `neo/core/llm_client.py` — mesaj içeriğine **image bloğu** ekleyebilmek
  (şu an sadece metin gönderiyor).
- `neo/core/agent.py` — araç sonucu bir görüntüyse onu image bloğu olarak
  bir sonraki isteğe koyma.
- `neo/ui/main_window.py` — "👁 Ekrana bak" düğmesi.

**Bağımlılık:** `mss`, `Pillow`.

**Dikkat:** Ekran görüntüsü kişisel veri içerir (açık sekmeler, mesajlar).
Bu yüzden: sadece kullanıcı istediğinde çekilecek, otomatik/sürekli asla;
güvenlik panelinde "ekran erişimi" kaydı tutulacak.

---

### 3. Asistan modu / Yardımcı modu

İki yetki seviyesi. NEO her zaman **asistan modunda** başlar; tam yetki
şifreyle açılır ve 10 dakika kullanılmazsa kendiliğinden düşer.

**Yeni dosyalar**
- `neo/core/access_mode.py` — mod durumu, şifre doğrulama, boşta kalma
  sayacı, `ModeChanged` sinyali
- `neo/config/credentials.py` — şifrenin **tuzlu hash'i** (`hashlib.scrypt`,
  standart kütüphane). Şifre hiçbir yerde açık metin tutulmaz.
- `neo/ui/mode_badge.py` — sağ üstte sürekli görünen mod göstergesi +
  kilit açma penceresi

**Değişecekler**
- `neo/core/permissions.py` — izin kararı artık risk seviyesi **ve** aktif
  moda birlikte bakacak
- `neo/core/agent.py` — sistem promptuna aktif mod bilgisi (NEO "bunu
  asistan modunda yapamam, yardımcı moduna geçmen gerek" diyebilsin)
- `neo/ui/main_window.py` — mod rozeti, her etkileşimde sayaç sıfırlama

**Yetki tablosu**

| Risk | Asistan modu | Yardımcı modu |
|------|--------------|---------------|
| LOW (saat, sistem durumu, hava, sohbet, araştırma) | Serbest | Serbest |
| MEDIUM (uygulama açma, takvime yazma, dosya arama) | **Onay sorar** | Serbest |
| HIGH (kapatma, kilitleme, mouse/klavye kontrolü, mesaj gönderme, dosya değiştirme) | **Tamamen kapalı** | **Yine de onay sorar** |

Önerim şu iki satırda: asistan modunda HIGH işlemler onay penceresiyle bile
açılmasın — "tam yetki" istiyorsan zaten moda geçeceksin, yoksa yanlışlıkla
onaylama riski kalır. Ve yardımcı modunda bile HIGH işlemler onay sorsun —
"tam yetki" *yapabilme* yetkisi olsun, *sormadan yapma* yetkisi değil.
Katılmazsan ikisini de değiştirebiliriz.

**Şifre tasarımı — dikkat edilecek üç şey**

1. **Şifre sesle söylenmeyecek, sadece pencereye yazılacak.** Sesle
   söylersen şifre Whisper'dan geçip konuşma kaydına yazılır, o kayıt da
   OneDrive'a senkronize oluyor — yani şifren düz metin olarak buluta çıkar.
   Bu yüzden şifreyi sadece klavyeyle alacağız. NEO "yardımcı moduna geç"
   dediğinde pencereyi açar, şifreyi sen yazarsın.
2. **Hash'lenerek saklanır** (`scrypt` + rastgele tuz). Şifrenin kendisi ne
   diskte ne kayıt dosyalarında ne de bellekte kalıcı olarak durur.
3. **Deneme sınırı:** 5 yanlış denemeden sonra 5 dakika kilit. Aksi halde
   kısa bir şifre saniyeler içinde denenip kırılabilir.

**Boşta kalma sayacı**
- Sayaç her etkileşimde sıfırlanır (sesli komut, yazılı mesaj, düğme)
- 10 dakika hareketsizlik → sessizce asistan moduna düşer (sesli anons yok,
  sadece rozet değişir). Süre ayarlanabilir olacak.
- Düşüş anında bir görev çalışıyorsa görev güvenli noktada durdurulur

**Dikkat:** Bu, ev arkadaşı/aile gibi *kazara* erişime karşı iyi bir koruma.
Bilgisayarına fiziksel erişimi olan teknik birine karşı mutlak koruma
değildir (dosyaları düzenleyebilir). Gerçek koruma Windows kullanıcı
şifresi — bu mod onun üstüne bir kat daha ekliyor.

---

### 4. Görev planlayıcı — V2'nin kalbi

Bu, "gör → düşün → planla → uygula → doğrula → bildir" döngüsü.

**Yeni dosyalar**
- `neo/core/planner.py`
  - `create_plan(goal)` → Claude'a hedefi verip adımlara böldürür
  - `execute(plan)` → adımları sırayla çalıştırır, her adımdan sonra
    **sonucu doğrular** ("beklediğim oldu mu?"), gerekirse planı revize eder
  - Adım başına maksimum deneme + toplam süre sınırı (sonsuz döngü koruması)
- `neo/memory/task_store.py` — görevler ve adımlar (SQLite), böylece uzun
  görev NEO kapansa bile kaybolmaz
- `neo/ui/task_panel.py` — canlı ilerleme paneli (adımlar, ✓/⏳/✗)

**Değişecekler**
- `neo/core/agent.py` — "bu tek adımlık bir istek mi, yoksa görev mi?"
  ayrımı; görevse planlayıcıya devret.

**Bağımlılık:** yok.

**Dikkat:** Her adım yine mevcut izin sisteminden geçecek. Planlayıcı,
onay gerektiren bir adıma geldiğinde durup soracak — sessizce atlamayacak.

---

### 5. Proaktif NEO

**Yeni dosyalar**
- `neo/core/scheduler.py` — QTimer tabanlı zamanlayıcı (yeni bağımlılık
  gerektirmez), işler SQLite'ta saklanır
- `neo/tools/scheduling.py` — `schedule_task`, `list_scheduled`, `cancel_task`

**Örnek işler:** sabah brifingi (takvim + hava + gece biriken araştırma),
disk %90'ı geçince uyarı, uzun süredir açık kalan görev hatırlatması.

**Dikkat:** Proaktif konuşma rahatsız edici olabilir. Varsayılan: sadece
sabah brifingi + kritik uyarılar; gerisi kullanıcı açarsa.

---

### 6. EXE + kurulum

- `neo.spec` zaten hazır (PyInstaller, `--onedir`)
- Eklenecek: Inno Setup ile kurulum sihirbazı, ilk açılış yapılandırma
  ekranı (API anahtarı + şehir + ses seçimi), hata kayıt sistemi
- `%LOCALAPPDATA%\NEO` altına veri taşıma (proje klasörüne bağımlılığın
  kalkması)

---

### 7. Bilgisayar kontrolü (mouse + klavye)

**Yeni dosyalar**
- `neo/tools/computer_control.py` — `click`, `type_text`, `press_keys`,
  `move_mouse`, `drag`

**Bağımlılık:** `pyautogui` (FAILSAFE açık — mouse'u köşeye götürmek her
şeyi iptal eder)

**Güvenlik tasarımı (bu maddede pazarlık yok):**
- Tamamı HIGH risk → **sadece yardımcı modunda** çalışır, orada da onay ister
- Onay penceresinde **ne yapılacağının önizlemesi** (hangi koordinat, hangi
  metin yazılacak)
- Eylem sonrası **otomatik doğrulama** (yeni ekran görüntüsü)
- Panik tuşu (ör. Ctrl+Alt+Shift+Q) → tüm otomasyonu anında durdurur
- Mesaj gönderme, dosya silme, ödeme ekranları → her zaman ayrı onay

---

### 8-11. Kısa notlar

| Özellik | Yeni dosya | Bağımlılık | Not |
|---------|-----------|------------|-----|
| Dosya zekâsı | `neo/tools/document_search.py` | `pypdf`, `python-docx`, `openpyxl` | Önce metin çıkar + Claude özetlesin; semantik arama (embedding) sonraki aşama |
| Medya kontrolü | `neo/tools/media.py` | `pycaw` (ses); medya tuşları `ctypes` ile bağımlılıksız | En kolay maddelerden |
| Güvenlik merkezi | `neo/memory/audit_store.py` + `neo/ui/security_panel.py` | yok | Her araç çalıştırması kaydedilir; "son 24 saatte ne yaptın" sorusu cevaplanır |
| Telefon/web | `neo/web/server.py` | `fastapi`, `uvicorn` | **Mutlaka PIN/token korumalı** — bu panel bilgisayarını kontrol ediyor |
| E-posta | `neo/tools/gmail.py` | Google kütüphaneleri zaten kurulu | Gönderme her zaman onaylı; taslak serbest |

---

---

## Sohbet arayüzü — küçük ama günlük kullanımda hissedilen

**Yeni dosya değil**, `neo/ui/chat_view.py` üzerinde çalışma. ~0.5 gün.

- **Mesaj saati** — her balonun altında/yanında gönderilme saati (`14:32`).
  Konuşma geçmişi `ConversationStore`'da zaten `timestamp` ile saklanıyor,
  yani veri var; sadece balona yazdırılmıyor.
- **Mesajı kopyalama** — şu an metin fareyle seçilebiliyor
  (`TextSelectableByMouse`) ama uzun bir cevabı seçmek zahmetli. Balonun
  üstüne gelince beliren bir kopyala düğmesi ve/veya sağ tık menüsü.

Not: Bunlar 2026-09-06'da kullanıcı tarafından istendi, V2'ye bırakıldı.

---

## Yapmayacaklarımız (ve nedeni)

- **Bildirim merkezi:** Windows bildirimlerini okumak `UserNotificationListener`
  (WinRT) gerektiriyor; Python'dan güvenilmez çalışıyor ve kullanıcıdan ayrı
  izin istiyor. Değerine göre maliyeti yüksek — şimdilik listeden çıkarıyorum.
- **Sesli kesme (NEO konuşurken "dur" demek):** Gerçek çözümü akustik yankı
  bastırma (AEC) gerektiriyor; hoparlörden çıkan sesi mikrofon duyduğu için
  NEO kendi sesini "dur" sanabiliyor. **Kulaklık kullanırsan bugün bile
  çalışır** — hoparlörle çalışması için ciddi DSP işi gerekiyor.
- **Ses tonundan duygu analizi:** Bilimsel olarak güvenilmez. Bunun yerine
  senin dediğin gibi konuşma bağlamı kullanılacak ("kısa cevap ver" → oturum
  boyunca kısa cevap).

---

## Nihai mimari (senin çizimin + eklenenler)

```
                        ┌─────────────────┐
                        │   NEO AGENT     │
                        │   + PLANNER     │  ← V2'nin yeni beyni
                        └────────┬────────┘
                                 │
     ┌────────────┬──────────────┼──────────────┬────────────┐
     │            │              │              │            │
  ┌──▼───┐   ┌────▼────┐   ┌────▼────┐   ┌─────▼────┐  ┌────▼─────┐
  │VOICE │   │  BRAIN  │   │ MEMORY  │   │  VISION  │  │SCHEDULER │
  │STT/  │   │ Claude  │   │Konuşma  │   │  Ekran   │  │ Proaktif │
  │ TTS  │   │tool-use │   │Tercih   │   │ (Claude  │  │  görevler│
  │      │   │         │   │Takvim   │   │  görür)  │  │          │
  └──────┘   └────┬────┘   │Görevler │   └──────────┘  └──────────┘
                  │        └─────────┘
     ┌────────────┼────────────┬─────────────┬──────────────┐
     │            │            │             │              │
 ┌───▼───┐  ┌────▼────┐  ┌────▼────┐  ┌────▼─────┐  ┌─────▼─────┐
 │Windows│  │  Dosya  │  │Araştırma│  │ Mouse +  │  │   Web /   │
 │ araç. │  │ zekâsı  │  │ motoru  │  │ Klavye   │  │  telefon  │
 └───────┘  └─────────┘  └─────────┘  └────┬─────┘  └───────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │  İZİN + DOĞRULAMA KATMANI  │
                              │  asistan / yardımcı modu   │
                              │  onay · denetim · panik    │
                              └────────────────────────────┘
```

---

## Nereden başlayalım?

Bence **1 (kişisel hafıza) + 2 (ekran görüşü) + 3 (mod sistemi)** birlikte,
tek oturumda bitirilebilir ve NEO'yu bugünkünden çok farklı hissettirir.
Sonra 4'e (planlayıcı) geçeriz — asıl dönüşüm orada oluyor.

Sen "EXE önce" dersen ona da başlarım; sadece sonrasında her özellik
eklemesinde yeniden paketleme işi çıkacağını bilerek karar ver.
