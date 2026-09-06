# NEO — Mevcut Durum ve Geliştirme Soruları

> Bu metin, projeyi hiç bilmeyen birine (ya da yeni bir yapay zekâ oturumuna)
> NEO'nun ne olduğunu, nerede olduğunu ve nereye gidebileceğini anlatmak için
> yazıldı. Sonundaki sorular, bir sonraki adımı birlikte seçmek için.

---

## NEO nedir?

Windows üzerinde çalışan, JARVIS'ten ilham alan ama gerçekçi bir **kişisel
yapay zekâ masaüstü asistanı**. Basit bir sohbet botu değil; sesle
konuşulabilen, bilgisayarla etkileşime giren, internette araştırma yapabilen
ve arka planda sürekli çalışan bir AI Agent.

**Teknoloji yığını:** Python 3.13 · PySide6 (arayüz) · Anthropic Claude API
(agent beyni, native tool-use) · faster-whisper (yerel konuşma tanıma) ·
Edge TTS + Windows SAPI (konuşma sentezi) · psutil / nvidia-ml-py (sistem
telemetrisi) · SQLite (takvim ve konuşma hafızası)

**Konum:** `C:\Users\USER\NEO` — 137 test geçiyor.

---

## Şu an neler çalışıyor?

### Sesli etkileşim
- **Push-to-talk** (🎙 basılı tut, konuş, bırak) — en güvenilir yöntem
- **"Neo" ile uyandırma** — kullanıcının kendi ses örneklerini öğrenen
  kişiselleştirilmiş akustik eşleştirici (MFCC + DTW). Genel amaçlı konuşma
  tanımayla "Neo"yu tespit etmek sürekli başarısız olduğu için (model onu hep
  "Ne?"/"Ne o?" diye duyuyordu) bu yaklaşım sıfırdan yazıldı. Kullanıcı 3 kez
  "Neo" diyerek sesini öğretiyor.
- Uyandırma algılanınca **bip sesi** çalıyor (ekrana bakmadan anlaşılsın diye)
- Komut yakalama, kullanıcı susana kadar bekliyor (sabit pencereyle kesmiyor)
- Cevaplar **doğal Türkçe erkek sesle** okunuyor (internet yoksa offline sese
  düşüyor); ⏹ Durdur ile ~1 saniyede kesilebiliyor
- NEO konuşurken mikrofon susturuluyor (kendi sesini duyup kendine cevap verme
  sorunu bu şekilde çözüldü)

### API olmadan da çalışan işler
Saat/tarih · CPU/RAM/GPU/disk durumu (gerçek telemetri) · hava durumu
(Open-Meteo) · "günaydın" deyince günün takvim notları · "bugün/yarın boş
muyum" · "nasılsın"/"neler yapabilirsin" gibi sohbet kalıpları

### Claude gerektiren işler
- Doğal, esprili sohbet (çok nadir Matrix göndermeleriyle — ismi Neo sonuçta)
- **🔬 Araştırma modu** — "Neo, araştırma modu" ile açılıyor. Birden fazla
  kaynağı web'de karşılaştırıp KONU/ÖZET/BULGULAR/ANALİZ/SONUÇ/KAYNAKLAR
  yapısında uzun rapor üretiyor; ekranda tamamı görünüyor ama **sesli olarak
  sadece kritik 2-4 cümlelik özet** okunuyor
- Takvime not ekleme (göreli tarihleri "yarın", "cuma" kendisi çözüyor)
- Uygulama/web sitesi açma, dosya arama, klasör açma

### Sistem ve altyapı
- **Kalıcı konuşma hafızası** → OneDrive klasörüne yazılıyor, telefondan ve
  diğer cihazlardan okunabiliyor; NEO yeniden başlayınca son konuşmaları
  hatırlıyor
- **Sistem tepsisi** — pencere kapatılınca arka planda çalışmaya devam ediyor
- **Windows açılışında otomatik başlatma** anahtarı
- **Risk seviyeli izin sistemi** — kilitle/kapat/yeniden başlat gibi işlemler
  ekranda onay penceresi olmadan asla çalışmıyor
- **Otomatik güncelleyici altyapısı** (sha256 doğrulamalı) — yayın adresi
  bekliyor
- Kaynak kullanımı: boştayken ~210 MB; konuşma modeli 5 dakika kullanılmazsa
  bellekten atılıyor

---

## Bilinçli tasarım kararları

1. **Asla veri uydurulmaz.** GPU sıcaklığına erişilemiyorsa "erişemiyorum"
   denir, tahmini bir sayı verilmez. Araştırmada kaynak uydurulmaz.
2. **API'siz de yaşayabilmeli.** Sık ve tek anlamlı sorular Claude'a hiç
   gitmeden yerel olarak cevaplanır — hem bedava hem internetsiz çalışır.
3. **Riskli işlem = canlı insan onayı.** Onay penceresine tıklanmadan
   çalışmaz; kimse başında yoksa işlem yapılmaz.
4. **Intent router yerine LLM tool-use.** Elle yazılmış anahtar kelime
   eşleştirme yerine Claude'un kendi araç seçimi kullanılıyor.

---

## Eksikler / bilinen sınırlar

- **EXE paketi henüz üretilmedi** (PyInstaller yapılandırması hazır, onay
  bekliyor)
- **Google Takvim senkronizasyonu** kodu hazır ama OAuth kurulumu yapılmadı
- **Telefon sürümü yok** — konuşma geçmişi okunabiliyor ama uygulama yok
- Öğrenilen kullanıcı tercihleri ("adım şu", "şunu sevmem") hafızası yok
- Uyandırma kelimesi kalitesi mikrofona çok bağlı (daha iyi mikrofonla test
  edilecek)
- Dosya silme/taşıma gibi riskli dosya işlemleri bilinçli olarak yok
- Ekran görüntüsü/OCR, medya kontrolü, e-posta, bildirimler, plugin sistemi
  gibi ileri özellikler yok

---

## Sorular — bundan sonra ne yapalım?

**Öncelik**
1. Önce hangisi: EXE + kurulum/otomatik güncelleme mi, yoksa telefon erişimi
   mi? (Telefon için en sorunsuzu, NEO'nun bir web arayüzü sunması — o zaman
   telefonda hiç güncelleme derdi olmaz.)
2. Uyandırma kelimesi hâlâ sorun çıkarırsa: gerçek bir wake-word motoruna
   (Porcupine/openWakeWord) geçelim mi, yoksa push-to-talk yeterli mi?

**Yetenekler**
3. NEO'nun ekranı görmesini ister misin? (ekran görüntüsü + OCR → "bu hatayı
   çöz", "bu ekranda ne yazıyor")
4. Medya/Spotify kontrolü, e-posta okuma, bildirim yönetimi — hangisi
   gerçekten günlük hayatta işine yarar?
5. Otomasyon: "her sabah 9'da bana günün özetini oku" gibi zamanlanmış
   görevler ister misin?
6. Kullanıcı tercih hafızası: NEO senin hakkında öğrendiklerini (isim,
   alışkanlıklar, tercih ettiğin şehir) kalıcı hatırlasın mı?

**Kalite**
7. Araştırma modu raporları yeterince derin mi, yoksa daha uzun/daha çok
   kaynaklı mı olsun?
8. Arayüzde eksik hissettiğin ne var? (Referans olarak Iron Man HUD teması
   verilmişti — dairesel göstergeler ve sohbet balonları eklendi, daha ileri
   gidilsin mi?)
9. Sesin tonu/hızı/karakteri iyi mi? Farklı bir ses denenmeli mi?

**Sürdürülebilirlik**
10. Kod GitHub'a taşınsın mı? (yedek + otomatik güncelleme + ileride telefon
    sürümü için gerekli)
