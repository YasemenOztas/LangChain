# TÜBİTAK Akademik Proje Simülasyonu

> **CrewAI** tabanlı çok-ajanlı (multi-agent) akademik simülasyon.
> Üç ajan (Araştırmacı · Yazar · Hakem) sıralı biçimde çalışır; hakem puanı düşük gelirse otomatik revizyon turu tetiklenir (en fazla `MAKS_REVIZYON_TURU` kadar).

---

## Mimari

```
Araştırmacı Ajanı
    │  literatür analizi & boşluklar
    ▼
Yazar Ajanı
    │  TÜBİTAK proje önerisi
    ▼
Hakem Ajanı
    │  akademik değerlendirme + puan (0-10)
    │
    ├─ puan >= 6.0 ──────────────▶  Rapor kaydet & bitir
    │
    └─ puan  < 6.0 ──────────────▶  [Revizyon Döngüsü — Max N Tur]
                                        Yazar (hakem geri bildirimiyle) → Hakem
                                            │
                                            ├──▶ puan >= 6.0  ──▶ Kabul & Bitir
                                            └──▶ tur < max     ──▶ Tekrar Revize Et
```

---

## Dosya Yapısı

| Dosya | Açıklama |
|---|---|
| config.py | Ajan model atamaları, eşik puanı ve `MAKS_REVIZYON_TURU` |
| main.py | Crew akışı, görevler, çok turlu revizyon döngüsü |
| requirements.txt | Python bağımlılıkları |
| .env.example | API anahtar şablonu (repoya girer) |
| .env | Gerçek API anahtarları (.gitignore'da) |
| raporlar/ | Otomatik oluşturulan tarih damgalı Markdown raporları |
| oturumlar/ | Terminal akışının kaydedildiği log dosyaları |

---

## Kurulum ve Çalıştırma

### 1. Sanal Ortam Oluştur

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarla

```bash
# .env.example'ı kopyala
copy .env.example .env        # Windows
cp .env.example .env          # macOS/Linux
```

Ardından `.env` dosyasını aç ve kullandığın sağlayıcının anahtarını gir:

```env
DEEPSEEK_API_KEY=sk-buraya-gercek-anahtari-gir
```

### 4. (İsteğe Bağlı) Model Değiştir

`config.py` dosyasını aç ve istediğin satırı düzenle:

```python
HAKEM_AI = "anthropic/claude-3-5-sonnet-20241022"
```

### 5. Çalıştır

```bash
# Varsayılan konu ile çalıştır
python main.py

# Özel konu ile çalıştır
python main.py "Kuantum Hesaplama Tabanlı Kriptografi"
python main.py "Enerji Verimli IoT Sensör Ağları"
python main.py "Tıbbi Görüntü Segmentasyonunda Transfer Öğrenme"
```

---

## Model Yapılandırması

`config.py` dosyasında LiteLLM formatında model string'leri kullanılır:

| Sağlayıcı | Format | Örnek |
|---|---|---|
| DeepSeek | deepseek/<model> | deepseek/deepseek-chat |
| OpenAI | openai/<model> | openai/gpt-4o |
| Anthropic | anthropic/<model> | anthropic/claude-3-5-sonnet-20241022 |
| Google | gemini/<model> | gemini/gemini-1.5-pro |

---

## Çıktılar

- **Terminalde**: Rich kütüphanesiyle renkli/biçimli özet panelleri
- **Dosyada**: `raporlar/` klasöründe tarih damgalı Markdown formatında tam akademik rapor
- **Loglarda**: `oturumlar/` klasöründe tüm terminal oturumunun metin kaydı

---

## Neden CrewAI?

> LangGraph düşük seviyeli graf mimarisi; CrewAI ise **rol/görev soyutlaması** sunar.
> Bu projede ajanların net akademik kimlikler (araştırmacı/yazar/hakem) üstlenmesi
> öncelikli olduğu için CrewAI seçilmiştir.
