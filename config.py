"""
config.py — Ajan Bazlı Model Tanımları
=======================================
crewai.LLM nesneleri kullanılır; böylece max_tokens gibi parametreler
doğrudan burada ayarlanır — ajan kodları rol ismiyle çağırır, model
detaylarına dokunmaz.

Model string'leri LiteLLM formatındadır: 'saglayici/model-adi'
  deepseek/deepseek-chat
  openai/gpt-4o
  anthropic/claude-3-5-sonnet-20241022
  gemini/gemini-1.5-pro
"""

from crewai import LLM

# ---------------------------------------------------------------------------
# Araştırmacı Ajanı — Literatür taraması; çıktısı orta uzunlukta
# ---------------------------------------------------------------------------
ARASTIRMACI_AI = LLM(
    model="deepseek/deepseek-chat",
    max_tokens=80000,
)

# ---------------------------------------------------------------------------
# Yazar Ajanı — TÜBİTAK formatında proje önerisi; uzun çıktı üretir
# ---------------------------------------------------------------------------
YAZAR_AI = LLM(
    model="deepseek/deepseek-chat",
    max_tokens=80000,
)

# ---------------------------------------------------------------------------
# Hakem Ajanı — En uzun çıktıyı o üretir (kriterler + revizyon maddeleri)
# ---------------------------------------------------------------------------
HAKEM_AI = LLM(
    model="deepseek/deepseek-chat",
    max_tokens=80000,
)

# Baska bir modele gecmek istersen sadece su blogu degistir, ajan kodu degismez:
# HAKEM_AI = LLM(model="anthropic/claude-3-5-sonnet-20241022", max_tokens=8000)
# ARASTIRMACI_AI = LLM(model="openai/gpt-4o", max_tokens=8000)

# ---------------------------------------------------------------------------
# Değerlendirme Eşiği — Hakem bu puanın altında verirse revizyon turu açılır
# ---------------------------------------------------------------------------
HAKEM_ESIK_PUANI: float = 6.0   # 10 üzerinden

# ---------------------------------------------------------------------------
# Maksimum Revizyon Tur Sayısı — Hakem puanı eşik altındaysa en fazla kaç kez
# revizyon döngüsü çalıştırılacak
# ---------------------------------------------------------------------------
MAKS_REVIZYON_TURU: int = 5
MAX_REVIZYON_TURU: int = MAKS_REVIZYON_TURU  # Alternatif isimlendirme için takma ad

