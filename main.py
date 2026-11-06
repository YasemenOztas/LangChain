# Neden CrewAI (LangGraph değil)?
# LangGraph, düşük seviyeli, düğüm/kenar (node/edge) tabanlı bir graf mimarisi sunar;
# akış kontrolü ve durum yönetimi mühendisin sorumluluğundadır (mimari odaklı).
# CrewAI ise "rol" ve "görev" soyutlamasıyla çalışır; her ajanın bir kimliği (role, goal,
# backstory) vardır ve iş birliği bu roller üzerinden kurulur (rol odaklı).
# Bu projede ajanların net akademik roller üstlenmesi (araştırmacı/yazar/hakem)
# öncelikli olduğu için CrewAI tercih edilmiştir.

"""
main.py — Akademik Proje & TÜBİTAK Hakemi Simülasyonu
=======================================================
Akış:
  1. Araştırmacı  →  Konuyla ilgili literatür analizi ve boşluk tespiti
  2. Yazar        →  TÜBİTAK formatında proje önerisi metni
  3. Hakem        →  Akademik değerlendirme + 10 üzerinden puan
  4. [Koşullu]    →  Puan < eşik ise Yazar revizyon turuna döner
  5. Çıktı        →  Terminalde renkli özet + tubitak_degerlendirme_raporu.md
"""

import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from config import (
    ARASTIRMACI_AI,
    HAKEM_AI,
    HAKEM_ESIK_PUANI,
    MAKS_REVIZYON_TURU,
    YAZAR_AI,
)

# ---------------------------------------------------------------------------
# CrewAI telemetri / tracing uyarısını sustur
# ---------------------------------------------------------------------------
os.environ["CREWAI_TRACING_ENABLED"] = "false"
try:
    from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
    set_suppress_tracing_messages(True)
except Exception:
    try:
        from crewai.utilities.console_formatter import set_suppress_tracing_messages
        set_suppress_tracing_messages(True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Terminal ciktisini hem ekrana hem dosyaya yazan Thread-Safe Tee akisi
# ---------------------------------------------------------------------------

class _TeeStream:
    """
    sys.stdout'u iki hedefe ayni anda yazar:
      - gercek_stdout : kullanicinin terminali
      - dosya         : oturumlar/ klasorudeki log dosyasi
    Boylece ajan konusmalari hem terminalde gorunur hem dosyaya kaydedilir.
    threading.Lock ile çoklu thread'lerin (CrewAI event worker vs.) logları
    birbirine karıştırması önlenir.
    """

    def __init__(self, gercek_stdout, dosya):
        self._gercek = gercek_stdout
        self._dosya  = dosya
        self._lock   = threading.Lock()

    def write(self, metin: str) -> int:
        with self._lock:
            self._gercek.write(metin)
            try:
                # Kapsamlı ANSI kontrol dizilerini temizle (renkler, imleç kontrolleri vb.)
                temiz = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', metin)
                temiz = temiz.replace('\r\n', '\n').replace('\r', '\n')
                self._dosya.write(temiz)
            except Exception:
                pass
        return len(metin)

    def flush(self) -> None:
        with self._lock:
            self._gercek.flush()
            try:
                self._dosya.flush()
            except Exception:
                pass

    def __getattr__(self, ad):
        return getattr(self._gercek, ad)


def _wait_for_crew_completion(timeout: float = 1.5) -> None:
    """
    CrewAI'ın arka plandaki event bus ve konsol yazdırma thread'inin
    tamamlanmasını bekler; böylece sonuç panelleriyle çakışma yaşanmaz.
    """
    try:
        from crewai.events.utils.console_formatter import ConsoleFormatter
        ConsoleFormatter.crew_completion_printed.wait(timeout=timeout)
    except Exception:
        time.sleep(timeout)
    time.sleep(0.4)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Ortam degiskenleri & konsol
# ---------------------------------------------------------------------------
load_dotenv()
console = Console(width=120)  # Sabit genislik: paneller birbirine karismaz

# ---------------------------------------------------------------------------
# Yardımcı: Mevcut anahtarı tespit et ve LiteLLM için ortam değişkenini ata
# ---------------------------------------------------------------------------

def _setup_api_key(llm) -> None:
    """
    crewai.LLM nesnesinden saglayiciyi cikarir ve
    LiteLLM icin OPENAI_API_KEY ortam degiskenini atar.
    """
    model_str = llm.model if hasattr(llm, "model") else str(llm)
    provider = model_str.split("/")[0].lower()
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    key_name = env_map.get(provider)
    if key_name and os.getenv(key_name):
        os.environ["OPENAI_API_KEY"] = os.getenv(key_name)   # LiteLLM fallback


def _check_keys() -> None:
    """Kullanılan modellerin API anahtarlarının .env'de tanımlı olup olmadığını kontrol eder."""
    provider_key_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    missing = []
    for llm in [ARASTIRMACI_AI, YAZAR_AI, HAKEM_AI]:
        model_str = llm.model if hasattr(llm, "model") else str(llm)
        provider = model_str.split("/")[0].lower()
        key_name = provider_key_map.get(provider, "")
        if key_name and not os.getenv(key_name):
            missing.append(f"  • {key_name}  ({model_str})")

    if missing:
        console.print(
            Panel(
                "[bold red]Eksik API Anahtarı![/bold red]\n\n"
                "Aşağıdaki anahtarlar [bold].env[/bold] dosyasında boş:\n"
                + "\n".join(missing)
                + "\n\n[yellow].env dosyasını düzenleyip ilgili anahtarı gir,[/yellow]\n"
                "ardından programı yeniden çalıştır.",
                title="⚠  Yapılandırma Hatası",
                border_style="red",
            )
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Ajan Tanımları
# ---------------------------------------------------------------------------

def _create_agents() -> tuple[Agent, Agent, Agent]:
    """Üç akademik ajanı oluşturup döndürür."""

    arastirmaci_ai = Agent(
        role="Literatür Araştırmacısı",
        goal=(
            "Verilen mühendislik konusundaki temel problemleri, mevcut çözüm yöntemlerini "
            "ve literatürdeki boşlukları sistematik biçimde tespit etmek."
        ),
        backstory=(
            "Lisansüstü düzeyde bilgisayar mühendisliği eğitimi almış, IEEE ve ACM "
            "veritabanlarında yoğun literatür taraması yapan kıdemli bir araştırmacısın. "
            "Teknik detaylara hakim, öz ve sade bir dille yazıyorsun."
        ),
        llm=ARASTIRMACI_AI,
        verbose=True,
    )

    yazar_ai = Agent(
        role="Proje Yazarı",
        goal=(
            "Araştırmacının bulgularını alarak TÜBİTAK 1001/1002 formatına uygun, "
            "özgün ve ikna edici bir proje önerisi metni hazırlamak."
        ),
        backstory=(
            "TÜBİTAK projelerinde çok sayıda başarılı proje teklifi kaleme almış "
            "deneyimli bir akademik yazarsın. Teknik içeriği politika yapıcılara da "
            "anlaşılır kılacak net bir anlatım stilin var."
        ),
        llm=YAZAR_AI,
        verbose=True,
    )

    hakem_ai = Agent(
        role="TÜBİTAK Hakemi",
        goal=(
            "Sunulan proje önerisini akademik standartlarda tarafsız ve eleştirel bir "
            "gözle değerlendirmek; Özgün Değer, Yöntem ve Uygulanabilirlik kriterlerine "
            "göre 10 üzerinden puan vermek ve somut revizyon maddeleri çıkarmak."
        ),
        backstory=(
            "Onlarca TÜBİTAK projesini daha önce hakem sıfatıyla değerlendirmiş, "
            "uluslararası dergilerde editörlük yapmış sert ama adil bir akademisyensin. "
            "Eksik metodoloji, zayıf özgünlük iddiası ve ölçülemeyen hedeflere karşı "
            "özellikle hassassın."
        ),
        llm=HAKEM_AI,
        verbose=True,
    )

    return arastirmaci_ai, yazar_ai, hakem_ai


# ---------------------------------------------------------------------------
# Görev Tanımları
# ---------------------------------------------------------------------------

def _create_tasks(
    konu: str,
    arastirmaci_ai: Agent,
    yazar_ai: Agent,
    hakem_ai: Agent,
    revizyon_notu: str = "",
    literatur_ozeti: str = "",
) -> tuple[Task, Task, Task]:
    """
    Üç görevi sırayla tanımlar; her görevin çıktısı bir sonrakine beslenir.

    Parametreler
    ------------
    revizyon_notu   : Hakem çıktısı — dolu gelirse Yazar revizyon modunda çalışır.
    literatur_ozeti : İlk turun literatür araştırması metni — revizyon turunda
                      Yazar'a bağlam olarak açıkça enjekte edilir; böylece
                      literatür bağlamı revizyon sırasında kaybolmaz.
    """

    # Revizyon turunda hakem geri bildirimi + literatür özetini description'a ekle.
    # Bu yaklaşım, `context=[literatur_gorevi]` zincirinin kurulmadığı durumlarda
    # (revizyon crew'u araştırmacısız çalışırken) bağlamı metin olarak taşır.
    revizyon_eki = ""
    if revizyon_notu:
        literatur_bolumu = (
            f"\n\n---\n**[BAĞLAM] Orijinal Literatür Araştırması Özeti:**\n"
            f"{literatur_ozeti}\n"
            if literatur_ozeti
            else ""
        )
        revizyon_eki = (
            literatur_bolumu
            + f"\n\n---\n**[REVİZYON TURU] Hakem Geri Bildirimi:**\n{revizyon_notu}\n"
            "Yukarıdaki literatür özetine sadık kal ve hakem eleştirilerini "
            "dikkate alarak metni yeniden yaz."
        )

    # Görev 1: Literatür Araştırması
    literatur_gorevi = Task(
        description=(
            f"Konu: **{konu}**\n\n"
            "Bu mühendislik konusunda:\n"
            "1. Temel problemleri ve araştırma motivasyonunu açıkla.\n"
            "2. Mevcut başlıca yöntem ve yaklaşımları (en az 3-4 kategori) özetle.\n"
            "3. Literatürdeki açık sorunları ve boşlukları (research gap) net biçimde listele.\n"
            "4. Olası araştırma yönlerini belirt.\n\n"
            "Yanıtın yapılandırılmış, madde işaretli ve Türkçe olsun."
        ),
        expected_output=(
            "Şu bölümleri içeren yapılandırılmış bir Türkçe rapor:\n"
            "- Araştırma Motivasyonu\n"
            "- Mevcut Yöntemler ve Kategoriler\n"
            "- Literatür Boşlukları\n"
            "- Önerilen Araştırma Yönleri"
        ),
        agent=arastirmaci_ai,
    )

    # Görev 2: TÜBİTAK Proje Önerisi Yazımı
    proje_gorevi = Task(
        description=(
            "Araştırmacının literatür bulgularını kullanarak TÜBİTAK 1001/1002 formatına uygun "
            "bir proje önerisi metni hazırla. Metin şu bölümleri içermeli:\n\n"
            "1. **Proje Başlığı** — kısa, özgün ve çarpıcı.\n"
            "2. **Özet** (200-250 kelime) — projenin amacı, yöntemi ve beklenen katkısı.\n"
            "3. **Özgün Değer** — neden bu proje daha önce yapılmamış veya mevcut çalışmalardan "
            "   farkı ne?\n"
            "4. **Yöntem** — araştırma adımları, kullanılacak teknikler, veri kaynakları.\n"
            "5. **Beklenen Çıktılar ve Etki** — somut ve ölçülebilir hedefler.\n\n"
            "Metin akademik ama erişilebilir Türkçeyle yazılsın."
            + revizyon_eki
        ),
        expected_output=(
            "Başlık, Özet, Özgün Değer, Yöntem ve Beklenen Çıktılar bölümlerini içeren "
            "tam ve biçimlendirilmiş bir TÜBİTAK proje önerisi metni (Türkçe)."
        ),
        agent=yazar_ai,
        context=[literatur_gorevi],
    )

    # Görev 3: TÜBİTAK Hakem Değerlendirmesi
    hakem_gorevi = Task(
        description=(
            "Önündeki TÜBİTAK proje önerisini aşağıdaki kriterler çerçevesinde değerlendir:\n\n"
            "**Değerlendirme Kriterleri (her biri 0-10):**\n"
            "| Kriter            | Ağırlık |\n"
            "|-------------------|---------|\n"
            "| Özgün Değer       | %40     |\n"
            "| Yöntem Kalitesi   | %35     |\n"
            "| Uygulanabilirlik  | %25     |\n\n"
            "Her kriter için:\n"
            "- Güçlü yönleri belirt.\n"
            "- Zayıf yönleri ve eksikleri somut örneklerle eleştir.\n"
            "- Puanı gerekçelendir.\n\n"
            "Raporun sonunda sırasıyla:\n"
            "- Ağırlıklı **Genel Puan** ver (0-10, bir ondalık basamak).\n"
            "- Zorunlu revizyon maddelerini numaralı liste olarak yaz.\n"
            "- **ZORUNLU SON SATIR** — Yanıtının kesinlikle son satırı şu formatta olsun:\n"
            "  GENEL_PUAN: X.X\n"
            "  (X.X yerine gerçek puanı yaz; nokta kullan, virgül değil; başka metin ekleme.)"
        ),
        expected_output=(
            "Üç kriterli detaylı hakem raporu, gerekçeli puanlar ve revizyon maddeleri listesi. "
            "Metnin kesinlikle son satırı `GENEL_PUAN: X.X` formatında (örn. GENEL_PUAN: 7.5) olmalıdır."
        ),
        agent=hakem_ai,
        context=[proje_gorevi],
    )

    return literatur_gorevi, proje_gorevi, hakem_gorevi


# ---------------------------------------------------------------------------
# Puan Ayrıştırıcı
# ---------------------------------------------------------------------------

def _puani_ayristir(hakem_ciktisi: str) -> float | None:
    """
    Hakem çıktısından puanı ayrıştırır. İki strateji:
      1. `GENEL_PUAN: X.X` formatı (tercihli, istenen format)
      2. Fallback: "ağırlıklı genel puan" ifadesinin yanındaki sayı
    """
    # Strateji 1: istenen format (satır başı/sonu, büykük/küçük harf)
    m = re.search(
        r"GENEL_PUAN\s*:\s*(\d+(?:[.,]\d+)?)",
        hakem_ciktisi,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1).replace(",", "."))

    # Strateji 2: tablo veya metin içine gömülü puan
    # "Ağırlıklı Genel Puan: 6.5 / 10" veya "TOPLAM ... 6.0" gibi
    for pattern in [
        r"(?:ağırlıklı\s+)?genel\s+puan\s*:\s*(\d+(?:[.,]\d+)?)\s*(?:/\s*10)?",
        r"TOPLAM[^\d]*(\d+(?:[.,]\d+)?)",
    ]:
        m = re.search(pattern, hakem_ciktisi, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", "."))
            if 0 <= val <= 10:   # makul aralık kontrolü
                return val
    return None


# ---------------------------------------------------------------------------
# Sonuç Yazdırıcılar
# ---------------------------------------------------------------------------

def _bolum_yazdir(baslik: str, icerik: str, renk: str = "cyan") -> None:
    console.print()
    console.print(Rule(f"[bold {renk}]{baslik}[/bold {renk}]", style=renk))
    console.print(Markdown(icerik))


def _raporu_kaydet(
    konu: str,
    literatur: str,
    proje: str,
    hakem: str,
    puan: float | None,
    revizyon_sayisi: int | bool,
) -> Path:
    """Tüm çıktıları raporlar/ klasörüne tarih damgalı Markdown dosyasına kaydeder."""
    rapor_klasoru = Path("raporlar")
    rapor_klasoru.mkdir(exist_ok=True)
    simdi = datetime.now()
    dosya_adi = simdi.strftime("tubitak_degerlendirme_raporu_%Y-%m-%d_%H%M.md")
    rapor_yolu = rapor_klasoru / dosya_adi
    simdi_str = simdi.strftime("%Y-%m-%d %H:%M")
    puan_str = f"{puan:.1f}/10" if puan is not None else "Ayrıştırılamadı"

    if isinstance(revizyon_sayisi, int) and revizyon_sayisi > 0:
        revizyon_str = f"Evet — {revizyon_sayisi} tur revizyon yapıldı."
    elif revizyon_sayisi:
        revizyon_str = "Evet — revizyon yapıldı."
    else:
        revizyon_str = "Hayır — ilk turda geçti."

    rapor_yolu.write_text(
        f"""# TÜBİTAK Akademik Proje Değerlendirme Raporu
**Oluşturulma Tarihi:** {simdi_str}
**Konu:** {konu}
**Genel Puan:** {puan_str}
**Revizyon Turu:** {revizyon_str}

---

## 1. Literatür Araştırması

{literatur}

---

## 2. Proje Önerisi

{proje}

---

## 3. Hakem Değerlendirmesi

{hakem}

---
*Bu rapor CrewAI Multi-Agent Simülasyonu tarafından otomatik üretilmiştir.*
""",
        encoding="utf-8",
    )
    return rapor_yolu


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------

def calistir(konu: str) -> None:
    """
    Tam crew akışını çalıştırır.
    Sıra: Araştırmacı → Yazar → Hakem → [Koşullu Revizyon Turu]

    Terminal çıktısının tamamı (ajan konuşmaları dahil) oturumlar/ klasörüne
    ANSI renk kodları temizlenmiş biçimde kaydedilir.
    """
    # ── Oturum log dosyasını aç (Tee: terminale + dosyaya aynı anda yazar) ─────
    oturum_klasoru = Path("oturumlar")
    oturum_klasoru.mkdir(exist_ok=True)
    simdi = datetime.now()
    oturum_yolu = oturum_klasoru / simdi.strftime("calistirma_%Y-%m-%d_%H%M.txt")

    _log_dosyasi = oturum_yolu.open("w", encoding="utf-8")
    _log_dosyasi.write("=== TÜBİTAK Simülasyonu Çalıştırma Logu ===\n")
    _log_dosyasi.write(f"Konu   : {konu}\n")
    _log_dosyasi.write(f"Tarih  : {simdi.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_dosyasi.write("=" * 60 + "\n\n")

    _gercek_stdout = sys.stdout
    sys.stdout = _TeeStream(_gercek_stdout, _log_dosyasi)

    try:
        console.print(
            Panel.fit(
                Text.assemble(
                    ("🎓  TÜBİTAK Akademik Proje Simülasyonu\n", "bold white"),
                    ("CrewAI Multi-Agent Sistemi", "dim"),
                ),
                border_style="bright_blue",
                padding=(1, 4),
            )
        )
        console.print(f"\n[bold yellow]📌 Araştırma Konusu:[/bold yellow] {konu}\n")

        _check_keys()

        # Sağlayıcı bazlı API key ayarı (LiteLLM uyumu için)
        for llm in [ARASTIRMACI_AI, YAZAR_AI, HAKEM_AI]:
            _setup_api_key(llm)

        arastirmaci_ai, yazar_ai, hakem_ai = _create_agents()

        # ── İlk Tur ─────────────────────────────────────────────────────────────
        console.print(Rule("[bold blue]▶  1. TUR — İlk Değerlendirme[/bold blue]", style="blue"))

        literatur_gorevi, proje_gorevi, hakem_gorevi = _create_tasks(
            konu, arastirmaci_ai, yazar_ai, hakem_ai
        )

        crew = Crew(
            agents=[arastirmaci_ai, yazar_ai, hakem_ai],
            tasks=[literatur_gorevi, proje_gorevi, hakem_gorevi],
            process=Process.sequential,
            verbose=True,
        )

        console.print("[dim]Ajanlar çalışıyor — her rolün yanıtı aşağıda görünecek...[/dim]\n")
        crew.kickoff()
        _wait_for_crew_completion()

        literatur_ciktisi = literatur_gorevi.output.raw if literatur_gorevi.output else ""
        proje_ciktisi     = proje_gorevi.output.raw     if proje_gorevi.output     else ""
        hakem_ciktisi     = hakem_gorevi.output.raw     if hakem_gorevi.output     else ""

        puan = _puani_ayristir(hakem_ciktisi)
        revizyon_sayisi = 0

        # ── Revizyon Döngüsü (Puan < Eşik olduğu sürece MAKS_REVIZYON_TURU kadar devam eder) ──
        while puan is not None and puan < HAKEM_ESIK_PUANI and revizyon_sayisi < MAKS_REVIZYON_TURU:
            revizyon_sayisi += 1
            tur_no = revizyon_sayisi + 1  # 2. Tur, 3. Tur...

            console.print()
            console.print(
                Panel(
                    f"[bold red]Hakem puanı düşük: {puan:.1f}/10[/bold red]  "
                    f"(eşik: {HAKEM_ESIK_PUANI}/10)\n"
                    f"Revizyon Turu [bold]{revizyon_sayisi}/{MAKS_REVIZYON_TURU}[/bold] başlatılıyor...",
                    border_style="red",
                    title=f"🔁 Revizyon Turu {revizyon_sayisi} Tetiklendi",
                )
            )
            console.print(Rule(f"[bold red]▶  {tur_no}. TUR — Revizyon {revizyon_sayisi}[/bold red]", style="red"))

            _, proje_gorevi_rev, hakem_gorevi_rev = _create_tasks(
                konu,
                arastirmaci_ai,
                yazar_ai,
                hakem_ai,
                revizyon_notu=hakem_ciktisi,
                literatur_ozeti=literatur_ciktisi,
            )
            revizyon_crew = Crew(
                agents=[yazar_ai, hakem_ai],
                tasks=[proje_gorevi_rev, hakem_gorevi_rev],
                process=Process.sequential,
                verbose=True,
            )
            console.print(f"[dim]Revizyon ajanları çalışıyor (Revizyon {revizyon_sayisi}/{MAKS_REVIZYON_TURU})...[/dim]\n")
            revizyon_crew.kickoff()
            _wait_for_crew_completion()

            proje_ciktisi = proje_gorevi_rev.output.raw if proje_gorevi_rev.output else proje_ciktisi
            hakem_ciktisi = hakem_gorevi_rev.output.raw if hakem_gorevi_rev.output else hakem_ciktisi
            yeni_puan = _puani_ayristir(hakem_ciktisi)
            if yeni_puan is not None:
                puan = yeni_puan

        # ── Değerlendirme Sonuç Bildirimi ──
        if puan is not None and puan >= HAKEM_ESIK_PUANI:
            aciklama = (
                f"{revizyon_sayisi} tur revizyon sonucunda başarıyla kabul edildi."
                if revizyon_sayisi > 0
                else "Revizyon turu gerekmedi (İlk turda geçti)."
            )
            console.print(
                Panel(
                    f"[bold green]✅ Hakem puanı: {puan:.1f}/10[/bold green]  (eşik: {HAKEM_ESIK_PUANI}/10)\n{aciklama}",
                    border_style="green",
                    title="Değerlendirme Sonucu: Kabul",
                )
            )
        elif puan is not None and puan < HAKEM_ESIK_PUANI:
            console.print(
                Panel(
                    f"[bold red]❌ Maksimum revizyon sınırına ({MAKS_REVIZYON_TURU}) ulaşıldı.[/bold red]\n"
                    f"Son Hakem puanı: [bold]{puan:.1f}/10[/bold]  (eşik: {HAKEM_ESIK_PUANI}/10)\n"
                    "Proje revizyonlara rağmen hakem eşik puanının altında kaldı.",
                    border_style="red",
                    title="Değerlendirme Sonucu: Red / Yetersiz",
                )
            )
        else:
            console.print(
                "[yellow]⚠  Hakem çıktısından otomatik puan ayrıştırılamadı. "
                "Raporu manuel kontrol edin.[/yellow]"
            )

        console.print()
        console.print(Rule("[bold cyan]📋  SONUÇLAR[/bold cyan]", style="cyan"))
        _bolum_yazdir("1️⃣  Literatür Araştırması", literatur_ciktisi, "cyan")
        _bolum_yazdir("2️⃣  TÜBİTAK Proje Önerisi", proje_ciktisi, "green")
        _bolum_yazdir("3️⃣  Hakem Değerlendirmesi", hakem_ciktisi, "red")

        rapor_yolu = _raporu_kaydet(
            konu, literatur_ciktisi, proje_ciktisi, hakem_ciktisi, puan, revizyon_sayisi
        )
        puan_gosterge = (
            f"[bold {'green' if (puan or 0) >= HAKEM_ESIK_PUANI else 'red'}]"
            f"{puan:.1f}/10"
            f"[/bold {'green' if (puan or 0) >= HAKEM_ESIK_PUANI else 'red'}]"
        ) if puan else "[yellow]N/A[/yellow]"

        revizyon_bilgisi = (
            f"Evet ({revizyon_sayisi}/{MAKS_REVIZYON_TURU} tur)"
            if revizyon_sayisi > 0
            else "Hayır (İlk turda geçti)"
        )

        console.print()
        console.print(
            Panel(
                f"Genel Puan       : {puan_gosterge}\n"
                f"Revizyon Turu    : {revizyon_bilgisi}\n"
                f"Rapor Kaydedildi : [underline]{rapor_yolu.resolve()}[/underline]\n"
                f"Terminal Logu    : [underline]{oturum_yolu.resolve()}[/underline]",
                title="🏁  Simülasyon Tamamlandı",
                border_style="bright_blue",
            )
        )

    finally:
        sys.stdout = _gercek_stdout
        _log_dosyasi.close()


# ---------------------------------------------------------------------------
# Giriş Noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Komut satırından konu verilebilir: python main.py "Konu Başlığı"
    if len(sys.argv) > 1:
        KONU = " ".join(sys.argv[1:])
    else:
        KONU = (
            "Derin Öğrenme Tabanlı Anomali Tespiti için "
            "Hafif ve Açıklanabilir Model Mimarileri"
        )

    calistir(KONU)