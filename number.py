# -*- coding: utf-8 -*-
"""
O'zbekcha raqam diktovkasi uchun deterministik "slot-filling" darvoza.

Muammo: EOU (turn-detector) modeli GRAMMATIK tugaganlikni baholaydi, raqam
diktovkasi esa SANASH masalasi ("yetti yuz o'n yigirma ikki" = 5 raqam, passport
7 raqam bo'lishi kerak -> hali tugamagan). Model buni ishonchli qila olmaydi.

Yechim: call-center oqimida qaysi maydon yig'ilayotgani ma'lum (assistant savolini
o'zing boshqarasan). Foydalanuvchi aytgan raqam so'zlarini raqamga aylantirib,
sonini deterministik tekshirasan. Tugaganlikni model emas, shu darvoza hal qiladi.

Asosiy funksiyalar:
    uz_words_to_digits("yetti yuz o'n yigirma ikki")  -> "71022"
    should_endpoint(user_text, collecting="passport_num", eou_prob=...) -> bool

STT ning uch formatini ham qo'llaydi:
    bo'lak-so'z:    "yetti yuz o'n yigirma ikki yetmish uch" -> "7102273"
    raqamma-raqam:  "yetti bir nol ikki ikki yetti uch"      -> "7102273"
    yaxlit raqam:   "7102273"                                 -> "7102273"
    aralash/prefiks:"raqamim 91 695 34 56"                    -> "916953456"
    harflar (passport seriyasi) e'tiborsiz qoldiriladi:
                    "ab 1234567" / "a be bir ikki uch ..."    -> "1234567"

LiveKit oqimiga ulash (soddalashtirilgan):

    collecting = None       # hozir qaysi raqam maydoni yig'ilyapti (None = erkin nutq)

    def on_agent_turn(agent_text):          # agent savol berganda
        nonlocal collecting
        collecting = detect_field(agent_text)   # yoki o'z oqim holatingdan ber

    def on_user_eou(user_text, eou_prob):   # STT to'liq transkript + EOU ehtimoli
        if should_endpoint(user_text, collecting, eou_prob):
            collecting = None               # maydon to'ldi -> holatni tozala
            return respond()
        return keep_listening()             # sukut taymerini uzaytirib kutamiz

Muhim maslahatlar:
  - Raqam maydonlarida sukut taymerini uzaytir (masalan 1.5-2s) — darvoza "kutamiz"
    deganda ham odam raqamni tugatib ulgursin.
  - Kritik maydonlar (passport/karta/jshshir) uchun read-back tasdiq qo'sh:
    "raqamingiz ... to'g'rimi?" — noto'g'ri uzunlik shu yerda ushlanadi.
  - So'z bilan aytilgan raqamlarni ishonchli sanash ID uslublarida ishlaydi
    (raqamma-raqam, 2-xonali juft, yaxlit). Noaniq "yumaloq" o'qishlar
    ("uch yuz sakson olti") tabiatan noaniq — bunda read-back kafolat beradi.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Normalizatsiya (dataset generatoridagi bilan bir xil) — apostrof variantlari
# ASCII ' ga keltiriladi, kichik harf, punktuatsiya olib tashlanadi.
# ---------------------------------------------------------------------------
_APOS = "ʻʼ‘’`´"


def _normalize(text: str) -> str:
    if not text:
        return ""
    for ch in _APOS:
        text = text.replace(ch, "'")
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(ch for ch in text
                   if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"]))
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Raqam-so'z lug'ati. Apostrofsiz variantlar ham (STT apostrofni tushirishi mumkin).
# ---------------------------------------------------------------------------
# Nol (0) ning yozilish/STT variantlari — alohida qaymiz (leading zero uchun)
ZERO_WORDS = {"nol", "no'l", "nul", "noil", "nöl"}
UNITS = {
    "nol": 0, "no'l": 0, "nul": 0, "bir": 1, "ikki": 2, "uch": 3,
    "to'rt": 4, "tort": 4, "besh": 5, "olti": 6, "yetti": 7,
    "sakkiz": 8, "to'qqiz": 9, "toqqiz": 9,
}
TENS = {
    "o'n": 10, "on": 10, "yigirma": 20, "o'ttiz": 30, "ottiz": 30, "qirq": 40,
    "ellik": 50, "oltmish": 60, "yetmish": 70, "sakson": 80,
    "to'qson": 90, "toqson": 90,
}
HUND = {"yuz": 100}
# Diqqat: bu darvoza ID RAQAM ketma-ketligi uchun (passport/telefon/jshshir/karta).
# "ming" (1000) ID o'qishda uchramaydi — u faqat yil/summa uchun; shuning uchun
# lug'atga kiritilmagan (uchrasa chegara so'zi sifatida qaraladi). Yil tugaganligi
# raqam soni bilan emas, kontekst bilan (model) hal qilinadi.

# "lik" qo'shimchasi (operator kodi belgisi): user "to'qson birlik" (91-lik),
# "to'qsonlik" (90-lik) deydi — shu kompaniyaga tegishli degani. Bu raqam GURUHINI
# yakunlaydi: "to'qsonlik olti yuz..." -> 90 | 6.. (aks holda "to'qson olti"=96
# bo'lib qo'shilib ketardi). Qo'shimchani ajratamiz va guruhni majburan yopamiz.
_LIK_SUFFIXES = ("talik", "taligi", "ligi", "lig'i", "lig", "lik")

EXPECTED = {
    "passport_num": 7,    # passport/ID raqami (seriya harflarisiz)
    "phone": 9,           # operator kodi (2) + 7  — 998 prefiksisiz
    "jshshir": 14,        # PINFL
    "card": 16,           # Uzcard/Humo
    "card_last4": 4,      # oxirgi 4 raqam
}

def _strip_lik(tok: str):
    """Raqam so'zidagi 'lik' qo'shimchasini ajratadi.
    Qaytaradi: (o'zak, True) agar qo'shimcha bo'lsa, aks holda (tok, False).
    'ellik' (50) kabi to'g'ridan-to'g'ri raqamlarga tegmaydi."""
    if tok in UNITS or tok in TENS or tok in HUND:
        return tok, False          # 'ellik', 'birlik'? -> avval to'g'ridan tekshir
    for suf in _LIK_SUFFIXES:
        stem = tok[:-len(suf)]
        if tok.endswith(suf) and (stem in UNITS or stem in TENS or stem in HUND):
            return stem, True
    return tok, False


def uz_words_to_digits(text: str) -> str:
    """O'zbekcha aytilgan raqamni raqam SATRiga aylantiradi (leading zero saqlanadi).
    Qaytadagi satr uzunligi = aytilgan raqamlar soni. Kirish — STT so'z chiqishi
    ("to'qson birlik olti yuz to'qson besh ..."), literal raqam kutilmaydi (lekin
    kelib qolsa ham qo'llanadi)."""
    tokens = _normalize(text).split()
    out = []              # yig'ilgan raqam bo'laklari
    cur = 0               # joriy guruh qiymati
    seen_tens = False
    seen_units = False
    active = False

    def flush():
        nonlocal cur, seen_tens, seen_units, active
        if active:
            out.append(str(cur))
        cur, seen_tens, seen_units, active = 0, False, False, False

    for raw in tokens:
        if raw.isdigit():                 # STT'da kutilmaydi; xavfsizlik uchun qoldirilgan
            flush()
            out.append(raw)
            continue
        tok, force_flush = _strip_lik(raw)
        if tok in ZERO_WORDS:             # 0 doim yangi guruh boshlaydi (leading zero)
            flush()
            out.append("0")
        elif tok in UNITS:
            if seen_units:                # ketma-ket birlik -> yangi guruh (raqamma-raqam)
                flush()
            cur += UNITS[tok]
            seen_units, active = True, True
        elif tok in TENS:
            if seen_tens or seen_units:   # o'nlik allaqachon bor edi -> yangi guruh
                flush()
            cur += TENS[tok]
            seen_tens, active = True, True
        elif tok in HUND:                 # "olti yuz" = 6*100
            if not active:
                cur = 1
            cur *= 100
            seen_tens = seen_units = False
            active = True
        else:                             # harf / to'ldiruvchi so'z -> guruh chegarasi
            flush()
            continue
        if force_flush:                   # 'lik' -> operator kodi guruhi yopildi
            flush()
    flush()
    return "".join(out)