# -*- coding: utf-8 -*-
"""
O'zbek tili uchun EOU (end-of-utterance) dataset generatori.
LiveKit turn-detector (Qwen2.5-0.5B) LoRA fine-tune uchun.

Chiqish formati (JSONL, har qator bitta misol):
  {"messages": [{"role": "assistant", "content": "..."},
                {"role": "user", "content": "..."}],
   "label": 1,            # 1 = gap tugagan (EOU), 0 = davom etadi
   "pattern": "...",      # qaysi lingvistik naqsh (tahlil uchun)
   "domain": "..."}       # stsenariy (tahlil uchun)

Training paytida faqat "messages" + "label" kerak; pattern/domain — eval breakdown uchun.
Matnlar notebook'dagi normalize_text bilan bir xil normalizatsiyadan o'tgan
(kichik harf, punktuatsiya olib tashlangan, ' va - saqlanadi).
"""

import argparse
import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalizatsiya — try.ipynb dagi normalize_text bilan aynan bir xil,
# qo'shimcha: apostrof variantlarini (ʻ ʼ ’ ` ) ASCII ' ga keltiradi.
# ---------------------------------------------------------------------------
APOSTROPHES = "ʻʼ‘’`´"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
    )
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Slot lug'atlari
# ---------------------------------------------------------------------------
SLOTS = {
    "PLACE": ["toshkent", "samarqand", "buxoro", "andijon", "namangan", "xiva",
              "chilonzor", "yunusobod", "sergeli", "olmazor", "yakkasaroy",
              "mirobod", "beshog'och", "minor", "aeroport", "vokzal", "bozor"],
    "DAY": ["bugun", "ertaga", "indinga", "dushanba kuni", "seshanba kuni",
            "chorshanba kuni", "payshanba kuni", "juma kuni", "shanba kuni",
            "yakshanba kuni"],
    "DAY_GA": ["bugunga", "ertagaga", "dushanbaga", "seshanbaga", "chorshanbaga",
               "payshanbaga", "jumaga", "shanbaga", "yakshanbaga"],
    "TIME": ["soat sakkizda", "soat to'qqizda", "soat o'nda", "soat o'n birda",
             "soat o'n ikkida", "soat bir yarimda", "soat ikkida", "soat uchda",
             "soat to'rtda", "soat beshda", "soat oltida", "soat yettida"],
    "TIME_GA": ["soat yettiga", "soat sakkizga", "soat to'qqizga", "soat o'nga",
                "soat oltiga"],
    "FOOD": ["osh", "lag'mon", "somsa", "shashlik", "manti", "chuchvara",
             "norin", "mastava", "sho'rva"],
    "NUM": ["bitta", "ikkita", "uchta", "to'rtta", "beshta", "oltita"],
    "DOCTOR": ["tish shifokori", "terapevt", "kardiolog", "nevropatolog",
               "lor", "okulist", "travmatolog"],
    "CARD": ["humo kartam", "uzcard kartam", "visa kartam"],
    "TARIFF": ["oylik tarif", "yangi tarif", "arzonroq tarif"],
    "CONJ": ["va", "lekin", "ammo", "chunki", "yoki", "keyin",
             "shuning uchun", "bundan tashqari", "yana", "ya'ni"],
    "DISTRICT": ["chilonzor", "yunusobod", "sergeli", "olmazor", "yakkasaroy",
                 "mirobod", "uchtepa", "bektemir", "shayxontohur"],
    "STREET": ["bobur", "navoiy", "amir temur", "mustaqillik", "paxtakor",
               "bunyodkor", "qatortol"],
    "ORD": ["birinchi", "ikkinchi", "uchinchi", "to'rtinchi", "beshinchi",
            "oltinchi", "yettinchi", "sakkizinchi", "to'qqizinchi", "o'ninchi",
            "o'n birinchi", "o'n ikkinchi", "o'n to'rtinchi", "o'n beshinchi",
            "yigirmanchi", "yigirma birinchi"],
    "SUM": ["o'n ming so'm", "yigirma ming so'm", "ellik ming so'm",
            "yuz ming so'm", "ikki yuz ming so'm", "besh yuz ming so'm",
            "bir million so'm"],
    "DAY_BERI": ["kechadan beri", "bugun ertalabdan beri", "o'tgan haftadan beri",
                 "ikki kundan beri", "uch kundan beri"],
}
# Ikkinchi nusxa slotlar (PLACE2 va h.k.) shu ro'yxatlardan olinadi,
# birinchisidan farqli bo'lishi shart.
ALIAS = {"PLACE2": "PLACE", "NUM2": "NUM", "FOOD2": "FOOD", "ORD2": "ORD"}

DIGITS = ["nol", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti",
          "sakkiz", "to'qqiz"]
TENS = [None, "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish",
        "yetmish", "sakson", "to'qson"]
ORDINALS = {"bir": "birinchi", "ikki": "ikkinchi", "uch": "uchinchi",
            "to'rt": "to'rtinchi", "besh": "beshinchi", "olti": "oltinchi",
            "yetti": "yettinchi", "sakkiz": "sakkizinchi", "to'qqiz": "to'qqizinchi",
            "o'n": "o'ninchi", "yigirma": "yigirmanchi", "o'ttiz": "o'ttizinchi",
            "qirq": "qirqinchi", "ellik": "ellikinchi", "oltmish": "oltmishinchi",
            "yetmish": "yetmishinchi", "sakson": "saksoninchi",
            "to'qson": "to'qsoninchi", "yuz": "yuzinchi", "ming": "minginchi"}
MONTHS = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul",
          "avgust", "sentabr", "oktabr", "noyabr", "dekabr"]
# O'zbekiston operator kodlari (Beeline, Ucell, Mobiuz, Uzmobile, Humans, OQ)
OPCODES = ["90", "91", "93", "94", "95", "97", "98", "99",
           "33", "50", "55", "77", "88", "20"]
# Passport / ID karta seriyalari: 2 harf + 7 raqam (masalan AB 1234567)
PASSPORT_SERIES = ["aa", "ab", "ac", "ad", "ae"]
LETTER_NAMES = {"a": "a", "b": "be", "c": "se", "d": "de", "e": "e"}
# Karta BIN: Uzcard 8600, Humo 9860
CARD_BINS = ["8600", "9860"]


def num2words(n: int) -> str:
    """0-999 sonini o'zbekcha so'zga aylantiradi."""
    if n < 10:
        return DIGITS[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + DIGITS[n % 10])
    return DIGITS[n // 100] + " yuz" + ("" if n % 100 == 0 else " " + num2words(n % 100))


def ordinalize(phrase: str) -> str:
    """Oxirgi so'zni tartib songa aylantiradi: 'sakson to'rt' -> 'sakson to'rtinchi'."""
    words = phrase.split()
    words[-1] = ORDINALS[words[-1]]
    return " ".join(words)


def group_words(s: str) -> str:
    """Raqamlar guruhini o'qish: '695'->'olti yuz to'qson besh', '05'->'nol besh',
    4+ xonali guruh juftlab o'qiladi: '3412'->'o'ttiz to'rt o'n ikki'."""
    if s.startswith("0"):
        return " ".join(DIGITS[int(c)] for c in s)
    if len(s) > 3:
        return " ".join(group_words(s[i:i + 2]) for i in range(0, len(s), 2))
    return num2words(int(s))


def digit_words(s: str) -> str:
    """Raqamma-raqam o'qish: '123' -> 'bir ikki uch'."""
    return " ".join(DIGITS[int(c)] for c in s)


# Raqam satrini bo'laklarga bo'lib o'qish sxemalari (STT shunday chiqaradi:
# "710 22 73" -> "yetti yuz o'n yigirma ikki yetmish uch").
GROUPINGS = {
    7: [[3, 2, 2], [2, 2, 3], [2, 2, 2, 1], [3, 2, 1, 1], [1, 3, 3], [3, 4]],
    4: [[4], [2, 2], [1, 3], [3, 1]],
    2: [[2], [1, 1]],
    3: [[3], [2, 1], [1, 2]],
    # telefon (9), jshshir (14), karta (16) — tabiiy juft/uch guruhlashlar
    9: [[3, 3, 3], [2, 3, 2, 2], [3, 2, 2, 2], [2, 2, 2, 2, 1], [3, 3, 2, 1]],
    14: [[2] * 7, [3, 3, 3, 3, 2], [2, 3, 3, 3, 3], [4, 4, 4, 2], [3, 2, 2, 2, 2, 3]],
    16: [[4, 4, 4, 4], [2] * 8, [3, 3, 3, 3, 3, 1], [4, 3, 3, 3, 3], [2, 2, 2, 2, 2, 2, 2, 2]],
    6: [[2, 2, 2], [3, 3], [2, 4]],
    5: [[3, 2], [2, 3], [2, 2, 1]],
}


def render_number(digits: str, rng: random.Random) -> str:
    """Raqam satrini tasodifiy tabiiy uslubda so'z bilan o'qiydi (noaniqlikdan xoli,
    so'z bilan aytilgan raqam soni = satr uzunligi). Uslublar:
      - raqamma-raqam: "bir ikki uch"
      - tabiiy bo'lak-so'z: read_chunks (juft/uch guruhlar)."""
    if not digits:
        return ""
    if rng.random() < 0.4:
        return digit_words(digits)
    return " ".join(read_chunks(digits, rng))


def boundary_cut_lengths(n: int, rng: random.Random, k: int):
    """Chala uzunliklar — CHEGARAGA fokus (N-1, N-2 ko'p, uzoq kam).
    Model N chegarasini aniq o'rganishi uchun eng muhim signal shu."""
    if n <= 1:
        return []
    pool = [n - 1, n - 1, n - 1, n - 2, n - 2] + list(range(1, n))
    picks = set()
    tries = 0
    while len(picks) < min(k, n - 1) and tries < 50:
        picks.add(rng.choice(pool))
        tries += 1
    return sorted(picks)


def read_chunks(digits: str, rng: random.Random):
    """Raqam satrini tabiiy bo'laklarga bo'lib, har bo'lakni so'z bilan o'qiydi.
    Qaytaradi: bo'lak-o'qishlar ro'yxati. full = hammasi, cut = boshidan bir qismi.
    Masalan '7102273' -> ['yetti yuz o'n', 'yigirma ikki', 'yetmish uch'].

    NOANIQLIKDAN XOLI: oxirgi bo'lmagan bo'lak nol bilan tugasa (masalan '710',
    '20'), uni guruh-so'z o'qish keyingi bo'lakni "yutib" yuboradi
    ('uch yuz' + 'sakson olti' -> 'uch yuz sakson olti' = 386). Bunday bo'laklar
    raqamma-raqam o'qiladi (oxiri 'nol' -> chegara aniq)."""
    n = len(digits)
    scheme = rng.choice(GROUPINGS.get(n, [[min(3, n)] * (n // 3) + ([n % 3] if n % 3 else [])]))
    chunks, i = [], 0
    for j, g in enumerate(scheme):
        grp = digits[i:i + g]
        i += g
        is_last = (j == len(scheme) - 1)
        if not is_last and grp.endswith("0"):
            chunks.append(digit_words(grp))     # ...nol bilan tugaydi -> aniq chegara
        else:
            chunks.append(group_words(grp))
    return chunks


def year_words(y: int) -> str:
    """1950-2010 yilni rasmiy o'qilishga aylantiradi."""
    if y < 2000:
        return "bir ming to'qqiz yuz " + ordinalize(num2words(y % 100)) + " yil"
    if y == 2000:
        return "ikki minginchi yil"
    return "ikki ming " + ordinalize(num2words(y % 100)) + " yil"

# ---------------------------------------------------------------------------
# Domen kontekstlari: assistant ochilish gaplari (ba'zan 2 bosqichli almashinuv)
# ---------------------------------------------------------------------------
A, U = "assistant", "user"

CONTEXTS = {
    "taxi": [
        [(A, "taksi xizmati assalomu alaykum qayerga borasiz")],
        [(A, "assalomu alaykum qayerdan olib ketamiz")],
        [(A, "salom taksi buyurtma qilasizmi")],
        [(A, "taksi xizmati assalomu alaykum"), (U, "taksi kerak edi"),
         (A, "xo'p qayerga borasiz")],
    ],
    "food": [
        [(A, "assalomu alaykum buyurtmangizni eshitaman")],
        [(A, "yetkazib berish xizmati salom nima buyurtma qilasiz")],
        [(A, "assalomu alaykum"), (U, "buyurtma bermoqchi edim"),
         (A, "marhamat eshitaman")],
    ],
    "bank": [
        [(A, "bank axborot xizmati assalomu alaykum sizga qanday yordam bera olaman")],
        [(A, "assalomu alaykum bank axborot xizmati eshitaman")],
        [(A, "bank axborot xizmati salom"), (U, "kartam bo'yicha savolim bor edi"),
         (A, "xo'p eshitaman")],
    ],
    "clinic": [
        [(A, "klinika qabulxonasi assalomu alaykum")],
        [(A, "salom shifoxona axborot xizmati qanday yordam beray")],
        [(A, "klinika qabulxonasi salom"), (U, "shifokorga yozilmoqchi edim"),
         (A, "qaysi shifokorga yozilasiz")],
    ],
    "telecom": [
        [(A, "aloqa operatori assalomu alaykum sizga qanday yordam bera olaman")],
        [(A, "salom mobil aloqa xizmati eshitaman")],
    ],
    "general": [
        [(A, "assalomu alaykum sizga qanday yordam bera olaman")],
        [(A, "salom bugun sizga qanday yordam beray")],
    ],
}

# ---------------------------------------------------------------------------
# TUGAGAN gaplar (label = 1)
# ---------------------------------------------------------------------------
COMPLETE = [
    # taxi
    ("taxi", "statement", "menga taksi kerak {PLACE}ga boraman"),
    ("taxi", "statement", "{PLACE}dan {PLACE2}ga borishim kerak"),
    ("taxi", "statement", "men hozir {PLACE}da turibman"),
    ("taxi", "request",   "{DAY} {TIME} mashina yuboring"),
    ("taxi", "question",  "yo'l haqi qancha bo'ladi"),
    ("taxi", "question",  "haydovchi qachon yetib keladi"),
    ("taxi", "statement", "mashinani bekor qilmoqchiman"),
    ("taxi", "statement", "yukim bor kattaroq mashina kerak"),
    ("taxi", "question",  "to'lovni karta orqali qilsam bo'ladimi"),
    # food
    ("food", "request",   "menga {NUM} {FOOD} yozib qo'ying"),
    ("food", "statement", "{NUM} {FOOD} va {NUM2} {FOOD2} buyurtma qilmoqchiman"),
    ("food", "question",  "yetkazib berish qancha turadi"),
    ("food", "question",  "buyurtmam qancha vaqtda tayyor bo'ladi"),
    ("food", "question",  "bugun {FOOD} bormi"),
    ("food", "statement", "buyurtmani bekor qilmoqchiman"),
    ("food", "question",  "naqd to'lasam bo'ladimi"),
    ("food", "question",  "qaysi taom tezroq tayyor bo'ladi"),
    # bank
    ("bank", "request",   "kartam yo'qolib qoldi bloklab qo'ying"),
    ("bank", "statement", "{CARD}ni bloklamoqchiman"),
    ("bank", "question",  "hisobimda qancha qoldiq borligini aytib bera olasizmi"),
    ("bank", "question",  "yangi karta ochmoqchiman nima qilishim kerak"),
    ("bank", "question",  "pul o'tkazmasi qancha vaqtda yetib boradi"),
    ("bank", "question",  "komissiya necha foiz bo'ladi"),
    ("bank", "statement", "kredit shartlari bilan qiziqayotgan edim"),
    ("bank", "question",  "kartamga pul tushmadi tekshirib bera olasizmi"),
    ("bank", "question",  "eng yaqin filialingiz qayerda"),
    ("bank", "statement", "kartamdan {SUM} yechib olinibdi bu qanday to'lov"),
    ("bank", "question",  "{SUM} o'tkazsam komissiyasi qancha bo'ladi"),
    ("bank", "statement", "{DAY} kartamga {SUM} tushishi kerak edi lekin tushmadi"),
    ("bank", "question",  "pin kodimni unutib qo'ydim nima qilishim kerak"),
    ("bank", "request",   "sms xabarnomani yoqib qo'ying"),
    ("bank", "request",   "kartamni aktivlashtirib bering"),
    ("bank", "question",  "kartam bankomatda qolib ketdi nima qilay"),
    ("bank", "request",   "kartamdan noma'lum to'lov yechilibdi tekshirib bering"),
    ("bank", "question",  "bugun dollar kursi qancha"),
    ("bank", "question",  "omonat ochmoqchiman foiz stavkasi qancha"),
    ("bank", "question",  "muddatli omonat foizi qancha"),
    ("bank", "request",   "ilovaga kira olmayapman parolni tiklashga yordam bering"),
    ("bank", "question",  "kreditimning navbatdagi to'lovi qachon"),
    ("bank", "question",  "{SUM} kredit olsam oyiga qancha to'layman"),
    ("bank", "question",  "kartamning muddati tugabdi yangisini qanday olsam bo'ladi"),
    ("bank", "question",  "chet eldan pul o'tkazmasi keldi qanday olsam bo'ladi"),
    ("bank", "statement", "kartamni yo'qotib qo'ydim yangisini rasmiylashtirmoqchiman"),
    # clinic
    ("clinic", "statement", "{DOCTOR}ga yozilmoqchiman"),
    ("clinic", "request",   "{DAY} {TIME} qabulga yozib qo'ying"),
    ("clinic", "question",  "qabul qancha turadi"),
    ("clinic", "question",  "{DAY} bo'sh vaqt bormi"),
    ("clinic", "question",  "analiz natijalarim chiqdimi"),
    ("clinic", "question",  "boshim qattiq og'riyapti qaysi shifokorga uchrashay"),
    ("clinic", "statement", "qabulni boshqa kunga ko'chirmoqchiman"),
    # telecom
    ("telecom", "question",  "internetim ishlamayapti tekshirib bera olasizmi"),
    ("telecom", "statement", "tarifimni o'zgartirmoqchiman"),
    ("telecom", "question",  "balansimda qancha pul bor"),
    ("telecom", "statement", "qo'shimcha internet paketi ulamoqchiman"),
    ("telecom", "question",  "raqamimni saqlab qolgan holda operatorni almashtirsam bo'ladimi"),
    ("telecom", "question",  "chet elda rouming qancha turadi"),
    ("telecom", "question",  "xizmat haqi qancha"),
    ("telecom", "statement", "{DAY_BERI} internet ishlamayapti"),
    ("telecom", "question",  "har oyda {SUM} to'layapman bu qaysi tarif"),
    # general
    ("general", "question",  "{DAY} ob havo qanday bo'ladi"),
    ("general", "question",  "soat necha bo'ldi"),
    ("general", "request",   "{TIME_GA} budilnik qo'yib qo'y"),
    ("general", "statement", "rahmat katta rahmat"),
    ("general", "statement", "boshqa savolim yo'q rahmat"),
    ("general", "statement", "xayr salomat bo'ling"),
    ("general", "statement", "tushunarli rahmat sizga"),
    ("general", "statement", "assalomu alaykum"),
    ("general", "statement", "salom yaxshimisiz"),
    ("general", "question",  "alo eshityapsizmi"),
    ("general", "statement", "bir daqiqa o'ylab ko'ray"),
    ("general", "statement", "menga bir daqiqa vaqt bering"),
]

# ---------------------------------------------------------------------------
# TUGALLANMAGAN gaplar (label = 0) — o'zbek morfologiyasi naqshlari
# ---------------------------------------------------------------------------
INCOMPLETE = [
    # fe'l tushib qolgan (SOV — fe'l oxirda bo'lishi kerak edi)
    ("taxi",    "verb_cut", "men {DAY} {TIME} {PLACE}ga"),
    ("taxi",    "verb_cut", "men aeroportga soat"),
    ("general", "verb_cut", "men bugun maktabga"),
    ("general", "verb_cut", "men {DAY} {PLACE}ga"),
    ("clinic",  "verb_cut", "men {DAY} ertalab shifokor"),
    ("bank",    "verb_cut", "men kartamdan {SUM}"),
    ("bank",    "verb_cut", "{DAY} kartamga {SUM}"),
    ("telecom", "verb_cut", "{DAY_BERI} internetim"),
    ("telecom", "verb_cut", "men har oyda {SUM}"),
    # ravishdosh (-ib/-b) — gap davomi kutiladi
    ("taxi",    "converb", "meni {PLACE}dan olib"),
    ("bank",    "converb", "kartam yo'qolib"),
    ("telecom", "converb", "balansimni tekshirib"),
    ("clinic",  "converb", "qornim og'rib"),
    ("clinic",  "converb", "bolamning isitmasi ko'tarilib"),
    ("food",    "converb", "ovqatni buyurtma qilib"),
    ("general", "converb", "ishdan chiqib"),
    ("general", "converb", "uyga borib"),
    # shart mayli (-sa) — bosh gap yo'q
    ("taxi",    "conditional", "agar mashina tez kelsa"),
    ("food",    "conditional", "agar tez yetkazib bersangiz"),
    ("bank",    "conditional", "agar komissiyasi baland bo'lsa"),
    ("clinic",  "conditional", "agar {DAY} bo'sh vaqt bo'lsa"),
    ("telecom", "conditional", "agar shu muammo hal bo'lmasa"),
    # osilib qolgan kelishik (tushum, qaratqich, jo'nalish)
    ("taxi",    "hanging_case", "haydovchining raqamini"),
    ("bank",    "hanging_case", "hisobimdagi qoldiqni"),
    ("clinic",  "hanging_case", "analiz natijalarimni"),
    ("food",    "hanging_case", "yetkazib berish narxini"),
    ("telecom", "hanging_case", "tarifimni boshqasiga"),
    ("bank",    "hanging_case", "men sizning bankingizdan"),
    ("general", "hanging_case", "xo'sh men sizdan"),
    ("general", "hanging_case", "endi men sizga"),
    # bank qo'shimcha naqshlari
    ("bank",    "converb",       "pin kodimni unutib"),
    ("bank",    "converb",       "kartam bankomatda qolib"),
    ("bank",    "converb",       "kartamni yo'qotib"),
    ("bank",    "hanging_case",  "omonat foizlari haqida"),
    ("bank",    "hanging_case",  "kreditimning navbatdagi to'lovini"),
    ("bank",    "hanging_case",  "dollar kursini"),
    ("bank",    "participle",    "kartamdan yechilgan"),
    ("bank",    "filler",        "demak mening kartamda"),
    ("bank",    "trailing_conj_hand", "ilovaga kirmoqchi edim lekin"),
    ("bank",    "verb_cut",      "men har oyda kartamga {SUM}"),
    # sabab ergash gap yolg'iz (-gani uchun, -guncha, -gach)
    ("telecom", "sub_clause", "telefonim ishlamay qolgani uchun"),
    ("bank",    "sub_clause", "kartamga pul tushmagani uchun"),
    ("food",    "sub_clause", "ovqat tayyor bo'lgach"),
    ("taxi",    "sub_clause", "mashina kelguncha men"),
    ("clinic",  "sub_clause", "shifokor qabulidan keyin men"),
    # sifatdosh (-gan) — ot kutilyapti
    ("food",    "participle", "menga achchiq bo'lmagan"),
    ("general", "participle", "men kecha aytgan"),
    ("general", "participle", "siz yuborgan"),
    # daraja so'zi osilib qolgan
    ("telecom", "degree", "internetim juda"),
    ("food",    "degree", "bu taom juda"),
    # to'xtab qolgan sanash
    ("food",    "enum_cut", "menga {NUM} {FOOD} {NUM2}"),
    ("bank",    "enum_cut", "birinchidan kartam ishlamayapti ikkinchidan"),
    ("general", "enum_cut", "birinchidan"),
    ("general", "enum_cut", "ikkinchidan esa"),
    # kirish so'z / to'ldiruvchi bilan tugagan
    ("general", "filler", "menga bitta narsa kerak edi anu"),
    ("general", "filler", "gap shundaki"),
    ("general", "filler", "aytmoqchi edimki"),
    ("general", "filler", "muammo shundaki"),
    ("general", "filler", "aytmoqchi bo'lganim"),
    ("general", "filler", "mening fikrimcha bu"),
    ("general", "filler", "qisqasi men"),
    ("general", "filler", "masalan men {DAY}"),
    ("general", "filler", "bir daqiqa men hozir"),
    ("general", "filler", "yo'q men aytmoqchi bo'lganim"),
    # bog'lovchi bilan boshlangan davom
    ("bank",    "trailing_conj_hand", "pul o'tkazmoqchi edim lekin"),
    ("food",    "trailing_conj_hand", "bitta osh olmoqchi edim lekin"),
    ("clinic",  "trailing_conj_hand", "boshim og'riyapti shuning uchun"),
    ("general", "trailing_conj_hand", "yo'q bilmadim lekin"),
    ("general", "trailing_conj_hand", "reysimni bekor qilishdi va"),
    ("taxi",    "trailing_conj_hand", "bitta yuk mashinasi kerak edi chunki"),
    ("telecom", "trailing_conj_hand", "kecha kechqurundan beri internet"),
]

# Bog'lovchi qo'shib tugallanmagan variant yasashga yaroqli TUGAGAN gaplar
# (darak gaplar; savol + bog'lovchi g'alati bo'lgani uchun savollarni olmaymiz)
CONJ_BASE_PATTERNS = {"statement", "request"}

# ---------------------------------------------------------------------------
# UZUN CHALKASH GAPLAR — eng muhim tuzatish.
# Model "uzun + ko'p bo'lakli jonli nutq = tugallanmagan" degan soxta qoidani
# o'rgangan, chunki oldingi datasetda uzun label=1 misollar faqat "toza"
# (diktovka raqamlari / silliq gaplar) edi. Bu yerda HAR BIR uzun chalkash gap
# ikki versiyada beriladi:
#   full — to'g'ri predikat / -mi so'roq yuklamasi / fe'l bilan TUGAGAN (label=1)
#   cut  — aynan o'sha gap o'rtada, osilib qolgan ot/bog'lovchi bilan (label=0)
# Shunday qilib uzunlik label bilan bog'liqligini yo'qotadi — model gapning
# OXIRIGA (predikat bormi?) qarashni o'rganadi, uzunligiga emas.
# (full, cut) — matnlar ataylab chalkash: "anu", "xullas", "bilasizmi",
# takror so'zlar, o'zini tuzatish — real STT transkriptidagidek.
# ---------------------------------------------------------------------------
LONG_RAMBLE = [
    ("bank",
     "u muzlatib qo'yiladimi karta puli ichida puli bilan yo'qolgan yoki boshqa humo kartamga o'tkazsa bo'ladimi",
     "u muzlatib qo'yiladimi karta puli ichida puli bilan yo'qolgan yoki boshqa humo kartamga"),
    ("bank",
     "menga tushuntiring kartamdan pul yechilibdi lekin men hech qanaqa xarid qilmaganman bu qanday bo'lishi mumkin",
     "menga tushuntiring kartamdan pul yechilibdi lekin men hech qanaqa xarid qilmaganman bu"),
    ("bank",
     "xullas men kecha bir odamga pul o'tkazgandim lekin unga hali yetib bormabdi shuni tekshirib bera olasizmi",
     "xullas men kecha bir odamga pul o'tkazgandim lekin unga hali yetib bormabdi shuni"),
    ("bank",
     "bilasizmi kartam eskirib qolgan edi yangisini olmoqchiman lekin qaysi filialga borishim kerakligini ayta olasizmi",
     "bilasizmi kartam eskirib qolgan edi yangisini olmoqchiman lekin qaysi filialga borishim kerakligini"),
    ("bank",
     "anu men pensiya olaman shu kartaga tushadi lekin bu oy nega tushmaganini bilmoqchi edim tekshirib bering",
     "anu men pensiya olaman shu kartaga tushadi lekin bu oy nega tushmaganini"),
    ("bank",
     "kartam bankomatda qolib ketdi kecha kechqurun edi hozir uni qanday qaytarib olsam bo'ladi",
     "kartam bankomatda qolib ketdi kecha kechqurun edi hozir uni qanday qaytarib olsam"),
    ("bank",
     "men sizdan so'ramoqchi edim agar kartani yo'qotib qo'ysam undagi pul saqlanib qoladimi yoki yo'qoladimi",
     "men sizdan so'ramoqchi edim agar kartani yo'qotib qo'ysam undagi pul saqlanib qoladimi yoki"),
    ("bank",
     "hisobimda qancha pul borligini aytolasizmi chunki men kredit to'lovini bugun qilishim kerak edi",
     "hisobimda qancha pul borligini aytolasizmi chunki men kredit to'lovini bugun qilishim kerak"),
    ("bank",
     "menga kelgan pul o'tkazmasini olishim uchun nima qilishim kerak passportim bilan borsam bo'ladimi",
     "menga kelgan pul o'tkazmasini olishim uchun nima qilishim kerak passportim bilan"),
    ("bank",
     "kartamga notanish raqamdan sms keldi parol so'rayapti bu firibgarlikmi yoki sizning xabaringizmi",
     "kartamga notanish raqamdan sms keldi parol so'rayapti bu firibgarlikmi yoki"),
    ("bank",
     "eshiting men shu ilovangizdan foydalanaman lekin bugun kira olmayapman parol xato deyapti nima qilsam ekan",
     "eshiting men shu ilovangizdan foydalanaman lekin bugun kira olmayapman parol xato deyapti"),
    ("bank",
     "menga bir savol bor edi agar dollarni so'mga aylantirsam bugungi kurs bo'yicha qancha bo'ladi",
     "menga bir savol bor edi agar dollarni so'mga aylantirsam bugungi kurs bo'yicha qancha"),
    ("bank",
     "kartamni bloklab qo'ygan edingiz endi topib oldim uni qayta ochib berolasizmi yoki yangisini olaymi",
     "kartamni bloklab qo'ygan edingiz endi topib oldim uni qayta ochib berolasizmi yoki yangisini"),
    ("bank",
     "men omonat ochmoqchiman lekin qaysi biri foydaliroq ekanini bilmayman muddatlisimi yoki oddiysimi",
     "men omonat ochmoqchiman lekin qaysi biri foydaliroq ekanini bilmayman muddatlisimi yoki"),
    ("bank",
     "otamning kartasi bor edi u kishi kasal bo'lib qoldi men uning o'rniga pul yechib olsam bo'ladimi",
     "otamning kartasi bor edi u kishi kasal bo'lib qoldi men uning o'rniga pul yechib olsam"),
    ("bank",
     "kartamdan har oy nimadir yechilyapti buni obuna deb o'ylayapman lekin qanday bekor qilishni bilmayman yordam bering",
     "kartamdan har oy nimadir yechilyapti buni obuna deb o'ylayapman lekin qanday bekor qilishni"),
    ("bank",
     "menga chet eldan pul kelishi kerak edi bir haftadan beri kutyapman hali ham kelmadi qayerdan tekshirsam bo'ladi",
     "menga chet eldan pul kelishi kerak edi bir haftadan beri kutyapman hali ham kelmadi qayerdan"),
    ("bank",
     "kredit olganman har oy to'lab turaman lekin bu oy summa ko'proq chiqdi nega bunday bo'lganini bilsam bo'ladimi",
     "kredit olganman har oy to'lab turaman lekin bu oy summa ko'proq chiqdi nega bunday bo'lganini"),
    ("bank",
     "kartam do'konda ishlamay qoldi to'lay olmadim juda uyaldim sababi nima ekanini bilib bera olasizmi",
     "kartam do'konda ishlamay qoldi to'lay olmadim juda uyaldim sababi nima ekanini"),
    ("bank",
     "manzilim o'zgardi shuning uchun yangi kartani boshqa filialga yuborishingizni so'ramoqchi edim iltimos shuni hisobga oling",
     "manzilim o'zgardi shuning uchun yangi kartani boshqa filialga yuborishingizni so'ramoqchi edim iltimos shuni"),
    ("telecom",
     "internetim uch kundan beri ishlamayapti routerni ham o'chirib yoqdim lekin foyda bermadi nima qilay",
     "internetim uch kundan beri ishlamayapti routerni ham o'chirib yoqdim lekin foyda bermadi"),
    ("telecom",
     "men shu tarifni ulagandim lekin internet juda tez tugab qolyapti balki boshqasiga o'tsam yaxshiroqmi",
     "men shu tarifni ulagandim lekin internet juda tez tugab qolyapti balki boshqasiga"),
    ("telecom",
     "raqamimni boshqa operatorga ko'chirmoqchiman eshitishimcha bir necha kun kerak ekan aniq qancha vaqt ketadi",
     "raqamimni boshqa operatorga ko'chirmoqchiman eshitishimcha bir necha kun kerak ekan aniq"),
    ("clinic",
     "bolamning isitmasi kecha kechqurundan beri tushmayapti ertaga shifokorga olib borsam qabul qiladimi",
     "bolamning isitmasi kecha kechqurundan beri tushmayapti ertaga shifokorga olib borsam"),
    ("clinic",
     "men avval ham shu yerga yozilgandim lekin kelolmadim endi qaytadan navbat olishim kerakmi",
     "men avval ham shu yerga yozilgandim lekin kelolmadim endi qaytadan navbat olishim"),
    ("clinic",
     "analiz topshirgandim natijasini kutyapman odatda qancha vaqtda tayyor bo'ladi",
     "analiz topshirgandim natijasini kutyapman odatda qancha vaqtda tayyor"),
    ("taxi",
     "taksi chaqirgandim lekin haydovchi boshqa manzilga keldi men esa hali kutib turibman qayta yuborasizmi",
     "taksi chaqirgandim lekin haydovchi boshqa manzilga keldi men esa hali kutib turibman"),
    ("food",
     "men buyurtma bergandim juda uzoq bo'lyapti bir soatdan oshdi qachon yetkazib berishadi aniq ayta olasizmi",
     "men buyurtma bergandim juda uzoq bo'lyapti bir soatdan oshdi qachon yetkazib berishadi aniq"),
    ("general",
     "aslida men sizga boshqa narsa uchun qo'ng'iroq qilgandim lekin hozir unutib qo'ydim esimga tushsa yana bog'lanaman",
     "aslida men sizga boshqa narsa uchun qo'ng'iroq qilgandim lekin hozir unutib qo'ydim esimga tushsa yana"),
    ("general",
     "xullas gap shundaki menga yordam kerak edi lekin qanday tushuntirishni bilmayapman shoshib ketdim uzr",
     "xullas gap shundaki menga yordam kerak edi lekin qanday tushuntirishni bilmayapman shoshib"),
]

# ---------------------------------------------------------------------------
# Savol-javob juftliklari: qisqa elliptik javoblar kontekstda TUGAGAN (label=1)
# ---------------------------------------------------------------------------
QA_PAIRS = [
    ("taxi",    "qayerga borasiz",                        "{PLACE}ga"),
    ("taxi",    "qayerdan olib ketamiz",                  "{PLACE}dan"),
    ("taxi",    "qachon mashina kerak",                   "{DAY} {TIME}"),
    ("taxi",    "to'lov naqdmi yoki karta orqalimi",      "karta orqali"),
    ("taxi",    "to'lov naqdmi yoki karta orqalimi",      "naqd pulda"),
    ("food",    "nechta bo'lsin",                         "{NUM}"),
    ("food",    "qaysi taomni xohlaysiz",                 "{FOOD}"),
    ("food",    "qachon yetkazib beraylik",               "{DAY} {TIME}"),
    ("bank",    "qaysi kartangizni bloklaymiz",           "{CARD}ni"),
    ("bank",    "qaysi valyutada o'tkazmoqchisiz",        "so'mda"),
    ("bank",    "qaysi valyutada o'tkazmoqchisiz",        "dollarda"),
    ("bank",    "qaysi valyutada o'tkazmoqchisiz",        "yevroda"),
    ("bank",    "yangi kartani qaysi filialdan olmoqchisiz", "{DISTRICT} filialidan"),
    ("bank",    "omonatni qaysi muddatga ochmoqchisiz",   "bir yilga"),
    ("bank",    "omonatni qaysi muddatga ochmoqchisiz",   "olti oyga"),
    ("clinic",  "qaysi shifokorga yozilasiz",             "{DOCTOR}ga"),
    ("clinic",  "qaysi kunga yozay",                      "{DAY_GA}"),
    ("clinic",  "qabul qaysi vaqtga qulay",               "{TIME_GA}"),
    ("telecom", "qaysi tarifga o'tmoqchisiz",             "{TARIFF}ga"),
]

# Tasdiq/rad javoblari — kontekstda TUGAGAN (label=1)
ACK_PAIRS = [
    ("buyurtmangizni tasdiqlaysizmi", ["ha", "ha tasdiqlayman", "yo'q bekor qiling",
                                       "ha hammasi to'g'ri"]),
    ("yana biror narsa kerakmi",      ["yo'q rahmat", "yo'q shu xolos",
                                       "kerak emas rahmat", "ha yana bitta savolim bor"]),
    ("sizga sms yuborsam bo'ladimi",  ["ha bo'ladi", "mayli", "xo'p", "yo'q kerak emas"]),
    ("to'g'ri tushundimmi",           ["ha to'g'ri", "ha aynan shunday", "yo'q unday emas"]),
    ("shu vaqt sizga to'g'ri keladimi", ["ha to'g'ri keladi", "yo'q boshqa vaqt bo'lsin",
                                         "bo'ladi kelishdik"]),
]

# ---------------------------------------------------------------------------
# Minnatdorchilik / suhbatni yakunlash — TUGAGAN (label=1)
# ---------------------------------------------------------------------------
GRATITUDE = [
    "rahmat", "xo'p rahmat", "hammasi joyida rahmat", "yaxshi rahmat sizga",
    "bo'ldi rahmat", "katta rahmat", "rahmat sizga", "rahmat yaxshi qoling",
    "bo'ldi shu yetarli rahmat", "tushunarli rahmat", "ha bo'ldi rahmat",
    "xo'p rahmat yaxshi qoling", "rahmat xayr", "minnatdorman",
    "katta rahmat sizga ham xayrli kun", "rahmat sizga ham omad",
]
# Kontrast: minnatdorchilikdan keyin gap davom etadi — TUGALLANMAGAN (label=0)
GRATITUDE_CUT = [
    "rahmat lekin", "rahmat faqat", "bo'ldi rahmat faqat bitta",
    "rahmat endi menga", "katta rahmat endi", "rahmat yana bitta",
]
# Yakunlash kontekstlari (bank call center uslubida)
CLOSE_CONTEXTS = [
    [(A, "so'rovingiz qabul qilindi yana savolingiz bormi")],
    [(A, "kartangiz muvaffaqiyatli bloklandi")],
    [(A, "ma'lumot sms tarzida yuborildi")],
    [(A, "arizangiz rasmiylashtirildi mutaxassis siz bilan bog'lanadi")],
    [(A, "yana biror narsa kerakmi")],
    [(A, "yordam bera olganimdan xursandman")],
]

# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------
SLOT_RE = re.compile(r"\{([A-Z_0-9]+)\}")


def expand_template(template: str, rng: random.Random, cap: int):
    """Shablondagi slotlarni to'ldirib, ko'pi bilan `cap` ta variant qaytaradi."""
    names = SLOT_RE.findall(template)
    if not names:
        return [template]
    pools = [SLOTS[ALIAS.get(n, n)] for n in names]
    combos = list(itertools.product(*pools))
    # PLACE2 != PLACE va h.k.
    def ok(combo):
        vals = dict()
        for n, v in zip(names, combo):
            base = ALIAS.get(n, n)
            if n in ALIAS and vals.get(base) == v:
                return False
            vals.setdefault(base, v)
        return True
    combos = [c for c in combos if ok(c)]
    rng.shuffle(combos)
    out = []
    for combo in combos[:cap]:
        text = template
        for n, v in zip(names, combo):
            text = text.replace("{" + n + "}", v, 1)
        out.append(text)
    return out


def pick_context(domain: str, rng: random.Random, force=None):
    """Kontekst tanlash: 20% kontekstsiz, qolgani domen kontekstidan."""
    if force is not None:
        return force
    if rng.random() < 0.20:
        return []
    return rng.choice(CONTEXTS[domain])


def context_variants(domain: str, rng: random.Random, k: int):
    """Bitta matn uchun k tagacha har xil kontekst (kontekstsiz varianti bilan)."""
    pool = list(CONTEXTS[domain]) + [[]]
    rng.shuffle(pool)
    return pool[:k]


def make_example(ctx, user_text, label, pattern, domain):
    # Ketma-ket bir xil rollarni birlashtiramiz (try.ipynb dagi format_chat_ctx
    # bilan bir xil): masalan baza (A,...) bilan tugab ustiga (A, savol) qo'shilsa,
    # ikkalasi bitta assistant turniga qo'shiladi — aks holda chat template'da
    # ikkita ketma-ket <|im_start|>assistant bloki paydo bo'ladi.
    messages = []
    for r, c in list(ctx) + [("user", user_text)]:
        c = normalize_text(c)
        if not c:
            continue
        if messages and messages[-1]["role"] == r:
            messages[-1]["content"] += f" {c}"
        else:
            messages.append({"role": r, "content": c})
    return {"messages": messages, "label": label,
            "pattern": pattern, "domain": domain}


def dedupe_key(ex):
    return (tuple((m["role"], m["content"]) for m in ex["messages"]),)


# ---------------------------------------------------------------------------
# Diktovka: telefon, passport, tug'ilgan sana, jshshir, karta, manzil.
# Bank call center'da shaxsni tasdiqlash oqimi eng ko'p uchraydigan holat,
# shuning uchun kontekstlar bank uslubida.
# ---------------------------------------------------------------------------
VERIFY_OPENERS = [
    "kartamni bloklamoqchiman",
    "kartamni qayta tiklamoqchiman",
    "hisobim bo'yicha ma'lumot kerak edi",
    "pul o'tkazmasi kelgan ekan shuni olmoqchi edim",
    "kartamdan noma'lum to'lov yechilibdi",
]
# Foydalanuvchi hujjatni izlab turgan paytdagi to'xtash almashuvi.
# Bu javobdan oldin qo'shilib, 2-3 bosqichli real tarix hosil qiladi:
#   A: passport seriyangizni ayting
#   U: xo'p hozir
#   A: aha kutyapman
#   U: ab 145 21 13   <- shu misolning javobi
STALL_USER = [
    "xo'p hozir", "ha hozir", "hozir aytaman", "bir soniya", "shoshmang",
    "hozir topib beray", "ozgina kuting", "hozir ko'rib aytaman",
    "shoshmang qidiryapman", "ha hozir topaman", "bir daqiqa hozir",
    "hozir hujjatni olib kelay",
]
STALL_ASSISTANT = [
    "aha kutyapman", "xo'p kutaman", "albatta shoshilmang",
    "mayli kutib turaman", "ha albatta shoshilmang", "xo'p bo'ladi kutaman",
]


def verify_ctx(rng: random.Random, ask: str, allow_stall: bool = True):
    """Diktovka konteksti: yo bevosita savol, yo bank tasdiqlash oqimi.
    ~35% holatda javobdan oldin 'kutish' almashuvi qo'shiladi (ko'p bosqichli tarix)."""
    if rng.random() < 0.45:
        ctx = [(A, ask)]
    else:
        opener = rng.choice(VERIFY_OPENERS)
        ctx = [(A, "bank axborot xizmati assalomu alaykum"),
               (U, opener),
               (A, f"albatta shaxsingizni tasdiqlash uchun {ask}")]
    if allow_stall and rng.random() < 0.35:
        ctx = ctx + [(U, rng.choice(STALL_USER)),
                     (A, rng.choice(STALL_ASSISTANT))]
    return ctx


PHONE_ASKS = [
    "telefon raqamingizni aytib yuboring",
    "qaysi raqamga sms yuboray",
    "bog'lanish uchun raqamingizni qoldiring",
    "raqamingizni to'liq ayting",
    "sms kod yuborish uchun raqamingizni ayting",
    "aloqa uchun telefon raqamingiz qanday",
    "qaysi raqamingizga qo'ng'iroq qilaylik",
]


def _op_reading(op: str, rng: random.Random) -> str:
    """Operator kodi o'qishi: 91 -> 'to'qson bir' / 'to'qson birlik'.
    'lik' — shu kompaniyaga tegishli belgisi, u kodni yakunlaydi.
    YUMALOQ o'nlik kodlar (20/50/90 -> 'yigirma'/'ellik'/'to'qson') keyingi
    raqam bilan qo'shilib ketishi mumkin ('to'qson'+'sakkiz'=98), shuning uchun
    ular DOIM aniq o'qiladi — 'lik' bilan yoki raqamma-raqam ('to'qqiz nol')."""
    words = group_words(op).split()
    if op.endswith("0"):                      # yumaloq o'nlik -> aniqlik shart
        if rng.random() < 0.6:
            words[-1] += "lik"                # to'qsonlik
            return " ".join(words)
        return digit_words(op)                # to'qqiz nol (raqamma-raqam)
    if rng.random() < 0.45:                   # oddiy kod: 'lik' ixtiyoriy
        words[-1] += "lik"
    return " ".join(words)


def gen_phone_examples(rng: random.Random, n_each: int):
    """Telefon = operator kodi (2) + 7 raqam = 9 raqam. Chegara-fokusli chala
    javoblar (8, 7... raqam) va 'lik' operator belgisi bilan boyitilgan."""
    out = []
    for _ in range(n_each):
        op = rng.choice(OPCODES)
        tail = "".join(str(rng.randrange(10)) for _ in range(7))
        prefix = rng.choice(["telefon raqamim ", "raqamim ", "", "", ""])
        # TO'LIQ (9 raqam): operator (lik bilan/siz) + 7 raqamli qism
        ctx = verify_ctx(rng, rng.choice(PHONE_ASKS))
        full = f"{_op_reading(op, rng)} {render_number(tail, rng)}"
        out.append(make_example(ctx, (prefix + full).strip(), 1, "phone_full", "dictation"))
        # CHALA (<9 raqam): chegaraga yaqin uzunliklar (8, 7) ko'p
        for L in boundary_cut_lengths(9, rng, k=2):
            cctx = verify_ctx(rng, rng.choice(PHONE_ASKS))
            if L == 1:
                cut = digit_words(op[:1])
            elif L == 2:                     # faqat operator kodi (lik bilan) — davomi kutiladi
                cut = _op_reading(op, rng)
            else:                            # operator + qisman 7 raqamli qism
                cut = f"{_op_reading(op, rng)} {render_number(tail[:L - 2], rng)}"
            out.append(make_example(cctx, (prefix + cut).strip(), 0, "phone_cut", "dictation"))
    return out


def gen_id_field(rng: random.Random, n_each: int, field: str, n: int,
                 asks, prefixes):
    """Umumiy N-xonali ID maydoni (jshshir=14, karta=16 ...).
    TO'LIQ = aynan N raqam (label 1), CHALA = chegaraga yaqin qisqa uzunliklar
    (label 0), turli o'qish uslublarida. Model kontekst(savol) -> kutilgan raqam
    SONini shu tarzda o'rganadi."""
    out = []
    for _ in range(n_each):
        d = "".join(str(rng.randrange(10)) for _ in range(n))
        prefix = rng.choice(prefixes)
        ctx = verify_ctx(rng, rng.choice(asks))
        out.append(make_example(ctx, (prefix + render_number(d, rng)).strip(),
                                1, field + "_full", "dictation"))
        for L in boundary_cut_lengths(n, rng, k=2):
            cctx = verify_ctx(rng, rng.choice(asks))
            out.append(make_example(cctx, (prefix + render_number(d[:L], rng)).strip(),
                                    0, field + "_cut", "dictation"))
    return out


PASSPORT_ASKS = [
    "passport seriya va raqamingizni ayting",
    "passport ma'lumotlaringizni aytib yuboring",
    "hujjat seriya raqamini ayting",
    "id karta seriya raqamingizni ayting",
    "passportingiz seriyasi va raqami qanday",
    "shaxsni tasdiqlovchi hujjat raqamini ayting",
]
# FAQAT raqam so'ralgan (harflarsiz) — javob 7 xonali raqam bo'lishi kerak.
# Foydalanuvchi kuzatgan xato aynan shu oqimda: STT raqamni bo'lak-bo'lak
# ("yetti yuz o'n yigirma ikki") beradi, chala bo'lsa ham grammatik "tugagan"dek
# ko'rinadi. Model raqamlar SONiga qarab tugaganlikni aniqlashi kerak.
PASSPORT_NUMBER_ASKS = [
    "passportingizning raqamlarini ayta olasizmi",
    "endi passport raqamini ayting seriyasi shart emas",
    "passport raqamingizni ayting yetti xonali raqam",
    "hujjatning raqam qismini ayting",
    "raqamlarini ham ayting",
]
# Passport so'ralgan kontekstda TUGAGAN, lekin diktovka bo'lmagan javoblar —
# model "passport konteksti = har doim raqam kutish" deb o'rganib qolmasligi uchun
PASSPORT_SIDE_COMPLETE = [
    "bir daqiqa passportimni olib kelay",
    "hozir passportimni ko'rib aytaman bir soniya",
    "passportim yonimda emas keyinroq aytsam bo'ladimi",
    "id karta ham bo'ladimi",
    "eski passport raqamini aytsam bo'ladimi",
    "passport raqamim esimda yo'q",
]
PASSPORT_SIDE_CUT = [
    "passport raqamim",
    "passportim seriyasi",
    "seriyasi",
    "passport raqamim esimda yo'q lekin",
    "hozir passportimni ochib",
]


def gen_passport_examples(rng: random.Random, n_each: int):
    out = []
    for _ in range(n_each):
        series = rng.choice(PASSPORT_SERIES)
        # STT harflarni ikki xil beradi: fonetik ("a de") yoki xom harf ("a d")
        spelled = rng.choice([
            " ".join(LETTER_NAMES[c] for c in series),
            " ".join(series),
        ])
        d7 = "".join(str(rng.randrange(10)) for _ in range(7))
        g1, g2, g3 = d7[:3], d7[3:5], d7[5:7]
        ctx = verify_ctx(rng, rng.choice(PASSPORT_ASKS))
        prefix = rng.choice(["passport seriyam ", "passportim ", "id kartam ",
                             "", "", ""])
        form = rng.randrange(5)
        if form == 0:    # yaxlit raqamlar: ab 1234567
            full = f"{series} {d7}"
            cut = f"{series} {d7[:rng.randrange(1, 6)]}"
        elif form == 1:  # guruhlangan raqamlar: ab 123 45 67
            full = f"{series} {g1} {g2} {g3}"
            cut = rng.choice([f"{series} {g1}", f"{series} {g1} {g2}"])
        elif form == 2:  # harflab + raqamma-raqam: a be bir ikki uch ...
            full = f"{spelled} {digit_words(d7)}"
            cut = rng.choice([spelled,
                              f"{spelled} {digit_words(d7[:rng.randrange(1, 6)])}"])
        elif form == 3:  # seriya yaxlit + raqamlar so'z bilan: ab bir ikki uch ...
            full = f"{series} {digit_words(d7)}"
            cut = f"{series} {digit_words(d7[:rng.randrange(1, 6)])}"
        else:            # harflab + bo'lak-o'qish: a be yetti yuz o'n yigirma ikki yetmish uch
            chunks = read_chunks(d7, rng)
            full = f"{spelled} {' '.join(chunks)}"
            # chala: harflar bor, lekin raqamlar to'liq emas (1..n-1 bo'lak)
            k = rng.randrange(1, len(chunks))
            cut = f"{spelled} {' '.join(chunks[:k])}"
        r = rng.random()
        if r < 0.10:     # bitta harfdan keyin uzilgan: "a"
            cut, prefix_cut = spelled.split()[0], prefix
        elif r < 0.20:   # o'zini tuzatish boshlangan: "ab yo'q kechirasiz"
            cut, prefix_cut = f"{series} yo'q kechirasiz", ""
        elif r < 0.30:   # osilib qolgan prefiks
            cut, prefix_cut = rng.choice(PASSPORT_SIDE_CUT), ""
        else:
            prefix_cut = prefix
        out.append(make_example(ctx, prefix + full, 1, "passport_full", "dictation"))
        out.append(make_example(ctx, (prefix_cut + cut).strip(), 0,
                                "passport_cut", "dictation"))

        # o'zini tuzatib to'liq aytgan: "ab yo'q kechirasiz ad 1234567" — TUGAGAN
        if rng.random() < 0.12:
            wrong = rng.choice([s for s in PASSPORT_SERIES if s != series])
            corr = f"{wrong} yo'q kechirasiz {series} {d7}"
            out.append(make_example(ctx, corr, 1, "passport_full", "dictation"))

        # "qaytadan ayting" oqimi: chala aytilgandan keyin operator qayta so'raydi
        if rng.random() < 0.15:
            rep_ctx = ctx + [(U, f"{series} {g1}"),
                             (A, "kechirasiz yaxshi eshitilmadi boshidan to'liq ayting")]
            if rng.random() < 0.5:
                out.append(make_example(rep_ctx, f"{series} {d7}", 1,
                                        "passport_full", "dictation"))
            else:
                out.append(make_example(rep_ctx, f"{series} {g1} {g2}", 0,
                                        "passport_cut", "dictation"))

    # FAQAT RAQAM so'rovi (harflarsiz) — chala 7 xonali raqam = TUGALLANMAGAN.
    # Foydalanuvchi kuzatgan xatoning aynan o'zi: "yetti yuz o'n yigirma ikki"
    # (5 raqam) grammatik tugagandek, lekin passport 7 raqam bo'lishi kerak.
    # Har xil o'qish uslubi va har xil chala uzunlik bilan puxta qamraymiz.
    for _ in range(n_each):
        d7 = "".join(str(rng.randrange(10)) for _ in range(7))
        ctx = verify_ctx(rng, rng.choice(PASSPORT_NUMBER_ASKS))
        prefix = rng.choice(["raqami ", "raqamim ", "", "", ""])
        style = rng.randrange(3)
        if style == 0:      # bo'lak-so'z: yetti yuz o'n yigirma ikki yetmish uch
            chunks = read_chunks(d7, rng)
            full = " ".join(chunks)
            cut = " ".join(chunks[:rng.randrange(1, len(chunks))])
        elif style == 1:    # raqamma-raqam: yetti bir nol uch ...
            full = digit_words(d7)
            cut = digit_words(d7[:rng.randrange(1, 7)])
        else:               # yaxlit raqamlar: 7102273
            full = d7
            cut = d7[:rng.randrange(1, 7)]
        out.append(make_example(ctx, (prefix + full).strip(), 1,
                                "passport_num_full", "dictation"))
        out.append(make_example(ctx, (prefix + cut).strip(), 0,
                                "passport_num_cut", "dictation"))

    # Kontekst kontrasti: faqat seriya so'ralganda "aa" TUGAGAN javob
    series_asks = ["passportingiz qaysi seriyada",
                   "hujjatingiz seriyasi qaysi harflar"]
    for series in PASSPORT_SERIES:
        spelled = " ".join(LETTER_NAMES[c] for c in series)
        for ask in series_asks:
            for ctx in ([(A, ask)], verify_ctx(rng, ask)):
                for ans in (series, spelled, f"{series} seriyada"):
                    out.append(make_example(ctx, ans, 1, "passport_series", "dictation"))

    # Diktovka bo'lmagan to'liq javoblar passport kontekstida
    for text in PASSPORT_SIDE_COMPLETE:
        for _ in range(3):
            ctx = verify_ctx(rng, rng.choice(PASSPORT_ASKS))
            out.append(make_example(ctx, text, 1, "passport_side", "dictation"))
    return out


# So'rov turlari — MUHIM: javobning tugaganligi so'rov qaysi maydonlarni
# talab qilishiga bog'liq. Bir xil "...yil" javobi YIL so'ralganda tugagan,
# KUN-OY-YIL so'ralganda tugallanmagan.
YEAR_ASKS = [
    "tug'ilgan yilingizni ayting",
    "necha yilda tug'ilgansiz",
    "tug'ilgan yilingiz qaysi",
    "qaysi yili tug'ilgansiz",
]
DAY_ASKS = [
    "oyning nechanchi kunida tug'ilgansiz",
    "tug'ilgan kuningiz oyning nechanchisida",
    "sananing faqat kunini ayting",
]
FULL_ASKS = [
    "tug'ilgan sanangizni kun oy va yil ko'rinishida to'liq ayting",
    "tug'ilgan sanangizni to'liq ayting kun oy va yil",
    "iltimos tug'ilgan sanangizni kun oy va yil bilan ayting",
    "to'liq tug'ilgan sanangiz kun oy va yil bo'yicha kerak",
]
# num2words da ikki so'zli (qo'shma) kunlar — partial (chala) kun yasash uchun
COMPOUND_DAYS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 31]


def _year_full_forms(y):
    yw = year_words(y)                       # bir ming to'qqiz yuz to'qson uchinchi yil
    forms = [yw, f"{y} yil", str(y)]
    if y % 100:                              # xalqona: to'qson uchinchi yil
        forms.append(ordinalize(num2words(y % 100)) + " yil")
    return forms


def _year_cut_forms(y):
    """Chala yil: oxirgi tartib son yoki 'yil' aytilmagan."""
    head = "bir ming to'qqiz yuz" if y < 2000 else "ikki ming"
    forms = [head, f"tug'ilgan yilim {head.split()[0]}"]
    w = num2words(y % 100)                   # "to'qson uch"
    if len(w.split()) > 1 and y < 2000:      # "...to'qqiz yuz to'qson" (uchinchisiz)
        forms.append(f"{head} {w.split()[0]}")
    return forms


def gen_birthdate_examples(rng: random.Random, n_each: int):
    out = []
    for _ in range(n_each):
        y = rng.randrange(1950, 2006)
        day = rng.randrange(1, 29)
        cday = rng.choice(COMPOUND_DAYS)     # chala kun yasash uchun qo'shma kun
        month = rng.choice(MONTHS)
        yw = year_words(y)
        dw = ordinalize(num2words(day))      # o'n beshinchi
        cdw = ordinalize(num2words(cday))    # o'ttiz birinchi
        cday_partial = num2words(cday).split()[0]   # "o'ttiz" (birinchisiz)

        kind = rng.choice(["year", "year", "full", "full", "full", "day"])

        if kind == "year":
            ask = rng.choice(YEAR_ASKS)
            ctx = verify_ctx(rng, ask)
            out.append(make_example(ctx, rng.choice(_year_full_forms(y)), 1,
                                    "birthdate_year_full", "dictation"))
            out.append(make_example(ctx, rng.choice(_year_cut_forms(y)), 0,
                                    "birthdate_year_cut", "dictation"))

        elif kind == "day":
            ask = rng.choice(DAY_ASKS)
            ctx = verify_ctx(rng, ask)
            full = rng.choice([dw, f"{dw} kuni", f"oyning {dw} kuni", f"{day}"])
            out.append(make_example(ctx, full, 1, "birthdate_day_full", "dictation"))
            # chala: qo'shma kunning yarmi ("yigirma"/"o'ttiz") yoki to'ldiruvchi
            cut = rng.choice([cday_partial, "oyning", "men oyning"])
            out.append(make_example(ctx, cut, 0, "birthdate_day_cut", "dictation"))

        else:  # full — kun+oy+yil hammasi kerak
            ask = rng.choice(FULL_ASKS)
            ctx = verify_ctx(rng, ask)
            full = rng.choice([
                f"{yw} {dw} {month}",                       # ...yil o'n beshinchi mart
                f"{dw} {month} {yw}",                       # o'n beshinchi mart ...yil
                f"{day} {month} {y} yil",                   # 15 mart 1993 yil
                f"{y} yil {day} {month}",                   # 1993 yil 15 mart
                f"{cdw} {month} {y} yil",
            ])
            out.append(make_example(ctx, full, 1, "birthdate_full", "dictation"))
            # chala — bir yoki bir necha maydon yetishmaydi:
            cut = rng.choice([
                yw,                                  # faqat yil (yil-so'rovda tugagan!)
                rng.choice(_year_full_forms(y)),     # faqat yil (raqamli/xalqona)
                f"{yw} {cday_partial}",              # ...yil o'ttiz  (foydalanuvchi misoli)
                f"{yw} {cdw}",                       # ...yil o'ttiz birinchi (oy yo'q)
                f"{y} yil {day}",                    # 1993 yil 15 (oy yo'q)
                f"{dw} {month}",                     # o'n beshinchi mart (yil yo'q, davomi kutiladi)
                rng.choice(_year_cut_forms(y)),      # chala yil
            ])
            out.append(make_example(ctx, cut, 0, "birthdate_cut", "dictation"))

        # KROSS-KONTRAST: aynan bir xil to'liq yil satri ikki xil so'rovda
        # qarama-qarshi label oladi — model matnga emas, so'rovga qarashi shart.
        if rng.random() < 0.5:
            ys = rng.choice(_year_full_forms(y))
            yctx = verify_ctx(rng, rng.choice(YEAR_ASKS), allow_stall=False)
            fctx = verify_ctx(rng, rng.choice(FULL_ASKS), allow_stall=False)
            out.append(make_example(yctx, ys, 1, "birthdate_year_full", "dictation"))
            out.append(make_example(fctx, ys, 0, "birthdate_ctx_cut", "dictation"))
    return out


JSHSHIR_ASKS = [
    "jshshir raqamingizni ayting",
    "pinfl raqamingizni aytib yuboring",
    "o'n to'rt xonali jshshir raqamingizni ayting",
    "jshshir yoki pinfl raqamingiz kerak bo'ladi ayting",
    "shaxsiy identifikatsiya raqamingizni to'liq ayting",
]
CARD_ASKS = [
    "karta raqamingizni ayting",
    "o'n olti xonali karta raqamini aytib yuboring",
    "qaysi karta bo'yicha murojaat qilyapsiz raqamini ayting",
    "kartangizning to'liq raqamini ayting",
]
CARD_LAST4_ASKS = [
    "kartangizning oxirgi to'rt raqamini ayting",
    "xavfsizlik uchun kartaning oxirgi to'rt raqamini ayting",
    "kartangizning oxirgi to'rt raqami qaysi",
]


def gen_jshshir_examples(rng: random.Random, n_each: int):
    return gen_id_field(rng, n_each, "jshshir", 14, JSHSHIR_ASKS,
                        ["jshshirim ", "pinfl raqamim ", "", "", ""])


def gen_card_examples(rng: random.Random, n_each: int):
    out = gen_id_field(rng, n_each, "card", 16, CARD_ASKS,
                       ["karta raqamim ", "", "", ""])
    # "oxirgi 4 raqam" konteksti — uch tomonlama:
    #   (a) 4 raqam + last4 so'rovi        -> TUGAGAN
    #   (b) 1-3 raqam + last4 so'rovi      -> TUGALLANMAGAN (aynan shu yetishmasdi:
    #       "oltmish" = 60 = 2 raqam, 4 kerak -> chala)
    #   (c) 4 raqam + TO'LIQ karta so'rovi -> TUGALLANMAGAN (16 kerak edi)
    for _ in range(n_each // 3):
        four = "".join(str(rng.randrange(10)) for _ in range(4))
        ctx = verify_ctx(rng, rng.choice(CARD_LAST4_ASKS))
        out.append(make_example(ctx, render_number(four, rng), 1, "card_last4", "dictation"))
        # (b) chala oxirgi-4: 1-3 raqam, last4 so'rovi ostida
        L = rng.randint(1, 3)
        cctx = verify_ctx(rng, rng.choice(CARD_LAST4_ASKS))
        out.append(make_example(cctx, render_number(four[:L], rng), 0,
                                "card_last4_cut", "dictation"))
        # (c) 4 raqam to'liq karta so'rovi ostida chala
        ctx2 = verify_ctx(rng, rng.choice(CARD_ASKS))
        out.append(make_example(ctx2, render_number(four, rng), 0, "card_cut", "dictation"))
    return out


# Maydonlarning "to'liq" uzunliklari — kross-kontrast uchun kalit sonlar.
FIELD_SPECS = [
    ("card_last4",    4,  CARD_LAST4_ASKS),
    ("passport_num",  7,  PASSPORT_NUMBER_ASKS),
    ("phone",         9,  PHONE_ASKS),
    ("jshshir",       14, JSHSHIR_ASKS),
    ("card",          16, CARD_ASKS),
]


def gen_length_contrast(rng: random.Random, n_each: int):
    """KROSS-MAYDON KONTRAST — shortcut'ni sindiradi.

    Bir xil raqam soni maydonga qarab to'liq yoki chala bo'ladi:
      7 raqam  -> passport ostida TO'LIQ, telefon/jshshir/karta ostida CHALA
      9 raqam  -> telefon ostida TO'LIQ, jshshir/karta ostida CHALA
      14 raqam -> jshshir ostida TO'LIQ, karta ostida CHALA
      4 raqam  -> card_last4 ostida TO'LIQ, boshqalar ostida CHALA
    Bir xil raqam-satri turli savollar ostida qarama-qarshi label oladi -> model
    raqam SONiga emas, KONTEKSTga (qaysi maydon so'ralgan) qarashga majbur bo'ladi."""
    out = []
    # Og'irlik: uzun maydonlar (9,14) tabiatan kamroq "chala" oladi (ulardan uzun
    # maydon kam), shuning uchun ularni ko'proq tortamiz -> balans yaxshilanadi.
    weighted_lens = [4, 7, 7, 9, 9, 9, 14, 14, 14, 16]
    for _ in range(n_each):
        L = rng.choice(weighted_lens)
        d = "".join(str(rng.randrange(10)) for _ in range(L))
        for field, N, asks in FIELD_SPECS:
            if L == N:                       # bu maydon uchun TO'LIQ
                pat = field + ("_full" if field != "card_last4" else "")
                lbl = 1
            elif L < N:                      # bu maydon uchun hali CHALA (davomi kutiladi)
                pat = field + ("_last4_cut" if field == "card_last4" else "_cut")
                lbl = 0
            else:
                continue                     # L > N: bu maydonga taalluqli emas
            ctx = verify_ctx(rng, rng.choice(asks))
            out.append(make_example(ctx, render_number(d, rng), lbl, pat, "dictation"))
    return out


# Raqamdan KEYIN keladigan so'zlar — endi SO'Z MA'NOSI hal qiladi, raqam soni emas.
# Model qoidasi: raqam bilan tugasa -> soniga qarab; SO'Z bilan tugasa -> ma'nosiga.
NUM_DONE_WORDS = [   # -> gap TUGADI (label 1), raqam kam bo'lsa ham (LLM keyin so'raydi)
    "mana shu raqamim", "shu raqamim", "mana shu", "hammasi shu", "shu xolos",
    "boshqa esimda yo'q", "boshqasini bilmayman", "shundan iborat", "mana shuncha",
    "boshqa raqam yo'q", "shu ekan xolos", "esimda shuncha qolibdi", "hammasi shuncha",
]
NUM_MORE_WORDS = [   # -> gap TUGAMADI (label 0), davomi bor
    "hozir ko'rinmayapti", "qolganini hozir topaman", "davomini aytaman",
    "shoshmang yana bor", "keyingi raqamlarni aytaman", "hozir davom ettiraman",
    "yana raqam bor", "to'xtang qolgani ham bor", "bir soniya qolganini aytaman",
    "hozir qolganini ko'rib aytaman", "yana bor to'xtang",
]


def gen_number_trailing_words(rng: random.Random, n_each: int):
    """Raqam + SO'Z bilan tugash: so'z MA'NOSI labelni hal qiladi (raqam soni EMAS).

      'ab 123 mana shu raqamim'       -> TUGADI (label 1) — raqam kam (4<7) bo'lsa ham
      'ab 123 hozir ko'rinmayapti'    -> TUGAMADI (label 0) — davomi bor
      'ab 123'  (raqam bilan tugadi)  -> soniga qarab (bu boshqa generatorlarda)

    Aynan bir xil raqamdan keyin turli ma'noli so'z -> qarama-qarshi label."""
    out = []
    specs = [("passport_num", 7, PASSPORT_NUMBER_ASKS, PASSPORT_SERIES),
             ("phone", 9, PHONE_ASKS, None),
             ("card", 16, CARD_ASKS, None),
             ("jshshir", 14, JSHSHIR_ASKS, None)]
    for _ in range(n_each):
        field, N, asks, series_pool = rng.choice(specs)
        done = rng.random() < 0.5
        # TUGADI so'zlari har uzunlikda; TUGAMADI so'zlari chala raqamda mantiqiy
        L = rng.randint(1, N) if done else rng.randint(1, N - 1)
        d = "".join(str(rng.randrange(10)) for _ in range(L))
        num = render_number(d, rng)
        if series_pool and rng.random() < 0.5:      # passportda ba'zan harf prefiksi
            num = " ".join(rng.choice(series_pool)) + " " + num
        prefix = rng.choice(["", "", "raqamim ", "mana raqamim ", "hozir raqamim "])
        ctx = verify_ctx(rng, rng.choice(asks))
        if done:
            text = f"{prefix}{num} {rng.choice(NUM_DONE_WORDS)}"
            out.append(make_example(ctx, text, 1, "num_done_word", "dictation"))
        else:
            text = f"{prefix}{num} {rng.choice(NUM_MORE_WORDS)}"
            out.append(make_example(ctx, text, 0, "num_more_word", "dictation"))
    return out


# Ikkilanish (hesitation) — user o'ylayotganini bildiradi.
HESITATION = [
    "hm", "hmm", "mmm", "ho'sh", "xo'sh", "to'xta", "to'xtang", "shoshmang",
    "ha hozir", "hozir", "hozir hozir", "anu", "haligi", "bir soniya",
    "bir sekund", "hozir eslay", "o'ylab ko'ray", "hozir o'ylab ko'ray",
    "kuting", "kutib turing", "nima edi", "aa", "ee", "ha shu", "ha hozir aytaman",
    "hozir aytaman", "hozir topaman", "shoshmang hozir", "eslayapman",
    "hozir esladim shoshmang", "ha o'ylayapman", "hozir bir qarab olay",
]
HES_ASKS_GENERAL = [
    "ismingizni ayting", "familiyangizni ayting", "sizni tinglayapman",
    "buyurtmangizni ayting", "manzilingizni ayting", "qanday yordam bera olaman",
]
HES_NUM_SPECS = [
    ("phone", 9, PHONE_ASKS), ("passport_num", 7, PASSPORT_NUMBER_ASKS),
    ("jshshir", 14, JSHSHIR_ASKS), ("card", 16, CARD_ASKS),
    ("card_last4", 4, CARD_LAST4_ASKS),
]


def gen_hesitation(rng: random.Random, n_each: int):
    """Ikkilanish so'zlari — user o'ylayapti, gapni tugatmagan.
      AI so'raydi -> user 'ha hozir' / 'hmm' / 'to'xta' -> TUGAMADI (label 0)
    Gap OXIRI ikkilanish bo'lsa -> hali o'ylayapti -> label 0.
    KONTRAST: ikkilanish BOSHDA + to'liq javob -> TUGADI (label 1) — model 'hmm'
    ning POZITSIYAsini o'rgansin (o'rtada/boshda bo'lsa gap tugagan bo'lishi mumkin)."""
    out = []
    for _ in range(n_each):
        field, N, asks = rng.choice(HES_NUM_SPECS)
        r = rng.random()
        if r < 0.55:                     # sof ikkilanish (javob boshlanmagan) -> 0
            if rng.random() < 0.25:
                ctx = [(A, rng.choice(HES_ASKS_GENERAL))]
            else:
                ctx = verify_ctx(rng, rng.choice(asks), allow_stall=False)
            out.append(make_example(ctx, rng.choice(HESITATION), 0, "hesitation", "dictation"))
        elif r < 0.75:                   # javob boshlanib, ikkilanishda uzildi -> 0
            ctx = verify_ctx(rng, rng.choice(asks), allow_stall=False)
            L = rng.randint(1, N - 1)
            d = "".join(str(rng.randrange(10)) for _ in range(L))
            out.append(make_example(ctx, f"{render_number(d, rng)} {rng.choice(HESITATION)}",
                                    0, "hesitation", "dictation"))
        else:                            # KONTRAST: ikkilanish BOSHDA + TO'LIQ javob -> 1
            ctx = verify_ctx(rng, rng.choice(asks), allow_stall=False)
            d = "".join(str(rng.randrange(10)) for _ in range(N))
            out.append(make_example(ctx, f"{rng.choice(HESITATION)} {render_number(d, rng)}",
                                    1, "hesitation_mid", "dictation"))
    return out


def gen_address_examples(rng: random.Random, n_each: int):
    out = []
    ctxs = [
        [(A, "manzilingizni ayting")],
        [(A, "qayerga yetkazib beramiz manzilni ayting")],
        [(A, "manzilni to'liq aytib yuboring")],
    ]
    for _ in range(n_each):
        d = rng.choice(SLOTS["DISTRICT"])
        o1, o2 = rng.sample(SLOTS["ORD"], 2)
        st = rng.choice(SLOTS["STREET"])
        ctx = rng.choice(ctxs)
        if rng.random() < 0.5:
            full = f"{d} tumani {o1} kvartal {o2} uy"
            cut = f"{d} tumani {o1}"
        else:
            full = f"{d} tumani {st} ko'chasi {o1} uy"
            cut = f"manzilim {d} tumani {st}"
        out.append(make_example(ctx, full, 1, "address_full", "dictation"))
        out.append(make_example(ctx, cut, 0, "address_cut", "dictation"))
    return out


# Uzun gaplar uchun yengil kontekstlar (gaplar o'zi to'liq, ko'p kontekst shart emas)
LONG_CTX_OPENERS = {
    "bank":    ["bank axborot xizmati assalomu alaykum eshitaman",
                "assalomu alaykum bank yordam markazi sizni tinglayapman"],
    "telecom": ["aloqa operatori assalomu alaykum sizga qanday yordam beray",
                "salom mobil aloqa xizmati eshitaman"],
    "clinic":  ["klinika qabulxonasi assalomu alaykum",
                "salom shifoxona axborot xizmati eshitaman"],
    "taxi":    ["taksi xizmati assalomu alaykum", "salom taksi xizmati eshitaman"],
    "food":    ["assalomu alaykum buyurtmangizni eshitaman"],
    "general": ["assalomu alaykum sizga qanday yordam bera olaman"],
}


def gen_long_examples(rng: random.Random):
    """Uzun chalkash gaplar — har biri full(1)/cut(0), bir nechta kontekstda.
    Kontekst: yo kontekstsiz, yo bitta operator salomi, yo 'kutish' almashuvi bilan."""
    out = []
    for domain, full, cut in LONG_RAMBLE:
        openers = LONG_CTX_OPENERS[domain]
        # 3 xil kontekst: kontekstsiz, salom bilan, salom+kutish (ko'p bosqichli)
        ctxs = [
            [],
            [(A, rng.choice(openers))],
            [(A, rng.choice(openers)),
             (U, rng.choice(STALL_USER)),
             (A, rng.choice(STALL_ASSISTANT))],
        ]
        for ctx in ctxs:
            out.append(make_example(ctx, full, 1, "long_full", domain))
            out.append(make_example(ctx, cut, 0, "long_cut", domain))
        # gapni yana bog'lovchi bilan cho'zib uzish (uzunroq label=0)
        for conj in rng.sample(["va", "lekin", "chunki", "keyin"], 1):
            out.append(make_example([], f"{full} {conj}", 0, "long_cut", domain))
    return out


# ---------------------------------------------------------------------------
# Asosiy generatsiya
# ---------------------------------------------------------------------------
def generate(n_total: int, seed: int):
    rng = random.Random(seed)
    examples = []

    # 1) Tugagan gaplar — har bir matn bir nechta kontekst bilan
    complete_statements = []  # bog'lovchi qo'shish uchun baza
    for domain, pattern, template in COMPLETE:
        for text in expand_template(template, rng, cap=60):
            for ctx in context_variants(domain, rng, k=3):
                examples.append(make_example(ctx, text, 1, pattern, domain))
            if pattern in CONJ_BASE_PATTERNS:
                complete_statements.append((domain, text))

    # 2) Savol-javob elliptik juftliklar (kontekst majburiy).
    # Savol oldiga qo'shiladigan baza faqat user turni bor kontekst bo'lishi
    # kerak, aks holda ketma-ket ikkita assistant savoli g'alati suhbat beradi.
    for domain, question, ans_template in QA_PAIRS:
        with_user = [c for c in CONTEXTS[domain]
                     if any(r == U for r, _ in c) and question not in c[-1][1]]
        for text in expand_template(ans_template, rng, cap=40):
            bases = [[]] + ([rng.choice(with_user)] if with_user else [])
            for base in bases:
                ctx = base + [(A, question)]
                examples.append(make_example(ctx, text, 1, "elliptical", domain))

    # 3) Tasdiq/rad javoblari (xuddi shu qoida)
    for question, answers in ACK_PAIRS:
        for ans in answers:
            for domain in ["taxi", "food", "bank", "clinic", "telecom"]:
                with_user = [c for c in CONTEXTS[domain] if any(r == U for r, _ in c)]
                for base in [[]] + with_user:
                    ctx = base + [(A, question)]
                    examples.append(make_example(ctx, ans, 1, "ack", domain))

    # 4) Tugallanmagan gaplar (qo'lda yozilgan naqshlar)
    for domain, pattern, template in INCOMPLETE:
        for text in expand_template(template, rng, cap=60):
            for ctx in context_variants(domain, rng, k=3):
                examples.append(make_example(ctx, text, 0, pattern, domain))

    # 5) Tugagan darak gap + osilib qolgan bog'lovchi -> tugallanmagan
    for domain, text in complete_statements:
        for conj in rng.sample(SLOTS["CONJ"], 2):
            ctx = pick_context(domain, rng)
            examples.append(make_example(ctx, f"{text} {conj}", 0, "trailing_conj", domain))

    # 6) Minnatdorchilik / yakunlash (+ kontrast: rahmatdan keyin davom)
    for text in GRATITUDE:
        for ctx in CLOSE_CONTEXTS:
            examples.append(make_example(ctx, text, 1, "gratitude", "general"))
    for text in GRATITUDE_CUT:
        for ctx in CLOSE_CONTEXTS:
            examples.append(make_example(ctx, text, 0, "gratitude_cut", "general"))

    # 7) Diktovka: telefon, passport, tug'ilgan sana, jshshir, karta, manzil.
    # Raqam maydonlari (phone/jshshir/card) endi CHEGARA-FOKUSLI: har full uchun
    # 2 ta chegaraga yaqin chala (N-1, N-2...) — model kontekst->raqam SONini o'rgansin.
    examples += gen_phone_examples(rng, n_each=360)
    examples += gen_passport_examples(rng, n_each=500)
    examples += gen_birthdate_examples(rng, n_each=380)
    examples += gen_jshshir_examples(rng, n_each=320)
    examples += gen_card_examples(rng, n_each=320)
    examples += gen_address_examples(rng, n_each=150)

    # 7b) KROSS-MAYDON KONTRAST — bir xil raqam soni maydonga qarab to'liq/chala.
    # Shortcut ("7 raqam = to'liq") ni sindiradi: model kontekstga qarashga majbur.
    examples += gen_length_contrast(rng, n_each=900)

    # 7c) RAQAM + SO'Z bilan tugash — so'z ma'nosi hal qiladi (raqam soni emas):
    #   raqam + 'mana shu raqamim' -> TUGADI;  raqam + 'hozir ko'rinmayapti' -> TUGAMADI
    examples += gen_number_trailing_words(rng, n_each=500)

    # 7d) IKKILANISH (hesitation) — user o'ylayapti: 'ha hozir', 'hmm', 'to'xta' -> TUGAMADI.
    # Kontrast: ikkilanish BOSHda + to'liq javob -> TUGADI (pozitsiya muhim).
    examples += gen_hesitation(rng, n_each=350)

    # 8) UZUN CHALKASH GAPLAR (full=1 / cut=0) — uzunlik≠tugallanmaganlik.
    # 4 marta takrorlab og'irlikni oshiramiz (dedup bir xillarni yig'ib tashlaydi,
    # lekin STALL/opener tasodifiy bo'lgani uchun har safar yangi variant chiqadi).
    for _ in range(4):
        examples += gen_long_examples(rng)

    # Deduplikatsiya (bir xil suhbat + har xil label bo'lsa - ikkalasini tashlaymiz)
    by_key = {}
    conflicts = set()
    for ex in examples:
        k = dedupe_key(ex)
        if k in by_key and by_key[k]["label"] != ex["label"]:
            conflicts.add(k)
        by_key.setdefault(k, ex)
    unique = [ex for k, ex in by_key.items() if k not in conflicts]

    # Balanslash: 50/50
    pos = [e for e in unique if e["label"] == 1]
    neg = [e for e in unique if e["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    per_class = min(len(pos), len(neg), n_total // 2)
    data = pos[:per_class] + neg[:per_class]
    rng.shuffle(data)
    return data, len(unique), len(conflicts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    data, n_unique, n_conflicts = generate(args.n_total, args.seed)

    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("uz_eou_train.jsonl", train), ("uz_eou_val.jsonl", val)]:
        with open(args.out_dir / name, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"unique candidates: {n_unique} (conflicts dropped: {n_conflicts})")
    print(f"train: {len(train)}  val: {len(val)}")
    for name, split in [("train", train), ("val", val)]:
        labels = Counter(e["label"] for e in split)
        print(f"{name} labels: {dict(labels)}")
    print("\npattern distribution (train):")
    for p, c in Counter(e["pattern"] for e in train).most_common():
        print(f"  {p:20s} {c}")
    print("\ndomain distribution (train):")
    for d, c in Counter(e["domain"] for e in train).most_common():
        print(f"  {d:12s} {c}")


if __name__ == "__main__":
    main()
