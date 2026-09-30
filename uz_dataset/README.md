# O'zbek tili EOU dataseti (livekit/turn-detector LoRA fine-tune uchun)

## Fayllar

| Fayl | Tavsif |
|---|---|
| `uz_eou_train.jsonl` | Training to'plami (~11500 misol, 50/50 balans) — **so'z ko'rinishi** |
| `uz_eou_val.jsonl` | Validatsiya to'plami (~1280 misol) — so'z ko'rinishi |
| `uz_eou_smoke_test.jsonl` | ~100 ta qo'lda tanlangan qiyin holat — fine-tune'dan oldin/keyin tez solishtirish uchun |
| `uz_eou_train_num.jsonl` | **v2 — RAQAMLI ko'rinish**: barcha raqam-so'zlar raqamga aylantirilgan ("besh yuz yigirma uch" → "523") |
| `uz_eou_val_num.jsonl` | v2 raqamli validatsiya |
| `uz_eou_smoke_test_num.jsonl` | v2 raqamli smoke-test |
| `generate_uz_eou_dataset.py` | Generator — `--n-total`, `--seed`, `--val-frac` flaglari bilan qayta ishlatish mumkin |
| `eval_baseline.py` | Modelni datasetda baholash: `python eval_baseline.py <file.jsonl> ...` |

## Ikki versiya: so'z (v1) va raqam (v2)

- **v1 (so'z)** — `uz_eou_*.jsonl`: raqamlar so'z bilan ("besh yuz yigirma uch"). Model
  so'z-guruhlaridan sanashni o'rganadi (qiyinroq).
- **v2 (raqam)** — `uz_eou_*_num.jsonl`: barcha raqam-so'zlar raqamga aylantirilgan
  ("523"). Qwen har raqamni alohida tokenlagani uchun **sanash osonlashadi**
  (5 token vs 7 token). Harflar ("ab"), tugatish so'zlari ("mana shu raqamim") va
  boshqa matn o'z holicha qoladi; faqat raqam-so'z ketma-ketliklari o'giriladi.
  Label/pattern/kontekst o'zgarmaydi; barcha full=N, cut<N tekshirilgan (0 nomuvofiqlik).

v2 ni ishlatmoqchi bo'lsang: inference'da ham xuddi shu konvertatsiya (`convert_numbers`,
`predict_pt2`) qo'llanishi shart — train/inference bir xil ko'rinishda bo'lishi uchun.

## Format

Har qator — bitta JSON obyekt:

```json
{"messages": [{"role": "assistant", "content": "qayerga borasiz"},
              {"role": "user", "content": "chilonzorga"}],
 "label": 1, "pattern": "elliptical", "domain": "taxi"}
```

- `label`: **1** = gap tugagan (EOU, agent javob berishi kerak), **0** = foydalanuvchi hali gapiryapti.
- Oxirgi xabar doim `user`. `pattern`/`domain` — faqat tahlil uchun, trainingda tashlab yuboriladi.
- Matnlar notebook'dagi `normalize_text` bilan bir xil normalizatsiyadan o'tgan:
  kichik harf, punktuatsiya olib tashlangan, `'` va `-` saqlanadi.

**Apostrof haqida:** hamma matnda ASCII `'` ishlatilgan. STT chiqishida `ʻ` (U+02BB),
`’` (U+2019) kabi variantlar uchraydi — generator ularni `'` ga keltiradi.
Runtime'dagi `normalize_text` ga ham xuddi shu almashtirishni qo'shish kerak,
aks holda train/inference orasida token nomuvofiqligi bo'ladi
(`’` punktuatsiya sifatida O'CHIB ketadi: "o'nda" → "onda").

## Qamrab olingan naqshlar

Tugallanmagan (label=0), o'zbek morfologiyasiga asoslangan:

- `verb_cut` — SOV tartibida fe'l tushib qolgan: "men bugun maktabga"
- `trailing_conj` — osilib qolgan bog'lovchi: "...bekor qilishdi va", "...lekin"
- `converb` — ravishdosh (-ib): "kartam yo'qolib"
- `conditional` — bosh gapsiz shart mayli (-sa): "agar komissiyasi baland bo'lsa"
- `hanging_case` — osilib qolgan kelishik: "hisobimdagi qoldiqni"
- `sub_clause` — yolg'iz ergash gap: "telefonim ishlamay qolgani uchun"
- `participle` — ot kutayotgan sifatdosh (-gan): "menga achchiq bo'lmagan"
- `degree` — daraja so'zi bilan uzilgan: "internetim juda"
- `enum_cut` — to'xtab qolgan sanash: "menga ikkita osh bitta", "birinchidan..."
- `filler` — kirish so'z bilan tugagan: "gap shundaki", "aytmoqchi edimki"
- `gratitude_cut` — minnatdorchilikdan keyin davom: "rahmat lekin"
- `long_cut` — uzun chalkash gap o'rtada uzilgan (pastda "Uzunlik" bo'limiga qarang)
- diktovka kesimlari: `phone_cut`, `passport_cut`, `birthdate_cut`,
  `jshshir_cut`, `card_cut`, `address_cut`

## Uzunlik ≠ tugallanmaganlik (`long_full` / `long_cut`)

**Muammo:** birinchi fine-tune qilingan model uzun, ko'p bo'lakli, chalkash jonli
gaplarning hammasini "tugallanmagan" deb baholardi. Sababi — datasetda uzun label=1
misollar faqat "toza" edi (diktovka raqamlari, grammatik silliq gaplar), uzun
chalkash tabiiy nutq esa faqat label=0 tomonda uchrardi. Model soxta qoidani
o'rgangan: *uzun + chalkash = tugallanmagan*.

**Yechim:** ~30 ta uzun, ataylab chalkash gap ("anu", "xullas", "bilasizmi",
takror so'zlar, o'zini tuzatish — real STT transkriptidagidek) har biri **ikki
versiyada**:

- `long_full` (label=1) — to'g'ri predikat bilan TUGAGAN: `-mi` so'roq yuklamasi
  ("...o'tkazsa bo'ladimi", "...saqlanib qoladimi"), fe'l yoki modal ("...nima qilay",
  "...yordam bering")
- `long_cut` (label=0) — aynan o'sha gap o'rtada, osilib qolgan ot/bog'lovchi bilan
  ("...boshqa humo kartamga", "...men hech qanaqa xarid qilmaganman bu")

Misol (foydalanuvchi topgan real holat):
```
A: ...yangi karta uchun ariza topshirishingiz mumkin yana biror yordam kerakmi
U: u muzlatib qo'yiladimi karta puli ichida puli bilan yo'qolgan yoki
   boshqa humo kartamga o'tkazsa bo'ladimi         -> long_full  (label=1)
U: ...yoki boshqa humo kartamga                     -> long_cut   (label=0)
```

Shu tarzda uzunlik label bilan bog'liqligini yo'qotadi — model gapning OXIRIGA
(predikat bormi?) qarashni o'rganadi, uzunligiga emas. Har juftlik kontekstsiz,
operator salomi bilan, va "kutish" almashuvi bilan — 3+ variantda beriladi.

## Sana javoblari kontekstga bog'liq (`birthdate_*`)

**Muammo:** javobning tugaganligi matnning o'zida emas, **assistant nimani
so'raganida**. Bir xil "...to'qson uchinchi yil" javobi:
- "tug'ilgan **yilingizni** ayting" so'raganda → **tugagan** (`birthdate_year_full`)
- "tug'ilgan sanangizni **kun oy va yil** ko'rinishida ayting" so'raganda →
  **tugallanmagan**, chunki kun-oy hali kerak (`birthdate_ctx_cut`)

Model matnga emas, **so'rovga** qarashi kerak. Buning uchun so'rovlar uch turga
ajratilgan va javob shu turga qarab belgilangan:

| So'rov turi | Misol so'rov | Tugagan javob | Tugallanmagan javob |
|---|---|---|---|
| **yil** | "tug'ilgan yilingizni ayting" | "to'qson uchinchi yil", "1993" | "bir ming to'qqiz yuz" |
| **kun** | "oyning nechanchi kunida tug'ilgansiz" | "beshinchi", "beshinchi mart" | "yigirma" (chala) |
| **to'liq** | "kun oy va yil ko'rinishida ayting" | "...yil o'ttiz birinchi mart" (3 maydon) | "...yil o'ttiz" (foydalanuvchi misoli), faqat yil, kun-oy (yilsiz) |

**Kross-kontrast:** har generatsiyada bitta to'liq yil satri ataylab **ikki
so'rovda qarama-qarshi label** bilan beriladi (yil-so'rovda 1, to'liq-so'rovda 0).
Datasetda ~130 ta shunday satr ikkala labelda uchraydi — model o'rganish uchun
kontekstni tahlil qilishga majbur bo'ladi.

Tugagan (label=1): to'liq darak/buyruq gaplar, savollar, minnatdorchilik/yakunlash
("rahmat", "bo'ldi rahmat", "hammasi joyida rahmat"), va — muhimi —
**kontekstda to'liq bo'lgan elliptik javoblar** ("qayerga borasiz" → "chilonzorga",
"nechta bo'lsin" → "uchta", tasdiq "ha/yo'q shu xolos"). Bular modelga
"fe'lsiz gap doim chala emas" ekanini o'rgatadi.

## Diktovka (bank call center — shaxsni tasdiqlash)

O'zbekiston standartlariga mos, har biri raqam / guruhlab so'z / raqamma-raqam
so'z shakllarida, to'liq (label=1) va uzilgan (label=0) juftliklar bilan:

- **Telefon**: operator kodi (90/91/93/94/95/97/98/99/33/50/55/77/88/20) + 7 raqam,
  ba'zan 998 prefiksi: "91 695 34 56", "to'qson bir olti yuz to'qson besh o'ttiz to'rt ellik olti"
- **Passport / ID karta** (eng katta blok, ~990 misol — bunda xato bo'lmasligi kerak):
  2 harf seriya (AA–AE) + 7 raqam, 5 xil o'qish shakli — "ab 1234567",
  "ab 123 45 67", "a be bir ikki uch to'rt besh olti yetti",
  "ab bir ikki uch...", "a be bir yuz yigirma uch qirq besh oltmish yetti".
  Uzilishlar: bitta harfdan keyin ("a"), seriyadan keyin, raqam o'rtasida,
  o'zini tuzatish boshlanishi ("ab yo'q kechirasiz").
  Qo'shimcha kontrastlar: tuzatib to'liq aytilgani TUGAGAN ("ab yo'q kechirasiz
  ad 7654321"); faqat seriya so'ralganda "ab" TUGAGAN, to'liq so'ralganda "ab"
  TUGALLANMAGAN; "qaytadan ayting" oqimi; diktovka bo'lmagan to'liq javoblar
  ("bir daqiqa passportimni olib kelay", "passport raqamim esimda yo'q")
- **Tug'ilgan sana** (kontekstga bog'liq — pastdagi bo'limga qarang):
  "bir ming to'qqiz yuz sakson to'rtinchi yil (o'n beshinchi mart)",
  raqamli "1984 yil 15 mart", xalqona "sakson to'rtinchi yil"
- **JShShIR / PINFL**: 14 raqam (1 asr-jins + DDMMYY + 7): yaxlit, juftlab, so'z bilan
- **Karta raqami**: 16 raqam, Uzcard 8600 / Humo 9860 BIN bilan; alohida
  **card_last4** kontrasti — "oxirgi to'rt raqam" so'ralganda 4 raqam TUGAGAN,
  to'liq raqam so'ralganda xuddi shunday qisqa javob TUGALLANMAGAN
- **Manzil**: tuman + kvartal/ko'cha + uy

Diktovka kontekstlarining ~yarmi bank tasdiqlash oqimida:
"kartamni bloklamoqchiman" → "shaxsingizni tasdiqlash uchun ... ayting" → javob.

### Chala RAQAM = tugallanmagan (`passport_num_*`)

**Muammo:** raqamli javobda tugaganlikni oxirgi so'z bildirmaydi — "yetti yuz o'n
yigirma ikki" (710 22 = 5 raqam) grammatik jihatdan tugagandek ko'rinadi, osilib
qolgan kelishik yoki fe'l yo'q. Ammo passport raqami **7 xonali** bo'lishi kerak,
demak 5 raqam — tugallanmagan. Model raqamlar SONiga qarab hukm qilishi kerak.

**Yechim:** `read_chunks` yordamchisi raqamni STT kabi tabiiy bo'laklarga bo'lib
o'qiydi (`[3,2,2]`, `[2,2,3]`, `[2,2,2,1]` va h.k.):
`7102273` → "yetti yuz o'n / yigirma ikki / yetmish uch". To'liq (7 raqam) = label 1,
har qanday boshlang'ich bo'lak (1..n-1) = label 0. Uch uslubda: bo'lak-so'z,
raqamma-raqam ("yetti bir nol..."), yaxlit ("7102273"). Harflarsiz "raqamlarini
ayting" so'rovi ham qo'shildi (foydalanuvchi kuzatgan aynan holat), va harf o'qishi
ikki xil — fonetik "a de" hamda xom "a d" (STT ikkalasini chiqaradi).

**Yondashuv (deterministik "gate" emas, model o'zi):** raqam maydonlarini
tashqi parser bilan sanash urinildi, lekin u tabiatan hal qilib bo'lmaydigan
holatlarga duch keladi ("olti yuz yigirma ikki o'n bir" = 6002211 yoki 62211?
— faqat ovoz ohangi ajratadi). Shuning uchun yechim modelning O'ZiGA topshirildi:
dataset kontekst(savol) → kutilgan raqam SONi bog'lanishini o'rgatadigan qilib
boyitilgan. (Multi-input — ovoz + matn — versiyasi kelajakda.)

### Standart uzunliklar va chegara-fokusli o'qitish

Har raqam maydoni uchun TO'LIQ = aynan N raqam (label 1), CHALA = N dan kam
(label 0), va chala misollar **chegaraga yaqin** (N-1, N-2) to'plangan — model N
chegarasini aniq o'rgansin:

| Maydon | N | So'rov namunasi |
|---|---|---|
| `passport_num` | 7 | "passport raqamini ayting" |
| `phone` | 9 | operator kodi (2) + 7; "lik" bilan: "to'qson birlik ..." |
| `jshshir` | 14 | "pinfl raqamingizni ayting" |
| `card` | 16 | "karta raqamini ayting" (+ `card_last4` = 4 kontrasti) |

Har o'qish **uslub-xilma-xil** (`render_number`): raqamma-raqam ("bir ikki uch")
va tabiiy bo'lak-so'z (`read_chunks`: juft/uch guruhlar). Telefon operator kodi
`_op_reading` orqali "lik" belgisi bilan ("to'qson birlik" = 91-lik), yumaloq
kodlar (20/50/90) esa doim aniq o'qiladi (lik yoki raqamma-raqam) — aks holda
"to'qson"+"sakkiz" = 98 bo'lib chegara yo'qolardi.

**Sifat kafolati:** har full/cut misolining raqam soni ichki parser bilan
tekshirilgan — barcha full = aynan N, barcha cut < N (0 nomuvofiqlik).

### Kross-maydon kontrast (shortcut'ni sindirish)

**Muammo:** raqam soni labelni deyarli bashorat qilardi — 7 raqam 65% "to'liq"
(passport ustunligidan), 9 raqam 86% (telefon). Model kontekstsiz "7 = to'liq"
shortcut'ini o'rganib, telefon ostidagi 7 raqamli javobni ham noto'g'ri "to'liq"
deyishi mumkin edi.

**Yechim** (`gen_length_contrast`): bir xil raqam soni turli savol ostida
qarama-qarshi label oladi — **7 raqam passport ostida TO'LIQ, telefon/jshshir/karta
ostida CHALA**; 9 raqam telefon ostida to'liq, jshshir/karta ostida chala; va h.k.
Bir xil raqam-satri ikki savol ostida beriladi -> model raqam SONiga emas,
qaysi maydon SO'RALGANiga qarashga majbur. Natijada raqam-soni↔label bog'lanishi
sindirildi: 7 raqam 65%→46%, 9 raqam 86%→57%, 14 raqam 81%→68% "to'liq"
(16 raqam 100% — bu tabiiy, kartadan uzun maydon yo'q).

### Raqamdan keyin SO'Z (`num_done_word` / `num_more_word`)

Umumiy qoida — javob nima bilan tugashiga qarab:
- **RAQAM bilan tugasa** -> raqam SONiga qarab (yuqoridagi kontekst-mantiq)
- **SO'Z bilan tugasa** -> o'sha so'zlarning MA'NOSIga qarab (raqam soni muhim emas)

Chunki turn-detector "user gapni tugatdimi" ni aniqlaydi, "raqam to'g'rimi" ni emas.
Agar user kam raqam aytib "mana shu raqamim" desa — u gapini TUGATDI (keyin LLM
"7 xona kerak, to'liq ayting" deydi).

- `num_done_word` (label 1): raqam + tugatish so'zi — "ab 123 **mana shu raqamim**",
  "...**boshqa esimda yo'q**", "...**hammasi shu**" (raqam kam bo'lsa ham TUGADI)
- `num_more_word` (label 0): raqam + davom so'zi — "ab 123 **hozir ko'rinmayapti**",
  "...**qolganini aytaman**", "...**yana bor to'xtang**" (davomi bor -> TUGAMADI)
- bare raqam ("ab 123", oxiri raqam) -> soniga qarab (boshqa generatorlarda)

Ya'ni aynan bir xil raqamdan keyin turli ma'noli so'z qarama-qarshi label beradi —
model raqam ortidagi gap ma'nosini o'qishni o'rganadi.

### Ikkilanish so'zlari (`hesitation` / `hesitation_mid`)

User o'ylayotganini bildiruvchi so'zlar — turn-detector "o'ylayaptimi yoki gapni
tugatdimi" ni bilishi kerak:

- `hesitation` (label 0): gap OXIRI ikkilanish so'zi — "**ha hozir**", "**hmm**",
  "**to'xta**", "**ho'sh**", "**o'ylab ko'ray**", "**bir soniya**", yoki raqam
  boshlanib ikkilanishda uzilsa ("...to'rt hmm") -> hali o'ylayapti -> TUGAMADI
- `hesitation_mid` (label 1): ikkilanish BOSHda + to'liq javob — "**hmm** [to'liq
  raqam]", "**to'xta** [to'liq raqam]" -> gap tugagan -> TUGADI

**Pozitsiya muhim:** "hmm" gap oxirida bo'lsa o'ylayapti (0), boshida bo'lib keyin
to'liq javob kelsa tugagan (1). Bu kontrast model "hmm" so'zining o'ziga emas,
o'rniga qarashini ta'minlaydi (aks holda har "hmm" ni "tugamagan" deb o'ylardi).

**Ochiq cheklov:** 0.5B model uchun so'z bilan aytilgan raqamlarni SANASH — eng qiyin
vazifa. Ko'p, chegara-fokusli, uslub-xilma-xil misollar bilan model kontekstdan
kutilgan sonni o'rganishga harakat qiladi, lekin bu soha eng zaif bo'lib qolishi
mumkin. Yakuniy aniqlikni real STT transkriptlarida tekshirish shart.

Domenlar: bank (kengaytirilgan — call center uchun), taksi, ovqat yetkazish,
klinika, telecom, umumiy assistant, diktovka. Kontekstlar: 0–3 oldingi xabar.

## Bazaviy natija (fine-tune'dan OLDIN, v0.4.1-intl model_q8)

```
val:   AUC 0.58   thr=0.011: TPR 0.90, TNR 0.12
smoke: AUC 0.62   thr=0.011: TPR 0.88, TNR 0.20
```

Model tugagan gaplarni yaxshi topadi, lekin tugallanmagan o'zbekcha gaplarning
~90% ini ham "tugagan" deb belgilaydi. Diktovkada ikkala yo'nalish ham buzilgan:
uzilgan hujjat raqamlarini "tugagan" deydi (jshshir_cut 0%, card_cut 9%,
phone_cut 19%, passport_cut 27% aniqlik), so'z bilan aytilgan TO'LIQ tug'ilgan
sanani esa "davom etadi" deb kutib qoladi (birthdate_full 12%). Passport
kontekstidagi "ab" (seriya javobi) va "bir daqiqa passportimni olib kelay"
kabi javoblarni ham noto'g'ri baholaydi. Fine-tune aynan shularni tuzatishi kerak.

## Ogohlantirishlar

- Dataset **sintetik** (shablon + slot). Val to'plam ham xuddi shu shablonlardan —
  undagi yaxshi natija realistik nutqqa to'liq kafolat emas. Yakuniy sifatni real
  STT transkriptlarida tekshirish kerak.
- Keyingi qadam sifatida real suhbatlardan (yoki Common Voice uz kabi manbalardan)
  bir necha yuz misol qo'shish datasetni ancha kuchaytiradi.
- Threshold'ni fine-tune'dan keyin val to'plamda qayta tanlash kerak
  (languages.json ga "uz" yozuvi qo'shiladi).
