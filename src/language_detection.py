from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

# Map langdetect ISO 639-1 -> NLLB language codes (partial; extend as needed)
LD_TO_NLLB = {
    "en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng",
    "gu": "guj_Gujr", "pa": "pan_Guru", "ur": "urd_Arab",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "zh-cn": "zho_Hans", "zh-tw": "zho_Hant", "ar": "arb_Arab",
}

def detect_lang_code(text: str) -> tuple[str, str]:
    """
    Returns (iso639_1, nllb_code). Defaults to English if unsure.
    """
    try:
        iso = detect(text)
        nllb = LD_TO_NLLB.get(iso, "eng_Latn" if iso == "en" else "eng_Latn")
        return iso, nllb
    except Exception:
        return "en", "eng_Latn"
