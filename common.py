import logging
import time
from functools import wraps
import os

enable_latency_funcs = len(os.getenv('ENABLE_LATENCY', '')) >= 1

STORAGE_DIR = "/model-cache"
STORAGE_DIR_MODEL = STORAGE_DIR + "/models"
STORAGE_DIR_DATA_FLEURS = STORAGE_DIR + "/data/fleurs"
STORAGE_DIR_CONVERSATION_DATA = STORAGE_DIR + '/data/conversation'
STORAGE_DIR_REDUCED_FLEURS = STORAGE_DIR + '/data/reduced-fleurs'
STORAGE_DIR_RESULTS = STORAGE_DIR + '/results'

logger = logging.getLogger(__name__)

latency_measurements = {}

def measure_latency(name):
    def decorator(func):
        if not enable_latency_funcs:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            s = time.time()
            result = func(*args, **kwargs)
            l = time.time() - s
            
            if name in latency_measurements:
                latency_measurements[name].append(l)
            else:
                latency_measurements[name] = [l]
                
            return result
        return wrapper
    return decorator

LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "id": "Indonesian",
    "hi": "Hindi",
    "ms": "Malay",
    "tl": "Tagalog",
    "vi": "Vietnamese",
    "th": "Thai",

    # These are all supported by whisper but not used for our product
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "fi": "Finnish",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
}

