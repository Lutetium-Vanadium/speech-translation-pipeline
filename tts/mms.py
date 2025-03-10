import torch
import gc
from transformers import VitsTokenizer, VitsModel
from common import STORAGE_DIR_MODEL
from pypinyin import lazy_pinyin, Style


model_list = {
    "en": "facebook/mms-tts-eng",
    "zh": "BricksDisplay/vits-cmn",
    "id": "facebook/mms-tts-ind",
    "hi": "facebook/mms-tts-hin",
    "ms": "facebook/mms-tts-zlm",
    "tl": "facebook/mms-tts-tgl",
    "vi": "facebook/mms-tts-vie",
    "th": "facebook/mms-tts-tha",
}


class MmsTts:
    def __init__(self, lang: str = "en", device: str = "cpu"):
        if lang not in model_list:
            raise Exception("Unknown language code: " + lang)
        self.model_id = model_list[lang]
        self.tokenizer = None
        self.model = None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        self.tokenizer = VitsTokenizer.from_pretrained(self.model_id, cache_dir=STORAGE_DIR_MODEL + '/tts')
        self.model = VitsModel.from_pretrained(
            self.model_id,
            cache_dir=STORAGE_DIR_MODEL + '/tts'
        ).to(self.device).eval()
        return self.tokenizer, self.model

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def synthesize(self, text: str):
        if self.tokenizer is None or self.model is None:
            self.load_model()

        if self.model_id.endswith('cmn'):
            text = ''.join(lazy_pinyin(text, style=Style.TONE, tone_sandhi=True))

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            outputs = self.model(inputs['input_ids'], speaker_id=17)
            return outputs.waveform[0].cpu().numpy()
    
