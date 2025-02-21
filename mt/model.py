import gc
import time
from common import logger, STORAGE_DIR_MODEL

from transformers import (
    AutoModelForSeq2SeqLM,
)
from ctranslate2 import Translator
from .tokenizer import NllbTokenizer

class Nllb200:
    def __init__(self, device: str = "auto"):
        self.model_id = "facebook/nllb-200-distilled-600M"
        self.tokenizers: NllbTokenizer= None
        self.model: Translator= None
        self.device = device

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        t = time.time()
        if self.model is not None and self.tokenizers is not None:
            return self.tokenizers, self.model
        logger.info(f'Loading Nllb200 model ({self.get_model_name()})...')
        self.tokenizers = NllbTokenizer(self.model_id, cache_dir=STORAGE_DIR_MODEL + '/nllb')
        self.model = Translator(
            STORAGE_DIR_MODEL + '/nllb-ctranslate-2',
            device=self.device,
            # RuntimeError: Flash attention 2 is not supported
            # flash_attention=True,
            compute_type="int8_float16",
        )
        self.model.load_model()
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

        return self.tokenizers, self.model

    def unload(self):
        if self.model is not None:
            del self.model
        if self.tokenizers is not None:
            del self.tokenizers
        # Flush the current model from memory
        gc.collect()

    def translate(self, text: str, source: str = "en", target: str = "en"):
        if len(text) == 0:
            return ''

        if self.tokenizers is None or self.model is None:
            self.load_model()

        tokenizer = self.tokenizers.get_tokenizer(source)

        # TODO: consider max length?
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

        target_prefix = [self.tokenizers.get_lang_id(target)]
        results = self.model.translate_batch([source], target_prefix=[target_prefix])
        translated_text = results[0].hypotheses[0][1:]

        return tokenizer.decode(tokenizer.convert_tokens_to_ids(translated_text))



class Nllb200Torch:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        import torch
        self.model_id = model_id
        self.tokenizers: NllbTokenizer= None
        self.model: AutoModelForSeq2SeqLM= None
        self.device = torch.device(device)

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def load_model(self):
        import torch
        t = time.time()
        if self.model is not None and self.tokenizers is not None:
            return self.tokenizers, self.model
        logger.info(f'Loading Nllb200 model ({self.get_model_name()})...')
        self.tokenizers = NllbTokenizer(self.model_id, cache_dir=STORAGE_DIR_MODEL + '/nllb')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            cache_dir=STORAGE_DIR_MODEL + '/nllb',
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        ).to(self.device).eval()
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

        return self.tokenizers, self.model

    def unload(self):
        import torch
        if self.model is not None:
            del self.model
        if self.tokenizers is not None:
            del self.tokenizers
        # Flush the current model from memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def translate(self, text: str, source: str = "en", target: str = "en", max_length=1000):
        import torch
        if len(text) == 0:
            return ''

        if self.tokenizers is None or self.model is None:
            self.load_model()

        tokenizer = self.tokenizers.get_tokenizer(source)

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)

            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(self.tokenizers.get_lang_id(target)),
                max_length=max_length
            )
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translation
