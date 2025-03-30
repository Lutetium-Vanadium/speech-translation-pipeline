from asr import Runner, logger, TranscriptHandler, create_args
from asr import create_tokenizer, VACOnlineASRProcessor, OnlineASRProcessor
from uart import Screen
from audio import AudioCapture, output_audio
from common import measure_latency
from zhconv import convert
import os

from mt import Nllb200
from tts import Tts

def process_transcription_text(text: str, lang: str):
    if lang == 'zh':
         text = convert(text, 'zh-hans')
    return text


class CascadePipeline(TranscriptHandler):
    """
    Translation settings is an enum:
    == 0: translate only confirmed transcription as it comes
    == 1: translate per sentence, only translate confirmed transcription
    == 2: translate per sentence, translate confirmed and unconfirmed transcription

    Default is 2
    """
    def __init__(self, languages: list[str], translation_setting = 2, device='cuda', enable_tts=False):
        super().__init__()
        self.languages = languages
        self.mt_model = Nllb200(device=device)
        self.tts_model = None
        if enable_tts:
            self.tts_model = Tts(device=device)
        self.translation_setting = translation_setting

        self.tokenizer = None
        if translation_setting != 0:
            self.tokenizer = create_tokenizer()

        self.last_transcribed_lang = None
        self.last_confirmed_transcription_timestamp = -1
        self.transcription_history = []
        self.translation_history = []
        self.confirmed_transcription = ''
        self.tts_spoken = 0
        self.unconfirmed_transcription = ''
        self.last_transcribed_sentence = ''
        self.confirmed_translation = ''
        self.unconfirmed_translation = ''
        self.screen = Screen()

    def reset(self):
        self.last_transcribed_lang = None
        self.transcription_history = []
        self.translation_history = []
        self.confirmed_transcription = ''
        self.unconfirmed_transcription = ''
        self.last_transcribed_sentence = ''
        self.confirmed_translation = ''
        self.tts_spoken = 0
        self.unconfirmed_translation = ''
        if self.asr is not None:
            self.asr.reset()

    def init(self, asr: OnlineASRProcessor):
        super().init(asr)
        self.mt_model.load_model()

        self.set_languages(self.languages)

    def set_languages(self, languages: list[str]):
        if len(language) != 2:
            raise "bad number of languages"
        self.languages = languages
        self.asr.set_languages(languages)

        if self.tts_model is not None:
            for l in self.languages:
                self.tts_model.load_lang(l)

        self.reset()

    def process_transcribed(self, transcript, src: str, src_speakerid: int):
        if src != self.last_transcribed_lang and self.last_transcribed_lang is not None:
            self.transcription_history.append((self.confirmed_transcription, self.last_transcribed_lang))
            self.confirmed_transcription = ''

        self.confirmed_transcription += transcript
        self.screen.send_text(transcript, src_speakerid, is_translation=False, is_confirmed=True)

        online = self.asr.online if isinstance(self.asr, VACOnlineASRProcessor) else self.asr
        buffer = online.transcript_buffer.buffer
        i = 0
        for _, end, _ in buffer:
            if end > self.last_confirmed_transcription_timestamp:
                break
            i += 1
        self.unconfirmed_transcription = process_transcription_text(online.to_flush(buffer[i:])[2], src)
        self.screen.send_text(self.unconfirmed_transcription, src_speakerid, is_translation=False, is_confirmed=False)

        logger.debug('ASR  CFM: ' + self.confirmed_transcription)
        logger.debug('ASR TODO: ' + self.unconfirmed_transcription)

    def process_final_translation_bit(self):
        if len(self.last_transcribed_sentence) == 0:
            return
        src, _src_speakerid, tgt, tgt_speakerid = self.get_speaker_data()
        cfm_translated = self.mt_model.translate(self.last_transcribed_sentence,source=src, target=tgt)
        self.confirmed_translation += cfm_translated
        self.screen.send_text(cfm_translated, tgt_speakerid, is_translation=True, is_confirmed=True)
        # Reset the translation context
        self.last_transcribed_sentence = ''
        logger.debug(' MT  CFM: ' + self.confirmed_translation)

    def process_translation(self, transcript: str, src: str, tgt: str, tgt_speakerid: int):
        if src != self.last_transcribed_lang and self.last_transcribed_lang is not None:
            # Language has changed, translate whatever was confirmed transcribed but not confirmed translated
            self.process_final_translation_bit() 
            self.translation_history.append((self.confirmed_translation, src))
            self.confirmed_translation = ''
            self.tts_spoken = 0

        if self.tokenizer is None:
            cfm_translated = self.mt_model.translate(transcript,source=src, target=tgt)
            self.confirmed_translation += cfm_translated
            self.screen.send_text(cfm_translated, tgt_speakerid, is_translation=True, is_confirmed=True)
            logger.debug(' MT  CFM: ' + self.confirmed_translation)
            return

        text = self.last_transcribed_sentence + transcript
        confirmed_len = len(text)

        if self.translation_setting == 2:
            text += self.unconfirmed_transcription

        sentences: list[str] = self.tokenizer.split(text, threshold=0.4)
        l = 0

        cfm_to_translate = ''
        uncfm_to_translate = ''
        self.last_transcribed_sentence = ''

        last_was_confirmed = True
        for i, s in enumerate(sentences):
            old_l = l
            l += len(s)
            if l <= confirmed_len and i+1 < len(sentences):
                cfm_to_translate += s
            else:
                if last_was_confirmed:
                    self.last_transcribed_sentence = s[:confirmed_len-old_l]
                uncfm_to_translate += s
                last_was_confirmed = False

        if len(cfm_to_translate) > 0:
            cfm_translated = self.mt_model.translate(cfm_to_translate,source=src, target=tgt)
            self.confirmed_translation += cfm_translated
            self.screen.send_text(cfm_translated, tgt_speakerid, is_translation=True, is_confirmed=True)

        if len(uncfm_to_translate) > 0:
            self.unconfirmed_translation = self.mt_model.translate(uncfm_to_translate,source=src, target=tgt)
            self.screen.send_text(self.unconfirmed_translation, tgt_speakerid, is_translation=True, is_confirmed=False)

        logger.debug(' MT TEXT: ' + text)
        logger.debug(' MT SENT: ' + str(sentences))
        logger.debug(' MT  CTR: ' + cfm_to_translate)
        logger.debug(' MT UCTR: ' + uncfm_to_translate)
        logger.debug(' MT  CFM: ' + self.confirmed_translation)
        logger.debug(' MT TODO: ' + self.unconfirmed_translation)
        logger.debug(' MT LAST: ' + self.last_transcribed_sentence)

    @measure_latency('pipeline')
    def handle(self, transcript: str, start_timestamp: float, end_timestamp: float, now: float):
        logger.debug('=====================')
        src = self.asr.get_last_language()
        transcript = process_transcription_text(transcript, src)
            
        try:
            src, src_speakerid, tgt, tgt_speakerid = self.get_speaker_data(src)

            if end_timestamp is not None:
                self.last_confirmed_transcription_timestamp = max(self.last_confirmed_transcription_timestamp, end_timestamp)
            self.process_transcribed(transcript, src, src_speakerid)
            self.process_translation(transcript, src, tgt, tgt_speakerid)

            self.last_transcribed_lang = src
        except ValueError:
            logger.debug(f"skipping different language {src}")
        logger.debug('=====================')

    def get_speaker_data(self, src=None):
        if src is None:
            src = self.last_transcribed_lang

        if src is None:
            raise ValueError('trying to get last speaker data when there is none')

        src_speakerid = self.languages.index(src)
        tgt_speakerid = 1 - src_speakerid
        tgt = self.languages[tgt_speakerid]
        return src, src_speakerid, tgt, tgt_speakerid

    def handle_silence(self):
        if self.tts_model is None:
            return
        if self.last_transcribed_lang is None:
            return
        self.process_final_translation_bit() 

        to_speak = self.confirmed_translation[self.tts_spoken:]

        if len(to_speak) == 0:
            return

        lang_idx = self.languages.index(self.last_transcribed_lang)
        tgt = self.languages[1-lang_idx]

        self.tts_spoken += len(to_speak)
        waveform = self.tts_model.synthesize(to_speak, tgt)
        output_audio(waveform)

    def finish(self):
        if self.last_transcribed_lang is None:
            return

        src, src_speakerid, tgt, tgt_speakerid = self.get_speaker_data()

        self.transcription_history.append([
            self.confirmed_transcription + self.unconfirmed_transcription, src,
        ])
        if len(self.unconfirmed_transcription) > 0:
            self.screen.send_text(self.unconfirmed_transcription, src_speakerid, is_translation=False, is_confirmed=True)


        self.translation_history.append([
            self.confirmed_translation + self.unconfirmed_translation, tgt
        ])
        if len(self.unconfirmed_translation) > 0:
            self.screen.send_text(self.unconfirmed_translation, tgt_speakerid, is_translation=True, is_confirmed=True)

if __name__ == '__main__':
    args = create_args().parse_args()
    is_audio_file = os.path.isfile(args.file or '')
    pipeline = CascadePipeline(languages=['en', 'zh'], enable_tts=is_audio_file==False)

    runner = Runner(pipeline, silence_time=4)
    runner.init(args)
    if is_audio_file:
        runner.run()
    else:
        audio_device = AudioCapture(output_audio=lambda a: runner.online.insert_audio_chunk(a))

        try:
            audio_device.start()
            print('listening...')
            while True:
                runner.process_iter()
        except KeyboardInterrupt:
            audio_device.stop()

    pipeline.finish()

    print(pipeline.transcription_history)
    print(pipeline.translation_history)


