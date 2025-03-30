from functools import cache
from tqdm import tqdm
from asr import create_args, Runner
from datasets import Dataset
from common import STORAGE_DIR_CONVERSATION_DATA, STORAGE_DIR_RESULTS
from normalizers import BasicTextNormalizer, EnglishTextNormalizer
from os import path
from pathlib import Path
from itertools import product

import json
import gc
import glob
import evaluate

from main import CascadePipeline

"""
To run predictions on batches using different models. 

Predictions are stored, and batches are concatenated at the end.
"""

def detect_chinese(text):
    has_chinese = any(
        "\u4E00" <= char <= "\u9FFF" or "\u3400" <= char <= "\u4DBF" for char in text
    )
    return has_chinese


def add_spaces_in_chinese(prediction):
    reconstruct = ""
    for i, char in enumerate(prediction):
        if detect_chinese(char):
            if i != len(predictions) - 1:
                reconstruct += char + " "
            else:
                reconstruct += char
        else:
            reconstruct += char

    return reconstruct

@cache
def get_normalizers(lang):
    normalizers = [
        BasicTextNormalizer() if lang != "en" else EnglishTextNormalizer()
    ]

    if lang == "zh":
        normalizers.append(add_spaces_in_chinese)

    return normalizers

# Define batch size
batch_size = 25

output_folder = Path(STORAGE_DIR_RESULTS) / 'conversation'
output_folder.mkdir(exist_ok=True)

# Define models
device = "cuda"

test_langs = [
    'zh',
    'id',
    'hi',
    'ms',
    'vi',
    'th',
    'tl',
]

translation_settings = [2]

configs = list(product(test_langs, translation_settings))

pipeline = CascadePipeline(['en', 'zh'], device=device)
runner = Runner(pipeline)

args = create_args().parse_args()
args.log_level = 'INFO'
args.vac = True

runner.init(args)

evaluation_results = {
}

for lang, translation_setting  in configs:
    print('### TESTING LANG:', lang, '###')

    pipeline.translation_setting = translation_setting
    pipeline.set_languages(['en', lang])
    
    gc.collect()

    dataset = Dataset.load_from_disk(path.join(STORAGE_DIR_CONVERSATION_DATA, lang))

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_output_file_path = output_folder / f'{lang}-{translation_setting}_batch_{i}.json'

        # To skip already processed batches
        if batch_output_file_path.exists():
            continue

        batch = dataset.select(
            list(range(i, min(i + batch_size, len(dataset))))
        )

        # Predict for batch
        predictions = []
        for sample in tqdm(batch, leave=False):
            args.file = path.join(STORAGE_DIR_CONVERSATION_DATA, sample[f'path'])
            runner.init(args)
            
            error = ''
            try:
                runner.run()
                pipeline.finish()
            except Exception as e:
                error = str(e)

            truth = ""
            pred = ""

            for text, tlang in sample['translation']:
                for n in get_normalizers(tlang):
                    text = n(text)
                truth += text + '\n'

            for text, tlang in pipeline.translation_history:
                for n in get_normalizers(tlang):
                    text = n(text)
                pred += text + '\n'

            predictions.append({
                "id": sample["id"],
                "ground_truth": truth.strip(),
                "raw_text": sample['transcription'],
                "raw_ground_truth": sample['translation'],
                "prediction": pred.strip(),
                "raw_transcription": pipeline.transcription_history,
                "raw_prediction": pipeline.translation_history,
                "error": error,
            })

        with open(batch_output_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)

del pipeline
del runner

# Define evaluation metrics
wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
bert_metric = evaluate.load("bertscore")

evaluation_metrics = [wer_metric, bleu_metric, chrf_metric, bert_metric]


print('Evaluating results...')
for lang, translation_setting  in configs:
    # Combine the batch outputs into a single file
    batch_output_file_paths = glob.glob(
        str(output_folder / f"{lang}-{translation_setting}_batch_*.json")
    )

    all_predictions = []
    for file in batch_output_file_paths:
        with open(file, "r", encoding="utf-8") as f:
            batch_predictions = json.load(f)
            all_predictions.extend(batch_predictions)

    with open(
        f"{output_folder}/{lang}-{translation_setting}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_predictions, f, indent=4)

    eval_key = f'{lang}_{translation_setting}'

    # Evaluate the predictions
    evaluation_results[eval_key] = {
        "translation_setting": translation_setting,
    }
    for metric in evaluation_metrics:
        predictions = [prediction["prediction"] for prediction in all_predictions]
        references = [prediction["ground_truth"] for prediction in all_predictions]

        kwargs = {}
        if metric == bleu_metric:
            kwargs['use_effective_order'] = True
        elif metric == chrf_metric:
           kwargs['word_order'] = 2
        elif metric == bert_metric:
            kwargs['model_type'] = "xlm-roberta-large"

        results = metric.compute(predictions=predictions, references=references, **kwargs)

        # Store results in the dictionary under the corresponding language and metric
        evaluation_results[eval_key][metric.name] = results

# Write the results to a JSON file
output_file_path = (
    f"{output_folder}/evaluation_results.json"
)
with open(output_file_path, "w") as output_file:
    json.dump(evaluation_results, output_file, indent=4)
