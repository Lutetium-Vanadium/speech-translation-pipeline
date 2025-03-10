from tqdm import tqdm
from asr import create_args, Runner
from datasets import Dataset
from common import STORAGE_DIR_REDUCED_FLEURS, STORAGE_DIR_RESULTS, latency_measurements
from os import path
from pathlib import Path
from statistics import mean
import numpy as np

import time
import json

from main import CascadePipeline

args, unknown = create_args().parse_known_args()
name = unknown[0]
args.log_level = 'INFO'
args.file = './jfk.wav'
args.vac = True

output_folder = Path(STORAGE_DIR_RESULTS) / 'latency'
output_folder.mkdir(exist_ok=True)

# Define models
device = "cuda"
lang = 'zh'
lang_codes = ['en', 'zh']

pipeline = CascadePipeline(['en', lang], translation_setting=2, device=device)
runner = Runner(pipeline)

runner.init(args)

print('Warming up the models')
runner.run()
pipeline.finish()

print(f'### TESTING {name} ###')
dataset = Dataset.load_from_disk(path.join(STORAGE_DIR_REDUCED_FLEURS, lang))

LATENCY_KEYS = [
    'asr',
    'mt',
    'pipeline',
]

results = {
    k: [] for k in LATENCY_KEYS
}
results['final_delay'] = []

output_path = output_folder / f"{name}.json"

if not output_path.exists():
    for sample in tqdm(dataset):
        for src_lang in lang_codes:
            for tgt_lang in lang_codes:
                if src_lang == tgt_lang:
                    continue

                args.file = path.join(STORAGE_DIR_REDUCED_FLEURS, sample[f'{src_lang}_audio_path'])
                runner.init(args)
                runner.run()
                pipeline.finish()
                results['final_delay'].append(time.time() - runner.start)

                for k in LATENCY_KEYS:
                    results[k].append(mean(latency_measurements.get(k, [float('nan')])))
                latency_measurements.clear()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
else:
    with open(output_path, "r") as f:
        results = json.load(f)

results_summary = {}
try:
    with open(
        f"{output_folder}/summary.json", "r", encoding="utf-8"
    ) as f:
        results_summary = json.load(f)
except:
    pass

results_summary[name] = {}
for k, v in results.items():
    results_summary[name][k] = np.nanmean(v)

with open(f"{output_folder}/summary.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=4)

