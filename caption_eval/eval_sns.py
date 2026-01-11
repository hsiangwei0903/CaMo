import os 
import os.path as osp
import json
import argparse
from evaluator import GeminiTextEvaluator
from tqdm import tqdm
from collections import defaultdict
from utils import get_summary

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, required=True, help='Path to the results JSON file')
args = parser.parse_args()

results_path = args.results_path

output_path = results_path.replace(".json", "_eval.json")
summary_path = results_path.replace(".json", "_eval_summary.json")

API_KEY = os.environ.get("gemini_api_key")

with open('test_tiny.jsonl') as f:
    test_data = [json.loads(line) for line in f]

mc_test_data = [sample for sample in test_data if sample['options'] is not None]
mc_test_data.sort(key=lambda x: x['id'])

with open(results_path) as f:
    captions = json.load(f)

with open(results_path.replace(".json", "_metadata.json")) as f:
    metadata = json.load(f)
    segment_length = metadata['segment_length']

if osp.exists(output_path):
    with open(output_path) as f:
        results = json.load(f)
    results = [res for res in results if res['predicted'] is not None]
    print(f"Loaded {len(results)} previously evaluated results.")
else:
    results = []

evaluated_ids = set([result['id'] for result in results])

evaluator = GeminiTextEvaluator(API_KEY)

for sample in tqdm(mc_test_data):
    qid = sample['id']
    if qid in evaluated_ids:
        continue
    evaluated_ids.add(qid)
    dataset = sample['dataset']
    scene_name = sample['scene_name']
    question = sample['question']
    options = str(sample['options'])
    question_type = sample['question_type']
    answer = sample['ground_truth']

    test_cap = captions[scene_name+'.mp4']
    input_cap = evaluator.format_clip_captions(test_cap, segment_length=segment_length)

    raw_output = evaluator.evaluate(input_cap, question, options)
    final_answer = evaluator.fetch_answer(raw_output)

    results.append({
        'id': qid,
        'scene_name': scene_name,
        'question': question,
        'options': options,
        'question_type': question_type,
        'ground_truth': answer,
        'predicted': final_answer,
        'correct': final_answer == answer
    })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

summary = get_summary(results)

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)

print(summary)