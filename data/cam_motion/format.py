import json
import os
import random
from tqdm import tqdm

random.seed(42)

INPUT_FILE = 'vid_to_caption.json'
OUTPUT_FILE = 'formatted_camera_captions.json'
TEMPLATE_FILE = 'template.txt'

with open(TEMPLATE_FILE, 'r') as f:
    templates = [line.strip() for line in f if line.strip()]

# 2. Load Source JSON
with open(INPUT_FILE, 'r') as f:
    source_data = json.load(f)

output_data = []
keys = list(source_data.keys())

# 3. Process Data tqdm
for i, video_path in enumerate(tqdm(keys)):

    raw_text = source_data[video_path]
    if not raw_text:
        continue

    # random six questions
    questions = random.sample(templates, 7)

    for question in questions:
        # Build entry
        entry = {
            "messages": [
                {
                    "content": f"<video>{question}",
                    "role": "user"
                },
                {
                    "content": raw_text,
                    "role": "assistant"
                }
            ],
            "videos": [video_path]
        }
        
        output_data.append(entry)

# shuffle the output data
random.shuffle(output_data)

# 4. Save Result
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Done! Processed {len(output_data)} videos. Saved to {OUTPUT_FILE}")