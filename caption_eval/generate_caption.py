import os
import os.path as osp
import json
from tqdm import tqdm
from utils import load_video_frames, get_video_paths, model_pipeline

MODEL_PATH = "../saves/qwen2_5vl-3b/full/camo"
DATA_PATH = "../eval_spld/datasets/VSI-Bench/"
USE_TAG = False
USE_SUBSET = False
FPS = 2
SEGMENT_LENGTH = 16
QUESTION = "Describe what is happening in the video and how the camera moves."
TEMPLATE = f"Use <scene> for the content and <camera> for the camera motion." if USE_TAG else ""
PROMPT = f"{QUESTION} {TEMPLATE}"
OUTPUT_PATH = f"./results/{osp.basename(MODEL_PATH)}_SG_{SEGMENT_LENGTH}.json"
OUTPUT_META_DATA_PATH = f"./results/{osp.basename(MODEL_PATH)}_SG_{SEGMENT_LENGTH}_metadata.json"

os.makedirs(osp.dirname(OUTPUT_PATH), exist_ok=True)

if osp.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {OUTPUT_PATH}")
else:
    results = {}

processed_videos = set(results.keys())

meta_data = {
    "model_path": MODEL_PATH,
    "data_path": DATA_PATH,
    "use_tag": USE_TAG,
    "fps": FPS,
    "segment_length": SEGMENT_LENGTH,
    "prompt": PROMPT
}

with open(OUTPUT_META_DATA_PATH, "w") as f:
    json.dump(meta_data, f, indent=4)

video_paths = get_video_paths(DATA_PATH, subset=USE_SUBSET, mcq=True)

pipeline = model_pipeline(MODEL_PATH)

for vid, video_path in tqdm(enumerate(video_paths), total=len(video_paths)):

    print(f"Processing {video_path}, {vid+1}/{len(video_paths)}")

    video_name = osp.basename(video_path)

    if video_name in processed_videos:
        print(f"Skipping {video_name} as it has already been processed")
        continue

    frames, timestamps, video_duration, total_frames = load_video_frames(video_path, fps=FPS)

    video_results = []

    if not frames:
        continue

    for i in range(0, len(frames), SEGMENT_LENGTH):

        frames_segment = frames[i:i+SEGMENT_LENGTH]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_segment
                    },
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]

        out = pipeline(messages)

        video_results.append(out)

    results[video_name] = video_results

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)