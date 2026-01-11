import os 
import os.path as osp
import json

CAPTION_PATH = "results/tiny/exp1_SG_16.json"
EVAL_PATH = "results/tiny/exp1_SG_16_eval.json"
OUT_PATH = "user_prediction.json"

def main():
    # 1) load caption dict: { "xxxx.mp4": ["cap1", "cap2", ...], ... }
    with open(CAPTION_PATH, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # load eval list of dict items
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            user_preds = json.load(f)
        print(f"Loaded existing {len(user_preds)} user predictions from {OUT_PATH}")
    else:
        user_preds = []

    target = [1485, 1531, 1571, 2440, 3016, 3133, 3180, 4145, 4198, 4243, 4381, 4990, 5103, 5139]
    target = [str(t) for t in target]

    # 1. sort by id
    items = sorted(items, key=lambda x: x.get("id", -1))
    items = [item for i, item in enumerate(items) if i%4==0]
    items = [item for item in items if str(item["id"]) in target]

    user_preds = [item for item in user_preds if str(item["id"]) not in target]

    for i, item in enumerate(items):
        # 2. continue if i!=0 or i%2==0  (hard-coded as you wrote)
        # This condition keeps ONLY i==0 and odd i (1,3,5,...) and skips even i>0.
        
        qid = str(item["id"])

        scene = item["scene_name"]
        video_key = f"{scene}.mp4"

        # 3. find corresponding caption
        cap_list = captions.get(video_key, [])

        # 4. print i, question, options, caption
        print("\n" + "=" * 80)
        print(f"i = {i}/{len(items)} | id = {item.get('id')} | scene = {scene}")
        print("Question:")
        print(item.get("question", ""))
        print("\nOptions:")
        print(item.get("options", ""))
        # print("\nGround Truth Answer:")
        # print(item.get("ground_truth", ""))
        # print("\nOriginal Prediction:")
        # print(item.get("predicted", ""))
        print("\nCaption:")
        for start_frame_id, c in enumerate(cap_list):
            print(f"[Frame {start_frame_id*16} to Frame {(start_frame_id+1)*16 - 1}]:")
            print(f"{c}")
            print("=" * 40)

        # 5. user input
        pred = input("\nYour prediction (e.g., A/B/C/D, or full 'A. ...'): ").strip()

        # copy item and overwrite predicted (and optionally correct)
        new_item = dict(item)
        new_item["predicted"] = pred
        if "ground_truth" in new_item:
            new_item["correct"] = (str(pred).strip().upper() == str(new_item["ground_truth"]).strip().upper())

        user_preds.append(new_item)

        # 6. save final list
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(user_preds, f, ensure_ascii=False, indent=2)

        print(f"\nSaved {len(user_preds)} items to {OUT_PATH}")

if __name__ == "__main__":
    main()
