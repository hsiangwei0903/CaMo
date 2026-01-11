import json
import os
import os.path as op
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from collections import defaultdict
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class model_pipeline:
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, 
                attn_implementation="flash_attention_2",
                torch_dtype="bfloat16").eval().to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to("cuda")

        ids = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            use_cache=True
        )

        trimmed = ids[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return out

def get_video_paths(data_path, subset=True, mcq=True):
    video_paths = []
    jsonl_file = 'test_tiny.jsonl' if subset else 'test.jsonl'
    with open(op.join(data_path, jsonl_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample = json.loads(line)
            dataset = sample['dataset']
            scene_name = sample['scene_name']
            if mcq and sample["options"] is None:
                continue
            video_paths.append(op.join(data_path, dataset, scene_name + '.mp4'))
    video_paths = list(set(video_paths))
    video_paths.sort()
    return video_paths

def load_video_frames(video_path, fps, target_pixels=512*28*28):
    """Use decord to read video frames and return timestamps of those frames."""
    
    def calculate_target_size(original_width, original_height, target_pixels):
        """Calculate target size maintaining aspect ratio based on target pixels."""
        aspect_ratio = original_width / original_height
        
        target_height = int((target_pixels / aspect_ratio) ** 0.5)
        target_width = int((target_pixels * aspect_ratio) ** 0.5)
        
        return target_width, target_height
    
    def resize_image_to_target_pixels(image, target_pixels):
        """Resize image maintaining aspect ratio to approximate target pixels."""
        original_width, original_height = image.size
        target_width, target_height = calculate_target_size(original_width, original_height, target_pixels)
        
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        video_duration = total_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else total_frames / 30  # Estimate duration
        video_duration = int(video_duration)
        target_frames = int(video_duration * fps)
        
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        
        frames_pil = [resize_image_to_target_pixels(Image.fromarray(f), target_pixels) for f in frames_np]
        
        timestamps = [int(idx / vr.get_avg_fps()) for idx in frame_indices] if vr.get_avg_fps() > 0 else [int(idx / 30) for idx in frame_indices]  # Get integer timestamps
        
        return frames_pil, timestamps, video_duration, total_frames
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return None, None, None


def get_summary(results):

    qtype_correct = defaultdict(int)
    qtype_total = defaultdict(int)

    for sample in results:
        correct = sample['correct']
        question_type = sample['question_type']

        qtype_total[question_type] += 1
        if correct:
            qtype_correct[question_type] += 1

    summary = {}
    overall_correct = 0
    overall_total = 0

    # collect object_rel_direction stats
    obj_rel_dir_correct = 0
    obj_rel_dir_total = 0

    for qtype in qtype_total:
        correct = qtype_correct[qtype]
        total = qtype_total[qtype]
        acc = correct / total if total > 0 else 0.0

        summary[qtype] = {
            "correct": correct,
            "total": total,
            "accuracy": acc
        }

        overall_correct += correct
        overall_total += total

        if qtype.startswith("object_rel_direction"):
            obj_rel_dir_correct += correct
            obj_rel_dir_total += total

    # add aggregated object_rel_direction accuracy
    summary["object_rel_direction"] = {
        "correct": obj_rel_dir_correct,
        "total": obj_rel_dir_total,
        "accuracy": (
            obj_rel_dir_correct / obj_rel_dir_total
            if obj_rel_dir_total > 0 else 0.0
        )
    }

    # add overall accuracy
    summary["overall"] = {
        "correct": overall_correct,
        "total": overall_total,
        "accuracy": (
            overall_correct / overall_total
            if overall_total > 0 else 0.0
        )
    }
    
    return summary