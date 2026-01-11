import json

with open('captionset.json', 'r') as file:
    all_data = json.load(file)

all_vid = set([item['videos'][0] for item in all_data])

with open('updated_captionset_gemini.json', 'r') as file:
    data = json.load(file)

vid_to_caption = {vid: None for vid in all_vid}

for item in data:
    video_name = item['videos'][0]
    caption = item['messages'][1]['content']
    vid_to_caption[video_name] = caption

with open('vid_to_caption.json', 'w') as file:
    json.dump(vid_to_caption, file, indent=4)

print(len(all_data))
print(len(vid_to_caption))