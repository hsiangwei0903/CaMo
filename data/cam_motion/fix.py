# import json

# # Load the JSON file
# with open("updated_captionset_2000_.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Iterate through all messages and update user content
# for item in data:
#     for message in item.get("messages", []):
#         if message.get("role") == "user":
#             message["content"] = "<video>" + message["content"]

# # Save the updated JSON to a new file
# with open("updated_captionset_2000.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

import json

# Load the JSON file
with open("camerabench_10K.json", "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

for sample in data:
    assert len(sample["messages"]) == 2
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"
    caption = sample["messages"][1]["content"]
    if "<camera>" not in caption:
        print(caption)
        count += 1

print(len(data))
print(count)