import json

# with open('camerabench_10K.json', 'r') as file:
#     data = json.load(file)

with open('../spld_spatial_data.json', 'r') as file:
    data = json.load(file)

for item in data:
    cont1_length = len(item["messages"][0]["content"])
    cont2_length = len(item["messages"][1]["content"])
    if cont1_length == 0 or cont2_length == 0:
        print(item)
        import pdb; pdb.set_trace()