import json

input_file = 'keypoints_test2017_results_epoch-1.json'
output_file = 'keypoints_test2017_results_epoch-1_pruned.json'

with open(input_file) as f:
	data = json.load(f)


new_data = []

keep_keys = ['image_id', 'category_id', 'keypoints', 'score']

for i, sample in enumerate(data):
	new_sample = {}

	for key in sample.keys():
		if key in keep_keys:
			new_sample[key] = sample[key]

	new_data.append(new_sample)
	print(i, len(data))

with open(output_file, 'w') as f:
	json.dump(new_data, f)