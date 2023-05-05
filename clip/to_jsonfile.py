import os
import pickle
import json

base_frame_path = '../TastyFrames/'
base_recipe_path = '../TASTY_dir/DATASET_tasty/'
all_frames = os.listdir(base_frame_path)
print(all_frames)

json_output = {'images':[]}
id=1
for video in all_frames:
    video_frames_path = base_frame_path + video + '/frames'
    # print(video_frames_path)
    recipe_path = base_recipe_path + video + '.pkl'
    try: 
        with open(recipe_path, 'rb') as f: 
            current_recipe = pickle.load(f)
    except Exception as e:
        print(e)
        continue
    c_ingredients = current_recipe['ingredients']
    c_sentences = current_recipe['recipe_steps']
    allings = current_recipe['ingredients_names']
    c_frame_indices = current_recipe['frame_indices']
    if len(c_frame_indices) != len(c_sentences):
        print(c_sentences, c_frame_indices)
        break

    for i, frame_i in enumerate(c_frame_indices):
        for count in frame_i:
            json_output['images'].append({'file_path': video_frames_path + '/{:05d}.jpg'.format(int(count)-1), 
                                'captions':[c_sentences[i]], 'id': id}) # captions are not align correctly 
            id+=1
print(len(json_output['images']))

with open('clip/jsonAll.json', 'w') as f:
    f.write(json.dumps(json_output))