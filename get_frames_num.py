import numpy as np
import os
import pickle
COMP_PATH = '/data1/Parker/Tasty_dir/'

tasty_path = os.path.join(COMP_PATH, 'DATASET_tasty')


def find_max_frames(tasty_path):
    frames_dict = {}

    print('Reading the test dataset')
    name_recs = os.listdir(tasty_path)
    for ite in range(len(name_recs)):
        curr_rec_loc = tasty_path + '/' + name_recs[ite]

        with open(curr_rec_loc, 'rb') as f:
            current_recipe = pickle.load(f)


        c_frame_indices = current_recipe['frame_indices']

        flat_list = [int(item) for sublist in c_frame_indices for item in sublist]
        frames_dict[current_recipe['title']] = [flat_list[0], flat_list[-1]]
    return frames_dict

with open('frame_range.pkl', 'wb') as f:
    pickle.dump(find_max_frames(tasty_path), f)

with open('frame_range.pkl', 'rb') as f:
    print(pickle.load(f))