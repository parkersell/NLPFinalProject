# -*- coding: utf-8 -*-

import os
import pickle
import re
import xml.etree.ElementTree as ET

import nltk
import numpy as np


def read_ingredients(cat_root):
    cats = cat_root.findall('ingredients')
    assert (cats != None)

    categories = cats[0].findall('ingredient')
    assert (categories != None)

    data_dict = []
    for cat in categories:
        curr_ings = cat.text
        if not curr_ings is None:
            tokens = nltk.tokenize.word_tokenize(curr_ings)
            reference_s = ' '.join(tokens).lower()
            reference_s = re.sub('[^A-Za-z]+', ' ', reference_s)
            reference_s = reference_s.replace('jalape', 'jalapeno')
            reference_s = re.sub(r"\b[a-zA-Z]\b", "", reference_s)
            reference_s = ' '.join(reference_s.split())
            data_dict.append(reference_s)
    return data_dict


def read_steps(cat_root):
    cats = cat_root.findall('steps_updated')
    assert (cats != None)

    categories = cats[0].findall('steps_updated')
    assert (categories != None)

    data_dict = []
    for cat in categories:
        data_dict.append(cat.text)
    return data_dict


def return_all_words(vocab_ing_dictionary):
    all_voc_words_v = ''
    key_dictionary = list(vocab_ing_dictionary.keys())
    for biki in range(len(key_dictionary)):
        curr_ins = vocab_ing_dictionary[key_dictionary[biki]]
        curr_ins = curr_ins.replace('_', ' ').lower()
        all_voc_words_v = all_voc_words_v + ' ' + curr_ins
    all_voc_words_v = all_voc_words_v.split()

    all_voc_words_v_us = ''
    key_dictionary = list(vocab_ing_dictionary.keys())
    for biki in range(len(key_dictionary)):
        curr_ins = vocab_ing_dictionary[key_dictionary[biki]]
        curr_ins = curr_ins.lower()
        all_voc_words_v_us = all_voc_words_v_us + ' ' + curr_ins
    all_voc_words_v_us = all_voc_words_v_us.split()

    return all_voc_words_v, all_voc_words_v_us


def read_label_frames(len_recipes, curr_recipe):
    dat_file_loc = VIDEO_PATH + curr_recipe + '/' + '/csvalignment.dat'
    list_annotations = [line.rstrip('\n') for line in open(dat_file_loc)]
    if not (len_recipes == len(list_annotations)):
        print('ERROR len_recipes == len(list_annotations)', curr_recipe)

    det_path_save_name = FEAT_PATH + '/' + curr_recipe + '.pkl'
    with open(det_path_save_name, 'rb') as f:
        features_all = pickle.load(f)

    frame_names = []
    for xi in range(len(features_all)):
        frame_names.append("{:05d}".format(xi) + '.jpg')
    # frame_names_features = [*features_all]
    # print(set(frame_names).difference(set(frame_names)))

    frame_labels = -1 * np.ones(len(features_all))
    for xi in range(len_recipes):
        if list_annotations[xi] == '-1,-1':
            continue

        annot_line = list_annotations[xi].split(',')
        start_frame = int(annot_line[0])
        end_frame = int(annot_line[1]) + 1

        start_frame_actual = start_frame * 5 - 1
        if start_frame == 1:
            start_frame_actual = 1
        end_frame_actual = end_frame * 5 - 1
        # end_frame_actual = min( ( end_frame + 1) * 5 - 1, len(features_all))

        frame_labels[start_frame_actual: end_frame_actual] = xi

    if not (len(frame_labels) == len(frame_names)):
        print('ERROR len(frame_labels) == len(frame_names)', curr_recipe)

    return frame_names, frame_labels


if __name__ == "__main__":
    COMP_PATH = "../"

    VIDEO_PATH = COMP_PATH + "TastyVideos/"  # COMP_PATH + "TastyVideos/"
    DATA_PATH = COMP_PATH + "TASTY_dir/DATA/"
    FEAT_PATH = COMP_PATH + "TASTY_dir/FEATURES/"

    # 1. read vocab
    vocab_fd = DATA_PATH + "vocab/"
    vocab_ing_dict = pickle.load(open(vocab_fd + 'vocab_ing_3769_dict.pkl', "rb"))
    key_dict = list(vocab_ing_dict.keys())
    val_dict = list(vocab_ing_dict.values())
    all_voc_word_post, all_voc_words_USs = return_all_words(vocab_ing_dict)

    # read splits
    split_path = DATA_PATH + "TASTY/"
    train_lines = [train_lines.rstrip('\n') for train_lines in open(split_path + '/TRAIN_4022.txt', 'r')]
    test_lines = [test_lines.rstrip('\n') for test_lines in open(split_path + '/TEST_4022.txt', 'r')]

    # recipe names
    name_recs = os.listdir(VIDEO_PATH)
    recipe_names = []
    for aa in range(len(name_recs)):
        curr_rec = name_recs[aa].replace('\n', '')
        recipe_names.append(curr_rec)
    recipe_names.sort()

    for i in range(0, len(recipe_names)):
        print(' [*] ', i, ' ', recipe_names[i])
        rec_f = VIDEO_PATH + recipe_names[i] + '/' + "recipe.xml"

        # 1 ****************************
        tree = ET.parse(rec_f)
        cat_root = tree.getroot()
        ingredient_dict = read_ingredients(cat_root)

        # 3 ****************************
        steps_dict = read_steps(cat_root)
        if len(steps_dict) == 0:
            print('ERROR')
            break
        for bi in range(len(steps_dict)):
            curr_sent = steps_dict[bi]
            steps_dict[bi] = curr_sent

        # 4 ****************************
        split = ''
        if recipe_names[i] in train_lines:
            split = 'train'
        elif recipe_names[i] in test_lines:
            split = 'test'
        else:
            print('ERROR')
            break

        # 5 ****************************
        _, frame_labels = read_label_frames(len(steps_dict), recipe_names[i])
        current_unique = list(set(frame_labels))
        current_unique.sort()
        frame_indices = []
        recipe_steps = []
        for kkl in range(len(steps_dict)):
            curr_steps = steps_dict[kkl]
            if (kkl in current_unique) or (steps_dict[kkl] == 'Enjoy!'):
                if kkl in current_unique:
                    aa = np.argwhere(frame_labels == kkl)
                else:
                    aa = []
                frame_indices.append(aa)
                recipe_steps.append(steps_dict[kkl])

        # 6 ****************************
        recipe_instance = dict()
        recipe_instance['title'] = recipe_names[i]
        recipe_instance['ingredients'] = ingredient_dict
        recipe_instance['split'] = split
        recipe_instance['recipe_steps'] = recipe_steps
        recipe_instance['frame_indices'] = frame_indices
