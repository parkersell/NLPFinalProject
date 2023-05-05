# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import torch


class TastyVideos:
    def __init__(self, args=None):

        self.sentences, self.ingredients, self.frame_feats = dict(), dict(), dict()

        with open(args.vocab_ing, 'rb') as f:
            vocab_ing = pickle.load(f)
        lenVoc = len(vocab_ing)

        # recipe names
        name_recs = os.listdir(args.tasty_path)
        counter_rec = 0
        for ite in range(len(name_recs)):
            curr_rec_loc = args.tasty_path + '/' + name_recs[ite]
            curr_feat_loc = args.feat_path + '/' + name_recs[ite]

            with open(curr_rec_loc, 'rb') as f:
                current_recipe = pickle.load(f)

            if current_recipe['split'] == 'test':
                continue

            c_ingredients = current_recipe['ingredients']
            c_sentences = current_recipe['recipe_steps']
            c_frame_indices = current_recipe['frame_indices']

            c_ingredients = list(map(int, c_ingredients))
            c_ingredients = np.unique(c_ingredients)
            c_ingredients = c_ingredients[c_ingredients != 30170]  # remove the 'unk>' ingredient
            ingredient_arr = np.zeros(lenVoc)  # ingredients
            for iks in c_ingredients:
                ingredient_arr[vocab_ing(iks)] = 1

            len_sentences = len(c_sentences)
            if not (np.sum(ingredient_arr) > 0 and len_sentences > 1):
                print('ERRORR ', np.sum(ingredient_arr) > 0, len_sentences > 1)
                continue

            recipe_steps = []
            frame_feats = []
            with open(curr_feat_loc, 'rb') as f:
                data_feats = pickle.load(f)

            for bii in range(len_sentences):
                current_indices = c_frame_indices[bii]
                flag_in = False
                if len(current_indices) > 0:
                    current_indices = np.concatenate(current_indices, axis=0)
                    sel_frames = np.arange(current_indices[0], current_indices[-1], args.maxpoolDim)
                    rec_frame_feats = []
                    for fii in range(len(sel_frames)):
                        if args.frame_based_flag:
                            frame_name = "{:05d}".format(sel_frames[fii] - 1) + '.jpg'
                            rec_frame_feats.append(data_feats[frame_name])
                        else:
                            final_frame_sel = current_indices[-1] - 1
                            if fii + 1 < len(sel_frames):
                                final_frame_sel = sel_frames[fii + 1] - 1
                            sel_curr = []
                            for bkb in range(sel_frames[fii] - 1, final_frame_sel):
                                frame_name = "{:05d}".format(bkb) + '.jpg'
                                sel_curr.append(data_feats[frame_name])
                            sel_curr = np.concatenate(sel_curr, axis=0)
                            max_curr = np.max(sel_curr, axis=0)
                            max_curr = np.expand_dims(max_curr, axis=0)
                            rec_frame_feats.append(max_curr)
                    flag_in = True
                elif len_sentences - 1 == bii:  # predict with enjoy
                    rec_frame_feats = []
                    frame_name = "{:05d}".format(len(data_feats) - 1) + '.jpg'
                    rec_frame_feats.append(data_feats[frame_name])
                    flag_in = True
                else:
                    raise Exception('frame selection error')

                if flag_in:
                    rec_frame_feats = torch.tensor(np.concatenate(rec_frame_feats, 0), dtype=torch.float)
                    if len(rec_frame_feats.shape) == 1:
                        raise Exception('shape conflict when concatenating chunk features')
                    frame_feats.append(rec_frame_feats)
                    recipe_steps.append(c_sentences[bii])

            self.ingredients[str(counter_rec)] = ingredient_arr
            self.frame_feats[str(counter_rec)] = frame_feats
            self.sentences[str(counter_rec)] = recipe_steps

            if counter_rec % 50 == 0:
                print(counter_rec)
            counter_rec = counter_rec + 1

        print('total number of recipes in tasty: ' + str(counter_rec))
