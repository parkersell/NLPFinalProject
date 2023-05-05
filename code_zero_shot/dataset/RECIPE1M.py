# -*- coding: utf-8 -*-
import json
import pickle

import nltk
import numpy as np
from tqdm import *


class RECIPE1M:
    def __init__(self, args=None):

        self.sentences, self.ingredients = dict(), dict()

        layer1 = json.load(open(args.json_joint, 'r'))  # Load json

        with open(args.vocab_ing, 'rb') as f:
            vocab_ing = pickle.load(f)
        lenVoc = len(vocab_ing)

        counter_rec = 1
        for i, entry in tqdm(enumerate(layer1)):
            if not entry['valid'] == 'true':
                continue
            if entry['partition'] == 'test':
                continue

            c_ingredients = entry['ingredient_list']
            c_sentences = entry['instructions']
            c_ingredients = np.unique(c_ingredients)
            c_ingredients = c_ingredients[c_ingredients != 30170]  # remove the 'unk>' ingredient

            ingredient_arr = np.zeros(lenVoc)  # ingredients
            for iks in c_ingredients:
                ingredient_arr[vocab_ing(iks)] = 1

            lenSent = len(c_sentences)
            if np.sum(ingredient_arr) > 0 and lenSent > 1:
                all_sentences = []
                for x in range(0, lenSent):
                    instr = c_sentences[x]['text']
                    tokens = nltk.word_tokenize(instr)
                    words = [word.lower() for word in tokens if word.isalpha()]
                    if len(words) > 0:
                        all_sentences.append(tokens)

                if len(all_sentences) > 0:
                    self.ingredients[str(counter_rec)] = ingredient_arr
                    self.sentences[str(counter_rec)] = all_sentences
                    counter_rec = counter_rec + 1

            if counter_rec % 10000 == 0:
                print(counter_rec)

        print('total number of recipes in Recipe1M: ' + str(counter_rec))
