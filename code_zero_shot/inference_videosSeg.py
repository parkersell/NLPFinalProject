# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
import sys

import nltk
import numpy as np
import torch

from model.model_ce import BLSTMprojEncoder_FRAME, SP_EMBEDDING
from model.model_ce import EncoderINGREDIENT, EncoderRECIPE, DecoderSENTENCES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

COMP_PATH = '../TASTY_dir/'

sys.path.insert(0, (COMP_PATH + 'LIBS/coco-caption/'))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor


def load_models(args):
    # Build the models
    encoder_ingredient = EncoderINGREDIENT(args).to(device)
    embed_words = SP_EMBEDDING(args).to(device)
    frame_lstm = BLSTMprojEncoder_FRAME(args).to(device)
    encoder_recipe = EncoderRECIPE(args).to(device)
    decoder_sentences = DecoderSENTENCES(args).to(device)

    # Load the trained model parameters
    encoder_ingredient.load_state_dict(torch.load(args.encoder_ingredient))
    encoder_recipe.load_state_dict(torch.load(args.encoder_recipe))
    frame_lstm.load_state_dict(torch.load(args.frame_lstm))
    decoder_sentences.load_state_dict(torch.load(args.decoder_sentences))
    embed_words.load_state_dict(torch.load(args.embed_words))

    return encoder_ingredient.eval(), encoder_recipe.eval(), frame_lstm.eval(), \
           decoder_sentences.eval(), embed_words.eval()


def loadVocabs(args):
    with open(args.vocab_ing, 'rb') as f:
        vocab_ing = pickle.load(f)
    with open(args.vocab_bin, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    return vocab_ing, vocab


def load_ingredients(current_recipe, vocab_ing):
    c_ingredients = current_recipe['ingredients']
    c_sentences = current_recipe['recipe_steps']
    allings = current_recipe['ingredients_names']

    lenVoc = len(vocab_ing)
    c_ingredients = list(map(int, c_ingredients))
    c_ingredients = np.unique(c_ingredients)
    c_ingredients = c_ingredients[c_ingredients != 30170]  # remove the 'unk>' ingredient

    del_ing_list = [30170, 5085, 1563, 2159, 2297, 23, 5031, 7511, 869, 32, 66, 16016, 422, 3162, 2059, 1198, 4595,
                    15236, 3322, 5126, 6400, 8186, 18110, 3631, 1201, 929, 7283, 2269, 8376]
    if len(set(c_ingredients).intersection(set(del_ing_list))):
        print('----------------------- intersection', set(c_ingredients).intersection(set(del_ing_list)))

    ingredient_arr = np.zeros(lenVoc)  # ingredients
    for iks in c_ingredients:
        ingredient_arr[vocab_ing(iks)] = 1

    len_sentences = len(c_sentences)
    if not (np.sum(ingredient_arr) > 0 and len_sentences > 1):
        print('ERRORR ', np.sum(ingredient_arr) > 0, len_sentences > 1)

    return allings, c_ingredients, ingredient_arr


def encode_frames_test(args, current_recipe, frame_lstm, curr_feat_loc, name_recs):
    c_sentences = current_recipe['recipe_steps']
    c_frame_indices = current_recipe['frame_indices']
    recipe_steps = []
    frame_feats = []
    with open(curr_feat_loc, 'rb') as f:
        data_feats = pickle.load(f)

    if args.type_eval == 'window':
        lenSteps = args.prev_given
        frame_div = args.frame_div

        stop_point = c_frame_indices[lenSteps - 1][-1][0]
        remA = stop_point % frame_div
        multB = stop_point // frame_div

        start_frame = 0
        for bii in range(multB + 1):
            if bii == multB:
                if remA > 2:
                    current_indices = list(range(start_frame, start_frame + remA))
                else:
                    break
            else:
                current_indices = list(range(start_frame, start_frame + frame_div))
                start_frame += frame_div
            sel_frames = np.arange(current_indices[0], current_indices[-1], args.maxpoolDim)
            rec_frame_feats = []
            for fii in range(len(sel_frames)):
                frame_name = "{:05d}".format(sel_frames[fii]) + '.jpg'
                rec_frame_feats.append(data_feats[frame_name])
            rec_frame_feats = torch.tensor(np.concatenate(rec_frame_feats, 0), dtype=torch.float)
            frame_feats.append(rec_frame_feats)

        recipe_steps.append(c_sentences[lenSteps])

    elif args.type_eval == "proposal":
        proposals = json.load(open(COMP_PATH + "/DATA/seg_outs/NMS_proposals.json", "r"))
        curr_proporal = proposals[name_recs[:-4]]

        lenSteps = args.prev_given
        stop_point = c_frame_indices[lenSteps - 1][-1][0]

        frame_lists = []
        for kkl in range(len(curr_proporal)):
            ex_proposal = curr_proporal[kkl]
            if ex_proposal[0] == ex_proposal[1]:
                continue
            if stop_point > ex_proposal[1]:
                frame_lists.append(ex_proposal)
            else:
                ex2_proposal = [ex_proposal[0], stop_point]
                if ex2_proposal[0] > ex2_proposal[1]:
                    frame_lists.append([0, stop_point])
                elif not (ex2_proposal[0] == ex2_proposal[1]):
                    frame_lists.append(ex2_proposal)
                break

        for bii in range(len(frame_lists)):
            current_indices = frame_lists[bii]
            sel_frames = np.arange(current_indices[0], current_indices[-1], args.maxpoolDim)
            rec_frame_feats = []
            for fii in range(len(sel_frames)):
                frame_name = "{:05d}".format(sel_frames[fii]) + '.jpg'
                rec_frame_feats.append(data_feats[frame_name])
            if len(rec_frame_feats) == 0:
                aa = 1
            rec_frame_feats = torch.tensor(np.concatenate(rec_frame_feats, 0), dtype=torch.float)
            frame_feats.append(rec_frame_feats)

        recipe_steps.append(c_sentences[lenSteps])

    lengths_frames_lstm = [cap.shape[0] for cap in frame_feats]
    STs_frames = torch.zeros(len(lengths_frames_lstm), max(lengths_frames_lstm), 2048).float()
    for i, cap in enumerate(frame_feats):
        end = lengths_frames_lstm[i]
        STs_frames[i, :end] = cap[:end]

    target_instructions = frame_lstm.forward(STs_frames, torch.LongTensor(lengths_frames_lstm))

    return target_instructions, recipe_steps


def write_file_recipe_details(args, title, sentences, curr_ingredients, all_ingredients, vocab_words):
    with open(args.eval_file, "a") as text_file:
        text_file.write('-----------------------------------------------\n')
        text_file.write('Recipe name: %s, ' % title + '\n')
        text_file.write('Ingredients: %s' % all_ingredients + '\n')
        text_file.write('Ingredients in the Vocabulary: ')

    for ccy in range(0, len(curr_ingredients)):
        if curr_ingredients[ccy] > 1:
            val = curr_ingredients[ccy]
            with open(args.eval_file, "a") as text_file:
                text_file.write(' %s, ' % (vocab_words.idx2word[val]))
    with open(args.eval_file, "a") as text_file:
        text_file.write('\n\n')

    len_inst = len(sentences)
    for x in range(0, len_inst):
        with open(args.eval_file, "a") as text_file:
            text_file.write('gt%d: %s' % (x, sentences[x]) + '\n')
    with open(args.eval_file, "a") as text_file:
        text_file.write('\n')


def ids2words(vocab_words, target_ids):
    target_caption = []
    for word_id in target_ids:
        word = vocab_words.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        target_caption.append(word)
    target_sentence = ' '.join(target_caption)
    return target_sentence


def print_scores(gts, res, scorers, tokenizer, eval_values):
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    current_counts = np.zeros(eval_values)
    cco = 0
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                current_counts[cco] = sc
                cco = cco + 1
        else:
            current_counts[cco] = score
            cco = cco + 1
    return current_counts


def write_file_more_ingredient(args, xxi, reference_s, prediction_s, current_counts):
    with open(args.eval_file, "a") as text_file:
        text_file.write('gt%d: %s' % (xxi, reference_s) + '\n')
        text_file.write('pr%d: %s' % (xxi, prediction_s) + '\n')
        text_file.write('Ing : %0.4f' % current_counts + ' - ')
        text_file.write('\n\n')


def write_file_more_verb(args, xxi, reference_s, prediction_s, current_counts):
    with open(args.eval_file, "a") as text_file:
        text_file.write('gt%d: %s' % (xxi, reference_s) + '\n')
        text_file.write('pr%d: %s' % (xxi, prediction_s) + '\n')
        text_file.write('Verb : %0.4f' % current_counts + ' - ')
        text_file.write('\n\n')


def write_file_more_sentences(args, xxi, reference_s, prediction_s, current_counts):
    with open(args.eval_file, "a") as text_file:
        text_file.write('gt%d: %s' % (xxi, reference_s) + '\n')
        text_file.write('pr%d: %s' % (xxi, prediction_s) + '\n')
        text_file.write('Blue_1: %0.4f' % (current_counts[0]) + ' - ')
        text_file.write('Blue_2: %0.4f' % (current_counts[1]) + ' - ')
        text_file.write('Blue_3: %0.4f' % (current_counts[2]) + ' - ')
        text_file.write('Blue_4: %0.4f' % (current_counts[3]) + ' - ')
        text_file.write('Meteor: %0.4f' % (current_counts[4]) + ' - ')
        text_file.write('\n\n')


def write_all_means_verbs(args, global_s, counts_s, n_next, counter_rec):
    with open(args.eval_file, "a") as text_file:
        text_file.write('-----------------------------------------------\n')
        text_file.write('counter_rec %d: \n' % counter_rec)
    for xii in range(n_next):
        with open(args.eval_file, "a") as text_file:
            text_file.write('Mean step%d : ' % xii)
            text_file.write('Verb : %0.4f' % (global_s[xii] / counts_s[xii]) + ' - ')
            text_file.write('\n')


def write_all_means_ingredient(args, global_s, counts_s, n_next, counter_rec):
    with open(args.eval_file, "a") as text_file:
        text_file.write('-----------------------------------------------\n')
        text_file.write('counter_rec %d: \n' % counter_rec)
    for xii in range(n_next):
        with open(args.eval_file, "a") as text_file:
            text_file.write('Mean step%d : ' % xii)
            text_file.write('Ing : %0.4f' % (global_s[xii] / counts_s[xii]) + ' - ')
            text_file.write('\n')


def write_all_means_sentences(args, global_s, counts_s, n_next, counter_rec):
    with open(args.eval_file, "a") as text_file:
        text_file.write('-----------------------------------------------\n')
        text_file.write('counter_rec %d: \n' % (counter_rec))
    for xii in range(n_next):
        with open(args.eval_file, "a") as text_file:
            text_file.write('Mean step%d: ' % xii)
            text_file.write('Blue_1: %0.4f' % (global_s[0, xii] / counts_s[xii]) + ' - ')
            text_file.write('Blue_2: %0.4f' % (global_s[1, xii] / counts_s[xii]) + ' - ')
            text_file.write('Blue_3: %0.4f' % (global_s[2, xii] / counts_s[xii]) + ' - ')
            text_file.write('Blue_4: %0.4f' % (global_s[3, xii] / counts_s[xii]) + ' - ')
            text_file.write('Meteor: %0.4f' % (global_s[4, xii] / counts_s[xii]) + ' - ')
            text_file.write('\n')


def main_ingredients(args):
    n_next = args.nPred_next
    global_s_all = np.zeros(n_next)
    counts_s_all = np.zeros(n_next)
    global_s_one = np.zeros(n_next)
    counts_s_one = np.zeros(n_next)

    ingredients_dict = []

    vocab_ing, vocab = loadVocabs(args)
    encoder_ingredient, encoder_recipe, frame_lstm, decoder_sentences, embed_words = load_models(args)

    counter_rec = 0
    print('Reading the test dataset')
    name_recs = os.listdir(args.tasty_path)
    for ite in range(len(name_recs)):
        curr_rec_loc = args.tasty_path + '/' + name_recs[ite]
        curr_feat_loc = args.feat_path + '/' + name_recs[ite]

        with open(curr_rec_loc, 'rb') as f:
            current_recipe = pickle.load(f)

        if not (current_recipe['split'] == 'test'):
            continue

        print(counter_rec, name_recs[ite])

        c_sentences = current_recipe['recipe_steps']

        lenSent = len(c_sentences)
        lenSteps = args.prev_given

        if not args.sent_len_fixed == 0:
            if not lenSent == args.sent_len_fixed:
                continue

        allings, curr_ingrt, ingredient_arr = load_ingredients(current_recipe, vocab_ing)

        if np.sum(ingredient_arr) > 0 and lenSent > 1:
            if lenSteps >= min(lenSent, lenSteps + n_next):
                continue

            if args.write_details:
                write_file_recipe_details(args, current_recipe['title'], c_sentences, curr_ingrt, allings, vocab)

            """  encode ingredients """
            ingredient_feats = encoder_ingredient(torch.tensor(ingredient_arr, dtype=torch.float).unsqueeze(0))

            """  decode recipe steps """
            if args.prev_given == 0:
                sampled_instructions = encoder_recipe.sample_INGgiven(ingredient_feats, lenSent).cuda()
            else:
                """  encode sentences """
                target_instructions, c_sentences = encode_frames_test(args, current_recipe, frame_lstm, curr_feat_loc,
                                                                      name_recs[ite])

                sampled_instructions = encoder_recipe.sample_nGiven_INF(ingredient_feats, target_instructions).cuda()
                # sampled_instructions -> [1, lenSent, 1024])

            """  evaluate generated steps """
            current_gt_and_pred = np.zeros(n_next)
            current_gt_only = np.zeros(n_next)
            cc_co = 0

            for x in range(1):
                # predictions
                recipe_enc_pred = sampled_instructions  # [1, 1024])

                if args.beam_yes == 1:
                    pred_ids = decoder_sentences.sample_beam_decode(recipe_enc_pred, embed_words)
                else:
                    # pred_ids = decoder_sentences.sample_greedy_decode(recipe_enc_pred, embed_words)
                    pred_ids = decoder_sentences.sample(recipe_enc_pred, embed_words)
                prediction_s = ids2words(vocab, pred_ids[0].cpu().numpy()).lower()

                # ground truths
                reference_s = c_sentences[x].lower()

                # compute scores
                tokens_gt = nltk.tokenize.word_tokenize(reference_s)
                tokens_pr = nltk.tokenize.word_tokenize(prediction_s)

                det_ing_gt = []
                det_ing_pred = []
                for ifft in range(len(curr_ingrt)):
                    vocab_el = vocab.idx2word[curr_ingrt[ifft]]
                    if vocab_el in tokens_gt:
                        det_ing_gt.append(vocab_el)
                        if vocab_el in tokens_pr:
                            det_ing_pred.append(vocab_el)
                            current_gt_and_pred[cc_co] = current_gt_and_pred[cc_co] + 1
                        current_gt_only[cc_co] = current_gt_only[cc_co] + 1
                if len(det_ing_gt) > 0:
                    ingredients_dict.append({'key': str(ite), 'len_sent': lenSent, 'step': cc_co, 'ing_gt': det_ing_gt,
                                             'ing_pred': det_ing_pred})

                if args.write_details:
                    write_file_more_ingredient(args, x, reference_s, prediction_s,
                                               float(current_gt_and_pred[cc_co]) / float(current_gt_only[cc_co]))
                cc_co = cc_co + 1

            global_s_all = global_s_all + current_gt_and_pred
            counts_s_all = counts_s_all + current_gt_only

            current_gt_and_pred[np.abs(current_gt_and_pred) > 1] = 1
            current_gt_only[np.abs(current_gt_only) > 1] = 1
            global_s_one = global_s_one + current_gt_and_pred
            counts_s_one = counts_s_one + current_gt_only

            counter_rec = counter_rec + 1

        if counter_rec % 100 == 0:
            print(counter_rec)
            # write_all_means(args, global_s_all, counts_s_all, n_next, counter_rec)

    if args.write_details:
        with open(args.dict_save_file, 'wb') as f:
            pickle.dump(ingredients_dict, f)
    print('  [*] counter_rec: ', counter_rec)
    write_all_means_ingredient(args, global_s_all, counts_s_all, n_next, counter_rec)


def main_verbs(args):
    n_next = args.nPred_next
    global_s_all = np.zeros(n_next)
    counts_s_all = np.zeros(n_next)
    global_s_one = np.zeros(n_next)
    counts_s_one = np.zeros(n_next)

    verb_dict = []
    with open(args.valid_words_path, 'rb') as handle:
        valid_words = pickle.load(handle)

    vocab_ing, vocab = loadVocabs(args)
    encoder_ingredient, encoder_recipe, frame_lstm, decoder_sentences, embed_words = load_models(args)

    counter_rec = 0
    print('Reading the test dataset')
    name_recs = os.listdir(args.tasty_path)
    for ite in range(len(name_recs)):
        curr_rec_loc = args.tasty_path + '/' + name_recs[ite]
        curr_feat_loc = args.feat_path + '/' + name_recs[ite]

        with open(curr_rec_loc, 'rb') as f:
            current_recipe = pickle.load(f)

        if not (current_recipe['split'] == 'test'):
            continue

        print(counter_rec, name_recs[ite])
        c_sentences = current_recipe['recipe_steps']
        lenSent = len(c_sentences)
        lenSteps = args.prev_given

        if not args.sent_len_fixed == 0:
            if not lenSent == args.sent_len_fixed:
                continue

        allings, curr_ingrt, ingredient_arr = load_ingredients(current_recipe, vocab_ing)

        if np.sum(ingredient_arr) > 0 and lenSent > 1:
            if lenSteps >= min(lenSent, lenSteps + n_next):
                continue

            if args.write_details:
                write_file_recipe_details(args, current_recipe['title'], c_sentences, curr_ingrt, allings, vocab)

            """  encode ingredients """
            ingredient_feats = encoder_ingredient(torch.tensor(ingredient_arr, dtype=torch.float).unsqueeze(0))

            """  decode recipe steps """
            if args.prev_given == 0:
                sampled_instructions = encoder_recipe.sample_INGgiven(ingredient_feats, lenSent).cuda()
            else:
                """  encode sentences """
                target_instructions, c_sentences = encode_frames_test(args, current_recipe, frame_lstm, curr_feat_loc,
                                                                      name_recs[ite])

                sampled_instructions = encoder_recipe.sample_nGiven_INF(ingredient_feats, target_instructions).cuda()
                # sampled_instructions -> [1, lenSent, 1024])

            """  evaluate generated steps """
            current_gt_and_pred = np.zeros(n_next)
            current_gt_only = np.zeros(n_next)
            cc_co = 0

            for x in range(1):
                # predictions
                recipe_enc_pred = sampled_instructions  # [1, 1024])

                if args.beam_yes == 1:
                    pred_ids = decoder_sentences.sample_beam_decode(recipe_enc_pred, embed_words)
                else:
                    # pred_ids = decoder_sentences.sample_greedy_decode(  recipe_enc_pred, embed_words)
                    pred_ids = decoder_sentences.sample(recipe_enc_pred, embed_words)
                prediction_s = ids2words(vocab, pred_ids[0].cpu().numpy()).lower()

                reference_s = c_sentences[x].lower()  # ground truths

                # compute scores
                tokens_gt = nltk.tokenize.word_tokenize(reference_s)
                tokens_pr = nltk.tokenize.word_tokenize(prediction_s)

                det_ing_gt = []
                det_ing_pred = []
                for ifft in range(len(valid_words)):
                    curr_verb_val = valid_words[ifft]
                    if curr_verb_val in tokens_gt:
                        det_ing_gt.append(curr_verb_val)
                        current_gt_only[cc_co] = current_gt_only[cc_co] + 1
                        if curr_verb_val in tokens_pr:
                            det_ing_pred.append(curr_verb_val)
                            current_gt_and_pred[cc_co] = current_gt_and_pred[cc_co] + 1
                if len(det_ing_gt) > 0:
                    verb_dict.append({'key': str(ite), 'len_sent': lenSent, 'step': cc_co, 'ing_gt': det_ing_gt,
                                      'ing_pred': det_ing_pred})

                if args.write_details:
                    write_file_more_verb(args, x, reference_s, prediction_s,
                                         float(current_gt_and_pred[cc_co]) / float(current_gt_only[cc_co] + 0.00000001))
                cc_co = cc_co + 1

            global_s_all = global_s_all + current_gt_and_pred
            counts_s_all = counts_s_all + current_gt_only

            current_gt_and_pred[np.abs(current_gt_and_pred) > 1] = 1
            current_gt_only[np.abs(current_gt_only) > 1] = 1
            global_s_one = global_s_one + current_gt_and_pred
            counts_s_one = counts_s_one + current_gt_only

            counter_rec = counter_rec + 1

        if counter_rec % 100 == 0:
            print(counter_rec)
            # write_all_means_verbs(args, global_s_all, counts_s_all, n_next, counter_rec)

    if args.write_details:
        with open(args.dict_save_file, 'wb') as f:
            pickle.dump(verb_dict, f)
    print('  [*] counter_rec: ', counter_rec)
    write_all_means_verbs(args, global_s_all, counts_s_all, n_next, counter_rec)


def main_sentences(args):
    tokenizer = PTBTokenizer()
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Meteor(), "METEOR")]

    n_next = args.nPred_next
    eval_vals = 5  # blue1-4 and meteor
    global_s = np.zeros((eval_vals, n_next))
    counts_s = np.zeros(n_next)
    sentences_dict = []

    ingredients_dict = []
    vocab_ing, vocab = loadVocabs(args)
    encoder_ingredient, encoder_recipe, frame_lstm, decoder_sentences, embed_words = load_models(args)

    counter_rec = 0
    print('Reading the test dataset')
    name_recs = os.listdir(args.tasty_path)
    for ite in range(len(name_recs)):
        curr_rec_loc = args.tasty_path + '/' + name_recs[ite]
        curr_feat_loc = args.feat_path + '/' + name_recs[ite]

        with open(curr_rec_loc, 'rb') as f:
            current_recipe = pickle.load(f)

        if not (current_recipe['split'] == 'test'):
            continue

        print(counter_rec, name_recs[ite])

        c_sentences = current_recipe['recipe_steps']

        lenSent = len(c_sentences)
        lenSteps = args.prev_given

        if not args.sent_len_fixed == 0:
            if not lenSent == args.sent_len_fixed:
                continue

        allings, curr_ingrt, ingredient_arr = load_ingredients(current_recipe, vocab_ing)

        if np.sum(ingredient_arr) > 0 and lenSent > 1:
            if lenSteps >= min(lenSent, lenSteps + n_next):
                continue

            if args.write_details:
                write_file_recipe_details(args, current_recipe['title'], c_sentences, curr_ingrt, allings, vocab)

            """  encode ingredients """
            ingredient_feats = encoder_ingredient(torch.tensor(ingredient_arr, dtype=torch.float).unsqueeze(0))

            """  decode recipe steps """
            if args.prev_given == 0:
                sampled_instructions = encoder_recipe.sample_INGgiven(ingredient_feats, lenSent).cuda()
            else:
                """  encode sentences """
                target_instructions, c_sentences = encode_frames_test(args, current_recipe, frame_lstm, curr_feat_loc,
                                                                      name_recs[ite])

                sampled_instructions = encoder_recipe.sample_nGiven_INF(ingredient_feats, target_instructions).cuda()
                # sampled_instructions -> [1, lenSent, 1024])

            """  evaluate generated steps """
            current_sums = np.zeros((eval_vals, n_next))
            current_counts = np.zeros(n_next)
            cc_co = 0
            gts = {}
            res = {}

            for x in range(1):
                # predictions
                recipe_enc_pred = sampled_instructions  # [1, 1024])

                if args.beam_yes == 1:
                    pred_ids = decoder_sentences.sample_beam_decode(recipe_enc_pred, embed_words)
                else:
                    # pred_ids = decoder_sentences.sample_greedy_decode( recipe_enc_pred, embed_words)
                    pred_ids = decoder_sentences.sample(recipe_enc_pred, embed_words)
                prediction_s = ids2words(vocab, pred_ids[0].cpu().numpy()).lower()

                reference_s = c_sentences[x].lower()  # ground truths

                print('gt ', reference_s)
                print('pr ', prediction_s)
                print()

                gts['0'] = []
                gts['0'].append({'caption': reference_s})
                res['0'] = []
                res['0'].append({'caption': prediction_s})

                # compute the scores
                res_pr = print_scores(gts, res, scorers, tokenizer, eval_vals)

                if args.write_details:
                    write_file_more_sentences(args, x, reference_s, prediction_s, res_pr)

                current_sums[:, cc_co] = current_sums[:, cc_co] + res_pr
                current_counts[cc_co] = current_counts[cc_co] + 1

                sentences_dict.append({'key': str(ite), 'len_sent': lenSent, 'step': cc_co, 'gt': reference_s,
                                       'pred': prediction_s, 'scores': res_pr})
                cc_co = cc_co + 1

            global_s = global_s + current_sums
            counts_s = counts_s + current_counts

            counter_rec = counter_rec + 1

        if counter_rec % 100 == 0:
            print('-----------------------------------  ', str(counter_rec))
            # break
            # write_all_means_sentences(args, global_s, counts_s, n_next, counter_rec)

    if args.write_details:
        with open(args.dict_save_file, 'wb') as f:
            pickle.dump(sentences_dict, f)
    write_all_means_sentences(args, global_s, counts_s, n_next, counter_rec)


if __name__ == '__main__':

    eval_type = 'ing'
    eval_type = 'verb'
    eval_type = 'sent'

    in_type = 'proposal'  # 'window' 'proposal'

    name_repo = 'model_ce'
    def_model_file = 'models_e1024_he512_hre1024_hd512_ep50_b50_l0_001_video/'

    beam_yes_list = [1]

    for beam_yes in beam_yes_list:
        print(beam_yes)
        parser = argparse.ArgumentParser()

        data_folder = os.path.join(COMP_PATH, 'INTERMEDIATE/' + name_repo + '/')
        model_loc = data_folder + def_model_file
        parser.add_argument('--frame_lstm', type=str, default=model_loc + 'frame_lstm.ckpt', help='')
        parser.add_argument('--encoder_ingredient', type=str, default=model_loc + 'encoder_ingredient.ckpt', help='')
        parser.add_argument('--encoder_recipe', type=str, default=model_loc + 'encoder_recipe.ckpt', help='')
        parser.add_argument('--decoder_sentences', type=str, default=model_loc + 'decoder_sentences.ckpt', help='')
        parser.add_argument('--embed_words', type=str, default=model_loc + 'embed_words.ckpt', help='')

        tasty_path = os.path.join(COMP_PATH, 'DATASET_tasty')
        feat_path = os.path.join(COMP_PATH, 'FEATURES')
        parser.add_argument('--tasty_path', type=str, default=tasty_path, help='')
        parser.add_argument('--feat_path', type=str, default=feat_path, help='')

        json_fd = os.path.join(COMP_PATH, 'DATA/Recipe1M/')
        vocab_fd = os.path.join(COMP_PATH, 'DATA/vocab/')
        lmdb_fd = os.path.join(COMP_PATH, 'DATA/Recipe1M/subset_ST_features/')
        verb_fd = os.path.join(COMP_PATH, 'DATA/verbs/')

        parser.add_argument('--vocab_bin', type=str, default=(vocab_fd + 'vocab_bin_30171.pkl'), help='')
        parser.add_argument('--vocab_ing', type=str, default=(vocab_fd + 'vocab_ing_3769.pkl'), help='')
        parser.add_argument('--vocab_len', type=int, default=30171, help='')
        parser.add_argument('--dict_test', type=str, default=(lmdb_fd + 'test_dict.npy'), help='')

        # Model parameters
        parser.add_argument('--inredient_dim', type=int, default=3769, help='')
        parser.add_argument('--word_dim', type=int, default=256, help='')
        parser.add_argument('--sentEnd_hiddens', type=int, default=512, help='')
        parser.add_argument('--sentEnd_nlayers', type=int, default=1, help='')
        parser.add_argument('--recipe_inDim', type=int, default=1024, help='')
        parser.add_argument('--recipe_hiddens', type=int, default=1024, help='')
        parser.add_argument('--recipe_nlayers', type=int, default=1, help='')
        parser.add_argument('--sentDec_inDim', type=int, default=1024, help='')
        parser.add_argument('--sentDec_hiddens', type=int, default=512, help='')
        parser.add_argument('--sentDec_nlayers', type=int, default=1, help='')
        parser.add_argument('--featDim', type=int, default=2048, help='')
        parser.add_argument('--sentences_sorted', type=int, default=0, help='')
        parser.add_argument('--beam_yes', type=int, default=beam_yes, help='')

        # test parameters
        parser.add_argument('--prev_given', type=int, default=-1, help='')
        parser.add_argument('--sent_len_fixed', type=int, default=0, help='if 0 use all recipes')
        parser.add_argument('--valid_words_path', type=str, default=(verb_fd + 'valid_verbs.pkl'), help='')
        parser.add_argument('--eval_file', type=str, default=(data_folder + 'empty_' + '.txt'), help='')
        parser.add_argument('--dict_save_file', type=str, default=(data_folder + 'empty_' + '.pkl'), help='')
        parser.add_argument('--write_details', type=bool, default=False, help='')
        parser.add_argument('--nPred_next', type=int, default=1, help='')  # !!!!!

        parser.add_argument('--frame_based_flag', type=bool, default=True, help='')
        parser.add_argument('--maxpoolDim', type=int, default=3, help='')
        parser.add_argument('--frame_div', type=int, default=150, help='')
        parser.add_argument('--type_eval', type=str, default=in_type, help='gt, window, proposal')

        args = parser.parse_args()

        with open(args.vocab_bin, 'rb') as f:
            vocab = pickle.load(f, encoding='latin1')

        if eval_type == 'ing':
            name_save_fd = data_folder + def_model_file + 'seg_act_ING_' + str(args.sent_len_fixed) + '/'
            if not os.path.exists(name_save_fd):
                os.makedirs(name_save_fd)
            name_save_fd = name_save_fd + 'ings_'
            for kk in range(1, 9):
                args.prev_given = kk
                args.eval_file = name_save_fd + str(kk) + '.txt'
                args.dict_save_file = name_save_fd + str(kk) + '.pkl'
                main_ingredients(args)

        elif eval_type == 'verb':
            name_save_fd = data_folder + def_model_file + 'seg_act_VERB_' + str(args.sent_len_fixed) + '/'
            if not os.path.exists(name_save_fd):
                os.makedirs(name_save_fd)
            name_save_fd = name_save_fd + 'verbs_'
            for kk in range(1, 9):
                args.prev_given = kk
                args.eval_file = name_save_fd + str(kk) + '.txt'
                args.dict_save_file = name_save_fd + str(kk) + '.pkl'
                main_verbs(args)

        elif eval_type == 'sent':
            name_save_fd = data_folder + def_model_file + 'seg_act_SENTENCE_' + str(args.sent_len_fixed) + '/'
            if not os.path.exists(name_save_fd):
                os.makedirs(name_save_fd)
            name_save_fd = name_save_fd + 'sents_'
            for kk in range(1, 9):
                args.prev_given = kk
                args.eval_file = name_save_fd + str(kk) + '.txt'
                args.dict_save_file = name_save_fd + str(kk) + '.pkl'
                main_sentences(args)
