# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random
import socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence
from visdom import Visdom

from dataset.dataloader_text import get_loader
from model.model_ce import BLSTMprojEncoder, SP_EMBEDDING
from model.model_ce import EncoderINGREDIENT, EncoderRECIPE, DecoderSENTENCES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Plotter(object):
    """Plots to Visdom"""

    def __init__(self, envName='main', log_filename=None, port=8097):
        self.viz = Visdom(port=port, log_to_filename=log_filename, env=envName)
        self.env = envName
        self.windows = {}
        self.log_filename = log_filename

    def plot(self, window_name, legend_name, x, y, ytype='linear', xlabel='', ylabel='', **kwargs):
        opts = {
            'legend': [legend_name],
            'title': window_name,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'ytype': ytype,
        }
        for key in kwargs:
            opts[key] = kwargs[key]
        if window_name not in self.windows:
            self.windows[window_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                                      name=legend_name, opts=opts)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.windows[window_name],
                          name=legend_name, update='append')


def train(args, name_repo):
    timestamp = name_repo  # unique identifier with descriptive name
    # current timestamp with second precision to avoid overwriting old stuff
    timestamp += datetime.now().strftime("%y%m%d_%H%M%S_")
    # add computer name to allow easier finding data scattered over multiple computers
    timestamp += socket.gethostname()

    out_dir = args.model_path
    plotter_filename = out_dir + timestamp + '_losses.json'  # plot points also go in this file for potential later use
    plotter_port = 8097  # standard visdom port
    plotter = Plotter(envName=timestamp, log_filename=plotter_filename, port=plotter_port)

    # Build the models
    encoder_ingredient = EncoderINGREDIENT(args).to(device)
    embed_words = SP_EMBEDDING(args).to(device)
    encoder_sentences = BLSTMprojEncoder(args).to(device)
    encoder_recipe = EncoderRECIPE(args).to(device)
    decoder_sentences = DecoderSENTENCES(args).to(device)

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    params = list(embed_words.parameters()) + \
             list(encoder_recipe.parameters()) + \
             list(encoder_ingredient.parameters()) + \
             list(decoder_sentences.parameters()) + \
             list(encoder_sentences.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Build data loader
    with open(args.vocab_bin, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    train_loader = get_loader(args, args.batch_size, vocab, shuffle=True, num_workers=args.num_workers)

    # Train the models
    use_teacherF = False
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        epoch_loss_all = 0

        # Set mini-batch dataset
        for i, (ingredients_v, rec_lens, sentences_v, sent_lens, indices, indices_encoder) in enumerate(train_loader):

            ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
            sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
            sent_lens = sent_lens.to(device)  # Nb-> total sent. num, max

            """ 1. encode sentences """
            word_embs = embed_words(sentences_v)  # [Nb, Ns, 256]
            sentence_enc = encoder_sentences(word_embs, sent_lens)  # [Nb, 1024]

            """ reshape sentences wrt the recipe order """
            # sort the indices
            _, orgj_idx = indices_encoder.sort(0, descending=False)  # [Nb]
            orgj_idx = Variable(orgj_idx).cuda()  # [Nb]

            # permute sentence according to instructional order within a batch
            sentence_enc = sentence_enc.index_select(0, orgj_idx)  # [Nb, 1024]

            # split according to the batch, note that the batch is ordered
            sentence_enc_spl = torch.split(sentence_enc, rec_lens, dim=0)

            # pack and pad the ordered sentences
            recipes_v_pckd = pack_sequence(sentence_enc_spl)
            recipes_v = pad_packed_sequence(recipes_v_pckd, batch_first=True)[0]  # [N, rec_lens[0], 1024]

            """ 2. encode ingredient """
            ingredient_feats = encoder_ingredient(ingredients_v).unsqueeze(1)  # [N, 1, 1024]

            """ 3. encode recipe """
            if epoch >= 5:
                use_teacherF = random.random() < (0.5)
            recipe_enc = encoder_recipe(ingredient_feats, recipes_v, rec_lens, use_teacherF)  # [Nb, 1024]

            """ 4. decode sentences """
            idx = Variable(indices).cuda()
            recipe_enc = recipe_enc.index_select(0, idx)  # [Nb, 1024]

            sentence_dec = decoder_sentences(recipe_enc, word_embs, sent_lens)
            # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary

            sentence_target = pack_padded_sequence(sentences_v, sent_lens, batch_first=True)[0]  # [ sum(sent_lens) ]

            """ Compute the loss """
            all_loss = criterion_sent(sentence_dec, sentence_target)
            epoch_loss_all += all_loss

            """ Backpropagation """
            encoder_recipe.zero_grad()
            encoder_ingredient.zero_grad()
            decoder_sentences.zero_grad()
            encoder_sentences.zero_grad()
            embed_words.zero_grad()

            all_loss.backward()
            optimizer.step()

            """ Printing and evaluations """
            if i % args.log_step == 0:  # Print log info
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f} '.format(epoch, args.num_epochs, i, total_step,
                                                                           all_loss.item()))

            if (i + 1) % args.save_step == 0:  # Print sentences
                generate(sentences_v[0, :], vocab, recipe_enc, decoder_sentences, embed_words)

        if (epoch + 1) % 5 == 0:  # Save the model checkpoints
            save_models(args, (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words),
                        epoch + 1)
        plotter.plot(window_name='all losses (epochs)', legend_name='loss 1', x=epoch + 1, y=epoch_loss_all.item(),
                     ytype='log', xlabel='epochs')

    # Save the final model
    save_models(args, (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words),
                epoch + 1)


def save_models(args, all_models, epoch_val):
    if epoch_val == 0:
        num_epochs = ''
    else:
        num_epochs = '-' + str(epoch_val)

    (encoder_recipe, encoder_ingredient, decoder_sentences, encoder_sentences, embed_words) = all_models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(encoder_recipe.state_dict(),
               os.path.join(args.model_path, 'encoder_recipe{}.ckpt'.format(num_epochs)))
    torch.save(encoder_ingredient.state_dict(),
               os.path.join(args.model_path, 'encoder_ingredient{}.ckpt'.format(num_epochs)))
    torch.save(decoder_sentences.state_dict(),
               os.path.join(args.model_path, 'decoder_sentences{}.ckpt'.format(num_epochs)))
    torch.save(encoder_sentences.state_dict(),
               os.path.join(args.model_path, 'encoder_sentences{}.ckpt'.format(num_epochs)))
    torch.save(embed_words.state_dict(),
               os.path.join(args.model_path, 'embed_words{}.ckpt'.format(num_epochs)))


def generate(sentences_v, vocab, recipe_enc, decoder_sentences, embed_words):
    target_sentence = ids2words(vocab, sentences_v.cpu().numpy())

    recipe_enc_gen = recipe_enc[0, :].view(1, -1)
    pred_ids = decoder_sentences.sample(recipe_enc_gen, embed_words)
    pred_sentence = ids2words(vocab, pred_ids[0].cpu().numpy())

    print('gt   : ', target_sentence)
    print('pred : ', pred_sentence)


def ids2words(vocab, target_ids):
    target_caption = []
    for word_id in target_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        target_caption.append(word)
    target_sentence = ' '.join(target_caption)
    return target_sentence


if __name__ == '__main__':
    COMP_PATH = '../TASTY_dir/'

    parser = argparse.ArgumentParser()
    name_repo = 'model_mine'

    intermediate_fd = os.path.join(COMP_PATH, 'INTERMEDIATE/' + name_repo + '/')
    json_fd = os.path.join(COMP_PATH, 'DATA/Recipe1M/')
    vocab_fd = os.path.join(COMP_PATH, 'DATA/vocab/')

    parser.add_argument('--json_joint', type=str, default=(json_fd + 'layer1_joint.json'), help='path for annotations')
    parser.add_argument('--vocab_bin', type=str, default=(vocab_fd + 'vocab_bin_30171.pkl'), help='')
    parser.add_argument('--vocab_ing', type=str, default=(vocab_fd + 'vocab_ing_3769.pkl'), help='')

    # model parameters
    parser.add_argument('--vocab_len', type=int, default=30171, help='')
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
    parser.add_argument('--sentences_sorted', type=int, default=1, help='')

    # training parameters
    parser.add_argument('--log_step', type=int, default=20, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=45, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    param_all = ('e' + str(args.recipe_inDim) + '_he' + str(args.sentEnd_hiddens) + '_hre' + str(
        args.recipe_hiddens) + '_hd' + str(args.sentDec_hiddens) + '_ep' + str(args.num_epochs) + '_b' + str(
        args.batch_size) + '_l' + str(args.learning_rate).replace(".", "_"))
    parser.add_argument('--model_path', type=str, default=(intermediate_fd + 'models_' + param_all + '/'),
                        help='path for saving trained models')
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if name_repo == 'model_mine':
        train(args, name_repo)
