# -*- coding: utf-8 -*-
import operator
import random
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class BLSTMprojEncoder_FRAME(nn.Module):

    def __init__(self, args):
        super(BLSTMprojEncoder_FRAME, self).__init__()
        self.featDim = args.featDim
        self.enc_lstm_dim = args.sentEnd_hiddens
        self.pool_type = "max"
        self.dpout_model = 0.

        self.enc_lstm = nn.LSTM(self.featDim, self.enc_lstm_dim, 1, bidirectional=True, dropout=self.dpout_model)
        self.proj_enc = nn.Linear(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, bias=False)

    def forward(self, sent, sent_len):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent = sent.permute(1, 0, 2)
        bsize = sent.size(1)

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).cuda())

        # Handling padding in Recurrent Networks
        sent_len_tensor = torch.from_numpy(sent_len.copy())
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_tensor.cpu())
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort).cuda())

        sent_output = self.proj_enc(sent_output.view(-1, 2 * self.enc_lstm_dim)).view(-1, bsize, 2 * self.enc_lstm_dim)

        # max pooling
        emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb


class SP_EMBEDDING(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(SP_EMBEDDING, self).__init__()
        self.word_emb_dim = args.word_dim  # 256
        self.vocab_len = args.vocab_len  # 30171
        self.embed = nn.Embedding(self.vocab_len, self.word_emb_dim)

    def forward(self, sent_words):
        # sent_words --> [Nb, Ns] -> [total num sent, max sent len.]
        return self.embed(sent_words)  # [Nb, Ns, 256]


class BLSTMprojEncoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(BLSTMprojEncoder, self).__init__()
        self.word_emb_dim = args.word_dim
        self.enc_lstm_dim = args.sentEnd_hiddens
        self.pool_type = "max"
        self.dpout_model = 0.
        self.vocab_len = args.vocab_len
        self.sentEnd_nlayers = args.sentEnd_nlayers
        self.sentences_sorted = args.sentences_sorted

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.sentEnd_nlayers, bidirectional=True,
                                dropout=self.dpout_model)
        self.proj_enc = nn.Linear(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, bias=False)

    def forward(self, sent, sent_len):
        # sent: [Nb, Ns, 256],  Nb->total num sent., Ns->max sent len.]
        # sent_len:  Nb-> total sent. num, max

        sent = sent.permute(1, 0, 2)  # [Ns, Nb, 256]
        bsize = sent.size(1)  # Nb

        if not self.sentences_sorted:
            # Sort by length (keep idx)
            sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
            idx_sort = torch.LongTensor(idx_sort).cuda()
            sent = sent.index_select(1, idx_sort)  # sent --> [Ns, Nb, 256]

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        # [0] --> [NbxNs, 256]  #[1] --> Ns
        sent_out = self.enc_lstm(sent_packed)[0]
        # [0] --> [NbxNs, 1024]  # [1] --> Ns
        sent_out = nn.utils.rnn.pad_packed_sequence(sent_out)[0]
        # seqlen x batch x 1024 -- [Ns, Nb, 1024]

        if not self.sentences_sorted:
            # Un-sort by length
            idx_unsort = torch.LongTensor(np.argsort(idx_sort)).cuda()
            sent_out = sent_out.index_select(1, idx_unsort)
            # sent --> [Ns, Nb, 1024]

        sent_out = sent_out.view(-1, 2 * self.enc_lstm_dim)  # (Ns*Nb) X 1024
        sent_out = self.proj_enc(sent_out)  # (Ns*Nb) X 1024
        sent_out = sent_out.view(-1, bsize, 2 * self.enc_lstm_dim)  # [Ns, Nb, 1024]

        # Pooling
        if self.pool_type == "mean":
            sent_len2 = sent_len.float().unsqueeze(1).cuda()  # [Nb, 1]
            emb = torch.sum(sent_out, 0).squeeze(0)  # [Nb, 1024]
            emb = emb / sent_len2.expand_as(emb)  # [Nb, 1024]
        elif self.pool_type == "last":
            I_out = sent_len.cpu().view(1, -1, 1)  # [1, Nb, 1]
            I_out = Variable(I_out.expand(1, bsize, 2 * self.enc_lstm_dim) - 1).cuda()  # [1, Nb, 1024]
            emb = torch.gather(sent_out, 0, I_out).squeeze(0)  # [Nb, 1024]
        elif self.pool_type == "max":
            emb = torch.max(sent_out, 0)[0].squeeze(0)  # [Nb, 1024]

        return emb  # [Nb, 1024]


class EncoderINGREDIENT(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(EncoderINGREDIENT, self).__init__()
        self.ing_prj_dim = args.recipe_inDim
        self.inredient_dim = args.inredient_dim
        self.linear = nn.Linear(self.inredient_dim, self.ing_prj_dim)
        self.bn = nn.BatchNorm1d(self.ing_prj_dim, momentum=0.01)

    def forward(self, feats):
        # feats --> [N, Nv]
        features = self.linear(feats)  # [N, 1024]
        features = self.bn(features)  # [N, 1024]
        return features


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.logp < other.logp


class DecoderSENTENCES(nn.Module):
    def __init__(self, args, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderSENTENCES, self).__init__()
        self.sentDec_inDim = args.sentDec_inDim
        self.word_dim = args.word_dim
        self.sentDec_hiddens = args.sentDec_hiddens
        self.vocab_len = args.vocab_len
        self.sentDec_nlayers = args.sentDec_nlayers

        self.lstm = nn.LSTM(self.word_dim, self.sentDec_hiddens, self.sentDec_nlayers, batch_first=True)
        self.linear = nn.Linear(self.sentDec_hiddens, self.vocab_len)
        self.max_seg_length = max_seq_length

        self.linear_project = nn.Linear(self.sentDec_inDim, self.word_dim)

    def forward(self, recipe_enc, word_embs, sent_lens):
        """Decode sentence feature vectors and generates sentences."""
        # recipe_enc --> [Nb, 1024]
        # word_embs  --> [Nb, Ns, 256]
        # len(sent_lens)  --> Nb

        features = self.linear_project(recipe_enc)  # [Nb, 256]
        word_embs = torch.cat((features.unsqueeze(1), word_embs), 1)  # torch.Size([Nb, Ns + 1, 256])
        packed = pack_padded_sequence(word_embs, sent_lens.cpu(), batch_first=True)
        # [0] -> [sum(sent_lens), 256]   [1] -> [sent_lens[0]]

        out, _ = self.lstm(packed)  # [0] -> [sum(sent_lens), 512]   [1] -> [sent_lens[0]]
        outputs = self.linear(out[0])  # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary

        return outputs

    def forward_sample(self, embed_words, inputPred, hidden):

        embedded = embed_words(inputPred).unsqueeze(1)  # [Nb, 256]
        output, hidden = self.lstm(embedded, hidden)

        output = self.linear(output.squeeze(0))
        output = F.log_softmax(output, dim=1)
        return output, hidden

    def sample(self, recipe_enc, embed_words, states=None):
        # recipe_enc --> [Nb, 1024]
        max_seq_length = 20
        sampled_ids = []
        features = self.linear_project(recipe_enc)  # [Nb, 256]
        inputs = features.unsqueeze(1)  # [Nb, 1, 256]
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [Nb, 1, 512]

            outputs = self.linear(hiddens.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary

            _, predicted = outputs.max(1)  # Nb
            sampled_ids.append(predicted)

            inputs = embed_words(predicted)  # [Nb, 256]
            inputs = inputs.unsqueeze(1)  # [Nb, 1,256]

        sampled_ids = torch.stack(sampled_ids, 1)  # [Nb, max_seq_length]

        return sampled_ids

    def sample_greedy_decode(self, step_enc, embed_words, decoder_hidden=None):
        '''
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :return: decoded_batch
        '''
        batch_size = 1
        MAX_LENGTH = 20
        decoded_batch = torch.zeros((batch_size, MAX_LENGTH))

        features = self.linear_project(step_enc)  # [Nb, 256]
        decoder_input = features.unsqueeze(1)  # [Nb, 1, 256]
        decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)  # hiddens: [Nb, 1, 512]
        outputs = self.linear(decoder_output.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary
        _, predicted = outputs.max(1)  # Nb
        decoded_batch[:, 0] = predicted

        decoder_input = torch.LongTensor([predicted for _ in range(batch_size)])

        for t in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_sample(embed_words, decoder_input.cuda(), decoder_hidden)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach()

        return decoded_batch

    def sample_beam_decode(self, step_enc, embed_words, decoder_hidden=None):
        '''
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :return: decoded_batch
        vocab('<start>') = 30168
        vocab('<end>') = 30169
        '''
        batch_size = 1
        MAX_LENGTH = 20
        EOS_token = 30169
        beam_width = 5
        topk = 10  # how many sentence do you want to generate

        features = self.linear_project(step_enc)  # [Nb, 256]
        decoder_input = features.unsqueeze(1)  # [Nb, 1, 256]
        decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)  # hiddens: [Nb, 1, 512]
        outputs = self.linear(decoder_output.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary
        _, predicted = outputs.max(1)  # Nb

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([predicted])

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, nds = nodes.get()
            decoder_input = torch.LongTensor([nds.wordid])
            decoder_hidden = nds.h

            if nds.wordid.item() == EOS_token and nds.prevNode != None:
                endnodes.append((score, nds))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = self.forward_sample(embed_words, decoder_input.cuda(), decoder_hidden)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k]
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, nds, decoded_t, nds.logp + log_p, nds.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        decoded_batch_all = []
        for score, nco in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(nco.wordid)
            # back trace
            while nco.prevNode != None:
                nco = nco.prevNode
                utterance.append(nco.wordid)

            utterance = utterance[::-1]
            decoded_batch_all.append(utterance)

        decoded_batch = torch.zeros((batch_size, min(MAX_LENGTH, len(decoded_batch_all[0]))))
        for klik in range(min(MAX_LENGTH, len(decoded_batch_all[0]))):
            decoded_batch[:, klik] = decoded_batch_all[0][klik].cuda()

        return decoded_batch


class EncoderRECIPE(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRECIPE, self).__init__()
        self.recipe_inDim = args.recipe_inDim
        self.recipe_hiddens = args.recipe_hiddens
        self.recipe_nlayers = args.recipe_nlayers

        self.lstm = nn.LSTM(self.recipe_inDim, self.recipe_hiddens, self.recipe_nlayers, batch_first=True)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.recipe_nlayers, bsz, self.recipe_hiddens).zero_()),
                Variable(weight.new(self.recipe_nlayers, bsz, self.recipe_hiddens).zero_()))

    def forward(self, ingredient_feats, recipes_v, rec_lens, use_teacherF):
        # ingredient_feats -> [N, 1, 1024]
        # recipes_v -> [N, rec_lens[0], 1024]
        # len(rec_lens -> N

        if use_teacherF == False:
            recipes_v_i = torch.cat((ingredient_feats, recipes_v), 1)  # [N, rec_lens[0] + 1, 1024]

            recipes_v_packed = pack_padded_sequence(recipes_v_i, rec_lens, batch_first=True)
            out, _ = self.lstm(recipes_v_packed)  # [0] --> [sum(rec_lens), 1024],  [1] --> [rec_lens[0]]

            from torch.nn.utils.rnn import pad_packed_sequence
            out2 = pad_packed_sequence(out, batch_first=True)[0]

        else:
            inputs = ingredient_feats  # [N, 1, 1024]
            states = self.init_hidden(recipes_v.shape[0])
            # [0]-> [1, len(rec_lens), 1024], [1] -> [1, len(rec_lens), 1024]

            sampled_instructions = []
            for di in range(rec_lens[0]):  # max recipe len.
                hiddens, states = self.lstm(inputs, states)
                if random.random() < (0.5):
                    inputs = hiddens  # [N, 1, 1024]
                else:
                    inputs = recipes_v[:, di, :].unsqueeze(1)  # [N, 1, 1024]
                sampled_instructions.append(hiddens.squeeze(1))

            sampled_instructions = torch.stack(sampled_instructions, 1)  # torch.Size([N, rec_lens[0], 1024])

            out = pack_padded_sequence(sampled_instructions, rec_lens, batch_first=True)
            # [0] --> [sum(rec_lens), 1024],  [1] --> [rec_lens[0]]

        return out[0]  # [sum(rec_lens), 1024]

    def sample_INGgiven(self, features, lenmax, states=None):
        # features --> [1,1024]

        sampled_instructions = []
        features = features.view(1, -1)  # [1,1024]
        inputs = features.unsqueeze(1)  # [1, 1, 1024] - [batch_size, 1, 1024]
        for i in range(lenmax):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [1, 1, 1024]  -> [batch_size, 1, hidden_size]
            outputs = hiddens.squeeze(1)  # outputs:  [1, 1024] -> (batch_size, feature_size)
            sampled_instructions.append(hiddens.squeeze(1))
            inputs = outputs.unsqueeze(1)

        sampled_instructions = torch.stack(sampled_instructions, 1)
        # [1, lenmax, 1024] -> (batch_size, lenmax, feature_size)
        return sampled_instructions

    def sample_nGiven(self, features, instructions, nGiven, maxnums, states=None):
        # features --> [1,1024]
        # instructions --> [maxnums,1024]

        sampled_instructions = []
        features = features.view(1, -1)  # [1,1024]
        inputs = features.unsqueeze(1)  # [1, 1, 1024] - [batch_size, 1, 1024]
        for i in range(maxnums):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [1, 1, 1024]  -> [batch_size, 1, hidden_size]
            outputs = hiddens.squeeze(1)  # outputs:  [1, 1024] -> (batch_size, feature_size)
            sampled_instructions.append(outputs)
            if i < nGiven:
                inputs = instructions[i, :].view(1, -1).unsqueeze(1)  # [1, 1, 1024]
            else:
                inputs = outputs.unsqueeze(1)

        sampled_instructions = torch.stack(sampled_instructions, 1)
        # [1, lenmax, 1024] -> (batch_size, lenmax, feature_size)
        return sampled_instructions

    def sample_nGiven_INF(self, features, instructions, states=None):
        # features --> [1,1024]
        # instructions --> [maxnums,1024]
        if len(instructions.shape) == 1:
            instructions = instructions.unsqueeze(0)

        features = features.view(1, -1)  # [1,1024]
        inputs = features.unsqueeze(1)  # [1, 1, 1024] - [batch_size, 1, 1024]
        maxnums = instructions.shape[0] + 1

        for i in range(maxnums):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [1, 1, 1024]  -> [batch_size, 1, hidden_size]
            outputs = hiddens.squeeze(1)  # outputs:  [1, 1024] -> (batch_size, feature_size)
            if i < instructions.shape[0]:
                inputs = instructions[i, :].view(1, -1).unsqueeze(1)  # [1, 1, 1024]

        return outputs

    def sample_caption(self, features, instructions, states=None):
        # features --> [1,1024]
        # instructions --> [maxnums,1024]
        if len(instructions.shape) == 1:
            instructions = instructions.unsqueeze(0)
        maxnums = instructions.shape[0]

        sampled_instructions = []
        features = features.view(1, -1)  # [1,1024]
        inputs = features.unsqueeze(1)  # [1, 1, 1024] - [batch_size, 1, 1024]
        for i in range(maxnums + 1):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [1, 1, 1024]  -> [batch_size, 1, hidden_size]
            sampled_instructions.append(hiddens.squeeze(1))
            if instructions.shape[0] > i:
                print(instructions.shape[0], i)
                inputs = instructions[i, :].view(1, -1).unsqueeze(1)  # [1, 1, 1024]

        sampled_instructions = torch.stack(sampled_instructions, 1)
        # [1, lenmax, 1024] -> (batch_size, lenmax, feature_size)
        return sampled_instructions
