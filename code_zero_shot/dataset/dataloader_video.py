# -*- coding: utf-8 -*-

import nltk
import torch
import torch.utils.data as data

from dataset.TastyVideos import TastyVideos


class TastyVideosDataset(data.Dataset):
    """TastyVideosDataset Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, args, vocab):
        self.tastyData = TastyVideos(args)
        self.ids = list(self.tastyData.ingredients.keys())
        self.vocab = vocab

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (ingredients and recipe)."""
        tastyData = self.tastyData
        vocab = self.vocab
        ann_id = self.ids[index]
        ingredients = torch.tensor(tastyData.ingredients[ann_id], dtype=torch.float)
        frame_feats = tastyData.frame_feats[ann_id]

        c_sentences = tastyData.sentences[ann_id]
        target_captions = []
        for x in range(0, len(c_sentences)):
            tokens = nltk.word_tokenize(c_sentences[x])
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target_captions.append(torch.Tensor(caption))

        return ingredients, target_captions, frame_feats


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (ingredients, recipes).
    Args:
        data: list of tuple (ingredients, recipes).
            - ingredients: torch tensor of shape
            - recipes: torch tensor of shape (?); variable length.
    Returns:
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    ingredients, target_captions, frame_feats = zip(*data)

    ingredients_v = torch.stack(ingredients, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in target_captions]

    list_of_sents = []
    list_sent_encoding = []
    counter = 0
    for i in range(len(target_captions)):
        cap = target_captions[i]
        counts = i
        for x in range(0, len(cap)):
            numinc = len([v for v in lengths if v > x])
            list_of_sents.append((cap[x], counts))
            counts = counts + numinc
            list_sent_encoding.append((cap[x], counter))
            counter = counter + 1

    list_of_sents.sort(key=lambda x: len(x[0]), reverse=True)
    captions, indices = zip(*list_of_sents)
    list_sent_encoding.sort(key=lambda x: len(x[0]), reverse=True)
    _, indices_encoder = zip(*list_sent_encoding)

    lengths_captions = [len(cap) for cap in captions]
    captions_v = torch.zeros(len(captions), max(lengths_captions)).long()
    for i, cap in enumerate(captions):
        end = lengths_captions[i]
        captions_v[i, :end] = cap[:end]

    new_st_list = []
    for i in range(len(frame_feats)):
        cap = frame_feats[i]
        for x in range(0, len(cap)):
            new_st_list.append(cap[x])

    lengths_frames_lstm = [cap.shape[0] for cap in new_st_list]
    STs_frames = torch.zeros(len(lengths_frames_lstm), max(lengths_frames_lstm), 2048).float()
    for i, cap in enumerate(new_st_list):
        end = lengths_frames_lstm[i]
        STs_frames[i, :end] = cap[:end]

    return ingredients_v, lengths, captions_v, torch.LongTensor(lengths_captions), \
           torch.LongTensor(indices), torch.LongTensor(indices_encoder), \
           STs_frames, torch.LongTensor(lengths_frames_lstm)


def get_loader(args, batch_size, vocab, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Tasty Videos dataset."""

    tastyVideos = TastyVideosDataset(args, vocab)
    data_loader = torch.utils.data.DataLoader(dataset=tastyVideos,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
