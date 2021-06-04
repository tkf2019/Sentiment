import re
import os
import torch
import logging
import pandas as pd
import numpy as np
import torch.utils.data as data
from gensim.models import word2vec, KeyedVectors
from typing import *

CLASS = 7
word2label = {
    'shame': 0,
    'anger': 1,
    'fear': 2,
    'disgust': 3,
    'joy': 4,
    'sadness': 5,
    'guilt': 6
}


class PreTrain(object):
    def __init__(self, target_dir: str, model_name: str, vector_size: int, min_count: int) -> None:
        super(PreTrain, self).__init__()
        sentences = word2vec.Text8Corpus(target_dir)
        self.model = word2vec.Word2Vec(size=vector_size, min_count=min_count)
        if not os.path.exists(model_name):
            logging.info(
                'Build and save Word2vec pre-train model in <class Word2vec>')

            self.model.build_vocab(sentences=sentences)
            self.model.train(sentences=sentences,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.epochs)
            self.model.wv.save_word2vec_format(model_name, binary=True)
        else:
            logging.info(
                'Load existing Word2vec pre-train model in <class Word2vec>')
            self.model = KeyedVectors.load_word2vec_format(
                model_name, binary=True)


class PreDataset(data.Dataset):
    """ISEAR Data pre-process"""

    def __init__(self, pre_train: word2vec.Word2Vec, target_dir: str, pad_len: int) -> None:
        """
        Args
        ---
        pre_train (Word2Vec): word2vec pretrain model imported from text8, Google etc.
        target_dir (str): provided `.csv` file directory
        pad_len (int): maximum padding length for sentences
        """
        super(PreDataset, self).__init__()
        self.list = []
        self.pad_len = pad_len
        self.model = pre_train
        self.fq = np.zeros(CLASS, np.int32)
        self.ave_len = 0
        csv_data = pd.read_csv(target_dir, encoding='utf-8')
        for i, line in csv_data.iterrows():
            sentence = [''.join(re.findall(re.compile('\w'), word))
                        for word in line['sentence'].lower().split(' ')
                        if word != '' and word != 'á']
            label = word2label[line['label']]
            self.fq[label] += 1
            self.ave_len += len(sentence)
            self.list.append([sentence, label])
        self.ave_len = self.ave_len // len(self.list) + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        input_list = []
        sentence, label = self.list[index]
        for word in sentence:
            try:
                input_list.append(self.model[word])
            except KeyError:
                input_list.append(np.zeros(self.model.vector_size, np.float32))
        if len(input_list) > self.pad_len:
            input_list = input_list[:self.pad_len]
        input_list_len = len(input_list)
        input_list = torch.cat([torch.FloatTensor(input_list), torch.FloatTensor(
            [np.zeros(self.model.vector_size, np.float32)] * (self.pad_len - len(input_list)))])
        # input_list = F.pad(input=torch.tensor(input_list),
        #                    pad=(0, 0, 0, self.pad_len - input_list_len),
        #                    mode='constant', value=0)
        return input_list, input_list_len, label

    def __len__(self):
        return len(self.list)
