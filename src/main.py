import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import logging
import torch
import logging
import os
import time
import argparse
from tqdm import tqdm
from torchviz import make_dot
import sklearn.metrics as evaluate
from typing import Tuple

from data import *
from early import EarlyStopping
import models


def init_logger():
    """Initialize logger to save states when program running as `.log` files"""
    log_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/logs/'
    log_name = log_path + log_time + '.log'
    log_fmt = '%(asctime)s: %(filename)s[line: %(lineno)d] - %(levelname)s: %(message)s'
    logging.basicConfig(filename=log_name, filemode='w',
                        format=log_fmt, level=logging.INFO)


def init_argparser():
    """Initialize argument parser"""
    logging.info('Initialize argument parser')
    parser = argparse.ArgumentParser(description='Sentiment Classification')
    parser.add_argument('--version', '-v',
                        action='version',
                        version='%(prog)s version: v0.01',
                        help='show the version')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        default=False,
                        help='debug')
    parser.add_argument('--clean',
                        default='logs/*.log',
                        help='clean log files')
    parser.add_argument('--threads', '-t',
                        type=int, default=8,
                        help='thread number for <class DataLoader>')
    parser.add_argument('--seed', '-s',
                        type=int, default=13,
                        help='random seed initialization')
    parser.add_argument('--model', '-m',
                        type=str, default='RNN',
                        help='neraul network model name')
    parser.add_argument('--epoch', '-e',
                        type=int, default=20,
                        help='train data epoch')
    parser.add_argument('--bs', '-b',
                        type=int, default=64,
                        help='batch size')
    parser.add_argument('--padlen',
                        type=int, default=256,
                        help='max length of a sentence embedding')
    parser.add_argument('--mincount',
                        type=int, default=1,
                        help='minimum count of Word2vec model in <class PreDataset>')
    parser.add_argument('--vsize',
                        type=int, default=300,
                        help='vector size of Word2vec model in <class PreDataset>')
    parser.add_argument('--lr',
                        type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--train',
                        type=str, default='isear/isear_train.csv',
                        help='data source for training')
    parser.add_argument('--mode',
                        type=str, default='train',
                        help='choose mode')
    parser.add_argument('--hidden',
                        type=int, default=128,
                        help='size of hidden layers')
    parser.add_argument('--dropout',
                        type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--patience', default=5,
                        type=int, help='patience in early stopping')
    return parser.parse_args()


def init_seed(seed):
    """Setup seed in torch engine and random funtions"""
    logging.info('Initialize random seed as {}'.format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    model: nn.Module,
    train_data: DataLoader,
    valid_data: DataLoader,
    epoch: int,
    patience: int = 5,
    path: str = '../models/model.pt'
) -> Tuple[nn.Module, int, list, list]:
    """
    Args
    ---
        model (nn.Module): train target neural network model
        train_data (DataLoader): train data loaded from DataLoader as batches 
        valid_data (DataLoader): validation data loaded from DataLoader as batches
        epoch (int): max epoch to train
        patience (int): wait for several epochs before early stopping
        path (str): target directory to save model as dict

    Example
    ---
        >>> train_data_loader = DataLoader(dataset=train_dataset,
                                            num_workers=8,
                                            batch_size=64,
                                            shuffle=True,
                                            drop_last=True)
        >>> valid_data_loader = DataLoader(dataset=valid_dataset,
                                            num_workers=8,
                                            batch_size=64,
                                            shuffle=False,
                                            drop_last=False) 
        >>> train(CNN, train_data_loader, valid_data_loader, 20, 3, 'model.pt')
        model, stop, train_loss_list, valid_loss_list

    """
    def train_base(data: DataLoader, epoch: int) -> Tuple[float, list]:
        model.train()
        loss_sum = 0
        loss_list = []
        for _, (input_list, input_len, label) in tqdm(enumerate(data)):
            optimizer.zero_grad()
            input_list = Variable(input_list.cuda())
            input_len = Variable(input_len.cuda())
            label = Variable(label.cuda())
            output = model(input_list, input_len)
            loss = criterion(input=output, target=label)
            loss_sum += loss.item()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Epoch [{}] Loss {:.4f}\r'.format(epoch,  loss_sum))
        logging.info('Train epoch [{}] successfully with loss [{}] in total'
                     .format(epoch, loss_sum))
        return loss_sum, loss_list

    def valid_base(data: DataLoader, epoch: int) -> list:
        model.eval()
        loss_list = []
        for _, (input_list, input_len, label) in tqdm(enumerate(data)):
            input_list = Variable(input_list.cuda())
            input_len = Variable(input_len.cuda())
            label = Variable(label.cuda())
            output = model(input_list, input_len)
            loss = criterion(input=output, target=label)
            loss_list.append(loss.item())
        return loss_list

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    avg_train_loss_list = []
    avg_valid_loss_list = []
    stop = 0
    for epoch_ in range(1, epoch + 1):
        _, train_loss_list = train_base(train_data, epoch_)
        train_loss = np.average(train_loss_list)
        avg_train_loss_list.append(train_loss)

        valid_loss_list = valid_base(valid_data, epoch_)
        valid_loss = np.average(valid_loss_list)
        avg_valid_loss_list.append(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            stop = epoch_
            print('Early stop in epoch [{}]'.format(epoch_))
            break

    model.load_state_dict(torch.load(path))
    return model, stop, avg_train_loss_list, avg_valid_loss_list


def test(model: nn.Module, data: DataLoader) -> float:
    model.eval()
    output_list = []
    target_list = []
    with torch.no_grad():
        for i, (input_list, input_len, label) in tqdm(enumerate(data)):
            input_list = Variable(input_list.cuda())
            input_len = Variable(input_len.cuda())
            label = Variable(label.cuda())
            output = model(input_list, input_len)
            output = torch.max(output, 1)[1]
            output_list.extend(np.array(output.cpu()))
            target_list.extend(np.array(label.cpu()))
    accuracy = evaluate.accuracy_score(target_list, output_list)
    f1_macro = evaluate.f1_score(target_list, output_list, average='macro')
    f1_micro = evaluate.f1_score(target_list, output_list, average='micro')
    f1_weight = evaluate.f1_score(target_list, output_list, average='weighted')
    print('''Test finished successfully. Accuracy score: ({:.4f}). F-score macro: ({:.4f}). F-score micro: ({:.4f}). F-score weighted ({:.4f})'''
          .format(accuracy, f1_macro, f1_micro, f1_weight))


if __name__ == '__main__':
    init_logger()
    options = init_argparser()
    init_seed(options.seed)

    if torch.cuda.is_available():
        print('Cuda is available')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # text8 pretrain model: worse preformance compared to Google-vectors-negative300
    # pre_train_data = PreTrain(
    #     'pretrain/text8', 'pretrain/word2vec.bin', vector_size=options.vsize, min_count=options.mincount).model
    pre_train_data = KeyedVectors.load_word2vec_format('../pretrain/GoogleNews-vectors-negative300.bin',
                                                       binary=True)
    model = getattr(models, options.model)(input_size=pre_train_data.vector_size,
                                           pad_len=options.padlen,
                                           hidden_size=options.hidden,
                                           dropout_rate=options.dropout).to(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)

    if options.mode == 'train':
        train_data = PreDataset(pre_train=pre_train_data,
                                target_dir='../isear/isear_train.csv',
                                pad_len=options.padlen)
        train_data_loader = DataLoader(dataset=train_data,
                                       num_workers=options.threads,
                                       batch_size=options.bs,
                                       shuffle=True,
                                       drop_last=True)
        print("Train Data: {}".format(len(train_data)))

        valid_data = PreDataset(pre_train=pre_train_data,
                                target_dir='../isear/isear_valid.csv',
                                pad_len=options.padlen)
        valid_data_loader = DataLoader(dataset=valid_data,
                                       num_workers=options.threads,
                                       batch_size=options.bs,
                                       shuffle=False,
                                       drop_last=False)
        print("Valid Data: {}".format(len(valid_data)))

        _, stop, train_loss, valid_loss = train(model=model,
                                          train_data=train_data_loader,
                                          valid_data=valid_data_loader,
                                          epoch=options.epoch,
                                          patience=options.patience,
                                          path='../models/{}_model.pt'.format(options.model))
        # plot loss changing curve
        plt.plot([epoch for epoch in range(1, stop + 1)],
                 train_loss, label='Train Loss')
        plt.plot([epoch for epoch in range(1, stop + 1)],
                 valid_loss, label='Valid Loss')
        plt.legend()
        plt.savefig('../asset/{}_loss.png'.format(options.model), format='png')
        
    elif options.mode == 'test':
        test_data = PreDataset(pre_train=pre_train_data,
                               target_dir='../isear/isear_test.csv',
                               pad_len=options.padlen)
        test_data_loader = DataLoader(dataset=test_data,
                                      num_workers=options.threads,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False)
        print("Test Data: {}".format(len(test_data)))

        # load from existing models before test
        model.load_state_dict(torch.load(
            "../models/{}_model.pt".format(options.model)))
        test(model, test_data_loader)

        # generate state transformation image as visual model
        g = make_dot(model(Variable(torch.rand(1, 256, 300).cuda()), Variable(torch.tensor([256]).cuda())),
                     params=dict(model.named_parameters()))
        g.render('../asset/{}_model'.format(options.model),
                 view=False, format='png')
