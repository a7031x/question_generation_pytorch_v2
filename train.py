from data import Dataset, TrainFeeder, align2d, align3d
from model import Model
import numpy as np
import os
import config
import torch
import utils

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def print_prediction(feeder, similarity, pids, qids, labels, number=None):
    if number is None:
        number = len(pids)
    for k in range(min(len(pids), number)):
        pid, qid, sim, lab = pids[k], qids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        print(passage)
        if isinstance(lab, list):
            questions = [feeder.ids_to_sent(q) for q in qid]
            for q,s,l in zip(questions, sim, lab):
                if q:
                    print(' {} {:>.4F}: {}'.format(l, s, q))
        else:
            question = feeder.ids_to_sent(qid)
            print(' {} {:>.4F}: {}'.format(lab, sim, question))


def run_epoch(model, feeder, criterion, optimizer, batches):
    nbatch = 0 
    while nbatch < batches:
        pids, qids, labels, _ = feeder.next()
        nbatch += 1
        x = torch.tensor(pids).cuda()
        y = torch.tensor(qids).cuda()
        logit = model(x, y)
        loss = criterion(logit, y) / np.sum(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, loss.logit()))
    return loss


def train(auto_stop, steps=50, threshold=0.2):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    model = Model(len(dataset.ci2n)).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        feeder.load_state(ckpt['feeder'])
    loss = 1
    while True:
        #run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.2, 100)
        loss = run_epoch(model, feeder, criterion, optimizer, steps)
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'model':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'feeder': feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)