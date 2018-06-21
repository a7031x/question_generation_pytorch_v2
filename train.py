from data import Dataset, TrainFeeder, align2d, align3d
from model import Model
from collections import defaultdict
import numpy as np
import os
import config
import torch
import utils

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def tensor(v):
    return torch.tensor(v).cuda()


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


def run_epoch(model, feeder, optimizer, batches):
    nbatch = 0
    char_weights = tensor(feeder.dataset.norm_char_weights)
    weight = 1 / (char_weights + char_weights.max() / 3).float()
    criterion = torch.nn.NLLLoss(size_average=False, reduce=False, weight=weight)
    sm = torch.nn.LogSoftmax(dim=-1)
    while nbatch < batches:
        pids, qids, _,  _ = feeder.next()
        nbatch += 1
        x = tensor(pids)
        y = tensor(qids)
        logit = model(x, y)
        sm_logit = sm(logit).transpose(1,3).transpose(2,3)
        mask = (tensor(qids)!=config.NULL_ID).float()#[batch, num_questions, ids]
        logit_ids = logit.max(-1)[1].tolist()#[batch, num_questions, ids]
      #  logit_counter = torch.nn.functional.softmax(logit, dim=-1).sum(-2).cpu().detach().numpy()#[batch, num_questions, sum_sm_logit]
        logit_weight = np.ndarray(mask.shape, dtype=np.float32)
        for idx_batch in range(mask.shape[0]):
            for idx_question in range(mask.shape[1]):
                id_counter = defaultdict(lambda: 0)
                for idx_char in range(mask.shape[2]):
                    id = logit_ids[idx_batch][idx_question][idx_char]
                    id_counter[id] += 1
                for idx_char in range(mask.shape[2]):
                    id = logit_ids[idx_batch][idx_question][idx_char]
                    tid = qids[idx_batch][idx_question][idx_char]
                    if tid == config.NULL_ID:
                        w = 0
                    elif id != tid:
                        w = id_counter[id]
                    else:
                        w = 1
                    logit_weight[idx_batch, idx_question, idx_char] = w

        #mask[:,:,0] = 0.2
        loss = (criterion(sm_logit, y) * tensor(logit_weight)).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))
        if nbatch % 10 == 0:
            logit = model(x, None)
            gids = logit.argmax(-1).tolist()
            for k in range(len(gids)):
                question = feeder.ids_to_sent(gids[k])
                print('truth:   {}'.format(feeder.ids_to_sent(qids[k][0])))
                print('predict: {}'.format(question))
                print('----------')
    return loss


def train(auto_stop, steps=50, threshold=0.2):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    model = Model(len(dataset.ci2n)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        feeder.load_state(ckpt['feeder'])
    while True:
        #run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.2, 100)
        run_epoch(model, feeder, optimizer, steps)
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'model':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'feeder': feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)