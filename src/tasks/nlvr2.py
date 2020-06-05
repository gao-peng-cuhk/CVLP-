# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.nlvr2_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = NLVR2Dataset(splits)
    tset = NLVR2TorchDataset(dset)
    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class NLVR2:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.momentum = 0.9995
        self.model = NLVR2Model()
        self.siam_model = copy.deepcopy(self.model)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
            self.siam_model.lxrt_encoder.load(args.load_lxmert)

        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
            load_lxmert_qa(args.load_lxmert_qa, self.siam_model,
                       label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            self.siam_model.lxrt_encoder.multi_gpu()
        self.model = self.model.cuda()
        self.siam_model = self.siam_model.cuda()

        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def update_ema_variables(self):
        #        pdb.set_trace()
        # Use the true average until the exponential average is more correct
        alpha = self.momentum
        ema_model = self.siam_model
        model = self.model
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        # self.siam_model=ema_model


    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid1 = 0.
        best_valid2 = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
                self.model.train()

                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)

                loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                self.update_ema_variables()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score1, valid_score2 = self.evaluate(eval_tuple)
                if valid_score1 > best_valid1:
                    best_valid1 = valid_score1
                    self.save1("BEST")

                if valid_score2 > best_valid2:
                    best_valid2 = valid_score2
                    self.save2("BEST_siam")

                log_str += "Epoch %d: Valid1 %0.2f\n" % (epoch, valid_score1 * 100.) + \
                           "Epoch %d: Best1 %0.2f\n" % (epoch, best_valid1 * 100.)

                log_str += "Epoch %d: Valid2 %0.2f\n" % (epoch, valid_score2 * 100.) + \
                           "Epoch %d: Best2 %0.2f\n" % (epoch, best_valid2 * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save1("LAST1")
        self.save2("LAST2")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        self.siam_model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans1 = {}
        quesid2ans2 = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score1, predict1 = logit.max(1)
                for qid, l in zip(ques_id, predict1.cpu().numpy()):
                    quesid2ans1[qid] = l

        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit2 = self.siam_model(feats, boxes, sent)
                score2, predict2 = logit2.max(1)
                for qid, l in zip(ques_id, predict2.cpu().numpy()):
                    quesid2ans2[qid] = l

        if dump is not None:
            evaluator.dump_result(quesid2ans1, dump)
            evaluator.dump_result(quesid2ans2, dump)
        return quesid2ans1, quesid2ans2

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans1, quesid2ans2 = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans1), evaluator.evaluate(quesid2ans2)

    def save1(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def save2(self, name):
        torch.save(self.siam_model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    nlvr2 = NLVR2()

    # Load Model
    if args.load is not None:
        nlvr2.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'hidden' in args.test:
            nlvr2.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'hidden_predict.csv')
            )
        elif 'test' in args.test or 'valid' in args.test:
            result = nlvr2.evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, '%s_predict.csv' % args.test)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', nlvr2.train_tuple.dataset.splits)
        if nlvr2.valid_tuple is not None:
            print('Splits in Valid data:', nlvr2.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)


