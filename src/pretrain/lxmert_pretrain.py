# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
from param import args
from pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining
import copy
from tensorboardX import SummaryWriter
from lxrt.modeling import ConstantReplacementScheduler, LinearReplacementScheduler
from copy import deepcopy
DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    tset = LXMERTTorchDataset(dset, topk)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)


train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
valid_batch_size = 2048 if args.multiGPU else 512
valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < args.obj_mask_rate:
            prob /= args.obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat)

    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA')


class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=train_tuple.dataset.answer_table.num_answers,
            big_or_distillation='big'
        )
        self.siam_model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=train_tuple.dataset.answer_table.num_answers,
            big_or_distillation='distillation'
        )

        self.siam_model.model_type = 'negative'
        # self.siam_model.big_or_distillation = 'distillation'
#        self.siam_model = LXRTPretraining.from_pretrained(
#            "bert-base-uncased",
#            task_mask_lm=args.task_mask_lm,
#            task_obj_predict=args.task_obj_predict,
#            task_matched=args.task_matched,
#            task_qa=args.task_qa,
#            visual_losses=args.visual_losses,
#            num_answers=train_tuple.dataset.answer_table.num_answers
#        )
        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
#            self.siam_model.apply(self.siam_model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)

        # GPU Options
        # self.siam_model = copy.deepcopy(self.model)
        # self.siam_model.model_type = 'negative'
        self.model = self.model.cuda()
        self.siam_model = self.siam_model.cuda()

        # pdb.set_trace()


        ####init self.model.module.bert.encoder.

        scc_n_layer_r = self.siam_model.bert.encoder.scc_n_layer_r  # 2
        scc_n_layer_l = self.siam_model.bert.encoder.scc_n_layer_l  # 4
        scc_n_layer_x = self.siam_model.bert.encoder.scc_n_layer_x  # 2

        # self.scc_layer_r = self.scc_layer_r.append(self.r_layers[4])
        # self.scc_layer_l = self.scc_layer_l.append(self.layer[8])
        # self.scc_layer_x = self.scc_layer_x.append(self.x_layers[4])

        # for i in range(scc_n_layer_r):
        #     alpha = 0.9
        #     ema_model = self.model.lxrt_encoder.model.bert.encoder.scc_layer_r[i]##0
        #     for ix in range(2):
        #         model = self.model.lxrt_encoder.model.bert.encoder.r_layers[i*2+ix]
        #         for ema_param, param in zip(ema_model.parameters(),
        #                                     model.parameters()):
        #             ema_param.data.mul_(alpha*ix).add_(1 - alpha, param.data)
        ### init
        self.siam_model.bert.encoder.scc_layer_r = nn.ModuleList(
            [deepcopy(self.siam_model.bert.encoder.r_layers[ix]) for ix in range(scc_n_layer_r)])
        self.siam_model.bert.encoder.scc_layer_r = self.siam_model.bert.encoder.scc_layer_r.append(
            deepcopy(self.siam_model.bert.encoder.r_layers[4]))

        # for i in range(scc_n_layer_l):
        #     alpha = 0.9
        #     ema_model = self.model.lxrt_encoder.model.bert.encoder.scc_layer_l[i]##0
        #     for ix in range(2):
        #         model = self.model.lxrt_encoder.model.bert.encoder.layer[i*2+ix]
        #         for ema_param, param in zip(ema_model.parameters(),
        #                                     model.parameters()):
        #             ema_param.data.mul_(alpha*ix).add_(1 - alpha, param.data)
        self.siam_model.bert.encoder.scc_layer_l = nn.ModuleList(
            [deepcopy(self.siam_model.bert.encoder.layer[ix]) for ix in range(scc_n_layer_l)])
        self.siam_model.bert.encoder.scc_layer_l = self.siam_model.bert.encoder.scc_layer_l.append(
            deepcopy(self.siam_model.bert.encoder.layer[8]))

        # for i in range(scc_n_layer_x):
        #     alpha = 0.5
        #     ema_model = self.model.lxrt_encoder.model.bert.encoder.scc_layer_x[i]##0
        #     for ix in range(2):
        #         model = self.model.lxrt_encoder.model.bert.encoder.x_layers[i*2+ix]
        #         for ema_param, param in zip(ema_model.parameters(),
        #                                     model.parameters()):
        #             ema_param.data.mul_(alpha*ix).add_(1 - alpha, param.data)
        self.siam_model.bert.encoder.scc_layer_x = nn.ModuleList(
            [deepcopy(self.siam_model.bert.encoder.x_layers[ix]) for ix in range(scc_n_layer_x)])
        # self.model.lxrt_encoder.model.bert.encoder.scc_layer_x = self.model.lxrt_encoder.model.bert.encoder.scc_layer_x.append(
        #      deepcopy(self.model.lxrt_encoder.model.bert.encoder.x_layers[3]))
        self.siam_model.bert.encoder.scc_layer_x = self.siam_model.bert.encoder.scc_layer_x.append(
            deepcopy(self.siam_model.bert.encoder.x_layers[4]))

        # pdb.set_trace()

        # self.siam_model = copy.deepcopy(self.model)
        # no_decay = ['bias', 'LayerNorm.weight']
        # self.ptimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_r.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_r.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_l.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_l.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_x.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_x.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.pooler.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.pooler.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.model.named_parameters() if
        #             not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.model.named_parameters() if
        #             any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        #     # {'params': [p for n, p in self.model.logit_fc.named_parameters() if
        #     #             any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]


        self.lang_queue = []
        self.visn_queue = []
        self.momentum = 0.99995
        self.queue_size = 30
        self.loss = nn.CrossEntropyLoss()

        if args.scheduler_type == 'none':
            replacing_rate_scheduler = ConstantReplacementScheduler(
                bert_encoder=self.model.lxrt_encoder.model.bert.encoder2,
                replacing_rate=args.replacing_rate,
                replacing_steps=args.steps_for_replacing)
        elif args.scheduler_type == 'linear':
            self.replacing_rate_scheduler = LinearReplacementScheduler(
                bert_encoder1=self.siam_model.bert.encoder,
                base_replacing_rate=args.replacing_rate,
                k=args.scheduler_linear_k)
            print("**********************线性********************")

        if args.multiGPU:
            self.model = nn.DataParallel(self.model)
            self.siam_model = nn.DataParallel(self.siam_model)

    def forward(self, examples):
#        pdb.set_trace()
#        batch = len(train_features)


        # value_1 = self.model.module.bert.encoder.layer[8].attention.self.query.weight[0][0]
        # value_2 = self.siam_model.module.bert.encoder.layer[8].attention.self.query.weight[0][0]
        # print(value_1, value_2)
        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        batch_size = len(train_features)
        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        feats_2 = torch.from_numpy(np.stack([f.obj_labels['feat'][0] for f in train_features])).cuda()
        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)
        #
        # train_features_2 = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
        #                   for example in examples]

        # language Inputs
        # input_ids_2 = torch.tensor([f.input_ids for f in train_features_2], dtype=torch.long).cuda()
        # input_mask_2 = torch.tensor([f.input_mask for f in train_features_2], dtype=torch.long).cuda()
        # segment_ids_2 = torch.tensor([f.segment_ids for f in train_features_2], dtype=torch.long).cuda()
        #
        # # Visual Inputs
        # # feats_2 = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features_2])).cuda()
        # pos_2 = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features_2])).cuda()
        #
        #
        # feats_2 = torch.from_numpy(np.stack([f.obj_labels['feat'][0] for f in train_features_2])).cuda()
        # # pdb.set_trace()
        # # Language Prediction
        # lm_labels_2 = torch.tensor([f.lm_label_ids for f in train_features_2], dtype=torch.long).cuda()

        # Visual Prediction
#        obj_labels = {}
#        for key in ('obj', 'attr', 'feat'):
#            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
#            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
#            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
#            obj_labels[key] = (visn_labels, visn_mask)
        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
#        pdb.set_trace()

#        lang_output, visn_output, pooled_output = self.model(
#            input_ids, segment_ids, input_mask, lm_labels,
#            feats, pos, obj_labels, matched_labels, ans
#        )

#        with torch.no_grad():
        lang_output_2, visn_output_2, pooled_output_2 = self.siam_model(
                  input_ids, segment_ids, input_mask, lm_labels,
                  feats_2, pos, obj_labels, matched_labels, ans
            )


#        lang_output_2 = lang_output_2.detach()
#        visn_output_2 = visn_output_2.detach()
#        pooled_output_2 = pooled_output_2.detach()
 
#        concat_2 = torch.cat([lang_output_2, visn_output_2], dim=1)

 

#        lang_output = lang_output.view(-1, 768).unsqueeze(1)
#        visn_output = visn_output.view(-1, 768).unsqueeze(1)
#        lang_output_2 = lang_output_2.view(-1, 768).unsqueeze(2)
#        visn_output_2 = visn_output_2.view(-1, 768).unsqueeze(2)

#        l_lang_pos = torch.bmm(lang_output, lang_output_2)
#        l_visn_pos = torch.bmm(visn_output, visn_output_2)


#        self.lang_queue.append(lang_output_2)
#        self.visn_queue.append(visn_output_2)
        if len(self.lang_queue) < self.queue_size:
           self.lang_queue.append(lang_output_2)
           self.visn_queue.append(visn_output_2)
           return [-2]
        if len(self.lang_queue) == self.queue_size:
            #queue=30
#           pdb.set_trace()
#           lang_output, visn_output, pooled_output = self.model(
#                        input_ids, segment_ids, input_mask, lm_labels,
#                        feats, pos, obj_labels, matched_labels, ans




#           self.lang_queue.pop(0)
#           self.visn_queue.pop(0)

#           lang_output = lang_output.squeeze(1)
#           visn_output = visn_output.squeeze(1)
#           lang_queue = torch.stack(self.lang_queue, dim=0)
#           visn_queue = torch.stack(self.visn_queue, dim=0)
#           lang_queue = lang_queue.view(-1, 768)
#           visn_queue = visn_queue.view(-1, 768)
#           lang_queue = lang_queue.transpose(1, 0)
#           visn_queue = visn_queue.transpose(1, 0)
#           l_lang_neg = torch.mm(lang_output, lang_queue)
#           l_visn_neg = torch.mm(visn_output, visn_queue)

#           lang_logits = torch.cat([l_lang_pos.squeeze(1), l_lang_neg], dim=1)
#           visn_logits = torch.cat([l_visn_pos.squeeze(1), l_visn_neg], dim=1)

           lang_labels = torch.zeros(lang_output_2.shape[0], dtype=torch.long).cuda()
           visn_labels = torch.zeros(visn_output_2.shape[0], dtype=torch.long).cuda()

           loss = self.model(
                        input_ids, segment_ids, input_mask, lm_labels,
                        feats, pos, obj_labels, matched_labels, ans,
                        lang_output_2=lang_output_2, visn_output_2=visn_output_2,
                        lang_queue=torch.stack(self.lang_queue, dim=0),
                        visn_queue=torch.stack(self.visn_queue, dim=0),
                        lang_labels=lang_labels,
                        visn_labels=visn_labels)
            #(Pdb) p lang_queue.shape
            #torch.Size([30, 1600, 128])  batchsize = 80

           self.lang_queue.pop(0)
           self.visn_queue.pop(0)
           # pdb.set_trace()
           self.lang_queue.append(lang_output_2)
           self.visn_queue.append(visn_output_2)

           return loss

#           lang_loss = self.loss(lang_logits, lang_labels)
#           visn_loss  = self.loss(visn_logits, visn_labels)
#           loss = lang_loss + visn_loss
#        lang_output = lang_output.view(-1, 768)
#        visn_output = visn_output.view(-1, 768)
#        lang_output_2 
#        visn_output_2   

        return -1 

    def train_batch(self, optim, batch):
        optim.zero_grad()
        loss = self.forward(batch)
#        pdb.set_trace()
        if len(loss)==1:
           return -2
        if args.multiGPU:
            loss = loss.mean()##每个gpu 都给一个均值
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        # nn.utils.clip_grad_norm_(self.siam_model.module.bert.encoder.scc_layer_r.parameters(), 1.)
        # nn.utils.clip_grad_norm_(self.siam_model.module.bert.encoder.scc_layer_l.parameters(), 1.)
        # nn.utils.clip_grad_norm_(self.siam_model.module.bert.encoder.scc_layer_x.parameters(), 1.)
        # nn.utils.clip_grad_norm_(self.siam_model.module.bert.pooler.parameters(), 1.)
        optim.step()
#        pdb.set_trace()
        self.update_ema_variables()
        return loss.item()


    def update_ema_variables(self):
#        pdb.set_trace()
    # Use the true average until the exponential average is more correct

        alpha = self.momentum
        ema_model = self.siam_model
        model = self.model

        # pdb.set_trace()


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_r[0].parameters(),
                            model.module.bert.encoder.r_layers[0].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_r[1].parameters(),
                                    model.module.bert.encoder.r_layers[2].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_r[2].parameters(),
                            model.module.bert.encoder.r_layers[4].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_l[0].parameters(),
                                model.module.bert.encoder.layer[0].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_l[1].parameters(),
                            model.module.bert.encoder.layer[2].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_l[2].parameters(),
                            model.module.bert.encoder.layer[4].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_l[3].parameters(),
                                model.module.bert.encoder.layer[6].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_l[4].parameters(),
                            model.module.bert.encoder.layer[8].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_x[0].parameters(),
                            model.module.bert.encoder.x_layers[0].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_x[1].parameters(),
                                model.module.bert.encoder.x_layers[2].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.scc_layer_x[2].parameters(),
                                model.module.bert.encoder.x_layers[4].parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)




        for ema_param, param in zip(ema_model.module.bert.encoder.r_layers.parameters(), model.module.bert.encoder.r_layers.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.layer.parameters(),
                                    model.module.bert.encoder.layer.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.x_layers.parameters(),
                                    model.module.bert.encoder.x_layers.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.pooler.parameters(),
                                    model.module.bert.pooler.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.cls.parameters(),
                                    model.module.cls.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.visual_embed_head.parameters(),
                                    model.module.visual_embed_head.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.lang_embed_head.parameters(),
                                    model.module.lang_embed_head.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.embeddings.parameters(),
                                    model.module.bert.embeddings.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for ema_param, param in zip(ema_model.module.bert.encoder.visn_fc.parameters(),
                                    model.module.bert.encoder.visn_fc.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)




        # for ema_param, param in zip(ema_model.logit_fc.parameters(),
        #                             model.logit_fc.parameters()):
        #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # alpha = self.momentum
        # ema_model = self.siam_model
        # model = self.model
        # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
#        pdb.set_trace()
        train_ld = train_tuple.loader

        # Optimizer
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)


        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_r.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_r.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_l.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_l.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_x.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.encoder.scc_layer_x.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in self.siam_model.bert.pooler.named_parameters() if
        #                 not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.siam_model.bert.pooler.named_parameters() if
        #                 any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        ###list(self.model.parameters())
        optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)


        # if args.scheduler_type == 'none':
        #     replacing_rate_scheduler = ConstantReplacementScheduler(
        #         bert_encoder=self.model.lxrt_encoder.model.bert.encoder2,
        #         replacing_rate=args.replacing_rate,
        #         replacing_steps=args.steps_for_replacing)
        # elif args.scheduler_type == 'linear':
        #     replacing_rate_scheduler = LinearReplacementScheduler(
        #         bert_encoder1=self.siam_model.bert,
        #         base_replacing_rate=args.replacing_rate,
        #         k=args.scheduler_linear_k)
        #     print("**********************线性********************")

        # Train


        global_step = 0
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            # Train
            self.model.train()


            total_loss = 0.


            # total_losses = 0.
            uid2ans = {}
            # writer = SummaryWriter('./llog')
            # a = epoch*len(train_ld)
            for batch in tqdm(train_ld, total=len(train_ld)):
               loss = self.train_batch(optim, batch)

               print(loss)
               if loss==-2:
                  continue
#                loss, losses, logit = self.train_batch(optim, batch)
               total_loss += loss

               self.replacing_rate_scheduler.step()  # Update replace rate scheduler

               global_step += 1

               # writer.add_scalar('inter', loss, a)
               # a += 1


#                total_losses += losses

            # for i, (name, param) in enumerate(self.model.named_parameters()):
            #     # if 'bn' not in name:
            #     writer.add_histogram(name, param, 0)
            #
            # total_norm=0
            # for p in self.model.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # print("total_norm")
            # print(total_norm)
            self.save("Epoch%02d" % (epoch+1))
#               if args.task_qa:
#                    score, label = logit.max(1)
#                    for datum, l in zip(batch, label.cpu().numpy()):
#                        uid = datum.uid
#                        ans = train_tuple.dataset.answer_table.id2ans(l)
#                        uid2ans[uid] = ans

#            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch))
#            losses_str = "The losses are "
#            for name, loss in zip(LOSSES_NAME, total_losses):
#                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
#            print(losses_str)
#            if args.task_qa:
#                train_tuple.evaluator.evaluate(uid2ans, pprint=True)

            # Eval
#            avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1)

            # Save
#            if avg_eval_loss < best_eval_loss:
#                best_eval_loss = avg_eval_loss
#                self.save("BEST_EVAL_LOSS")
#            self.save("Epoch%02d" % (epoch+1))

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}

        for i, batch in enumerate(eval_ld):
            loss, losses, logit = self.valid_batch(batch)
            total_loss += loss
            total_losses += losses
            if args.task_qa:
                score, label = logit.max(1)
                for datum, l in zip(batch, label.cpu().numpy()):
                    uid = datum.uid
                    ans = train_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans
            if i == iters:
                break

        print("The valid loss is %0.4f" % (total_loss / len(eval_ld)))
        losses_str = "The losses are "
        for name, loss in zip(LOSSES_NAME, total_losses / len(eval_ld)):
            losses_str += "%s: %0.4f " % (name, loss)
        print(losses_str)

        if args.task_qa:
            eval_tuple.evaluator.evaluate(uid2ans, pprint=True)

        return total_loss / len(eval_ld)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_LXRT.pth" % name))
        torch.save(self.siam_model.state_dict(),
                   os.path.join(args.output, "%s_siam_LXRT.pth" % name))

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)
        self.siam_model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    lxmert = LXMERT(max_seq_length=20)


    lxmert.train(train_tuple, valid_tuple)
