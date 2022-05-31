#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： JMXGODLZZ
# datetime： 2022/4/13 上午11:16 
# ide： PyCharm
import json

import os

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from collections import Counter
from tqdm import tqdm

categories = []
for x in range(1, 55):
    if x not in [27, 45]:
        categories.append(str(x))


def vote_merge(filedir_lst, vote_thresh, savepath):
    entities_map = {}
    fw = open(savepath, encoding='utf-8', mode='w')
    for filepath in filedir_lst:
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    for (st, ed) in spans:
                        entities_map.setdefault(text, [])
                        entities_map[text].append((et, st, ed, tp))

    for k, v in entities_map.items():
        vc = Counter(v)
        res = []
        for etinf, count in vc.items():
            if count >= vote_thresh:
                res.append(etinf)

        lj = {}
        lj['text'] = k
        lj['label'] ={}

        for etinf in res:
            et, st, ed, tp = etinf
            lj['label'].setdefault(tp, {})
            lj['label'][tp].setdefault(et, [])
            lj['label'][tp][et].append([st, ed])


        fw.write('{}\n'.format(json.dumps(lj, ensure_ascii=False)))

def logits_merge_BIO(filedir_lst, savepath, threshold=0):
    resdict = {}
    fw = open(savepath, encoding='utf-8', mode='w')
    leng = -1
    for filepath in filedir_lst:
        fr = open(filepath, encoding='utf-8')
        resdict[filepath] = fr

    for lid in tqdm(range(10000)):
        mscores = None
        mmapping = None
        mmtext = None
        for rid, (k, v) in enumerate(resdict.items()):
            line = v.readline()
            lj = json.loads(line)
            text = lj['text']
            scores = lj['scores']
            mapping = lj['mapping']
            if rid == 0:
                mscores = np.array(scores)
                mmapping = mapping
                mmtext = text
            else:
                mscores += np.array(scores)
        mscores /= len(resdict)
        entities = []
        escores = []

        for l, start, end in zip(*np.where(mscores > threshold)):
            entities.append(
                (mmapping[start][0], mmapping[end][-1], categories[l])
            )
            escores.append(mscores[l][start][end])
        if True:
            alles = zip(entities, escores)
            alles = sorted(alles, key=lambda item: item[1], reverse=True)
            newentities = []
            for ((ns, ne, t), so) in alles:
                for (ts, te, _) in newentities:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        # for both nested and flat ner no clash is allowed
                        break
                    if ns <= ts <= te <= ne or ts <= ns <= ne <= te:
                        # for flat ner nested mentions are not allowed
                        break
                else:
                    newentities.append((ns, ne, t))
            entities = newentities

        labels = []
        for _ in mmtext:
            labels.append('O')
        for start, end, label in entities:
            for pid in range(start, end + 1):
                if pid == start:
                    labels[pid] = 'B-{}'.format(label)
                else:
                    labels[pid] = 'I-{}'.format(label)

        for (tw, tl) in zip(mmtext, labels):
            fw.write('{} {}\n'.format(tw, tl))
        fw.write('\n')


def model_vote_logits(in_file, out_file, threshold=0, rawmodellst=[], lessmodellst=[], duplessmodellst=[]):
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['text']
            l['label'] = {}
            words_less = kgsearch('./KG5_true_less/jdall.pickle', l['text'])
            words_dup_less = kgsearch('./KG5_true_dup_less/jdall.pickle', l['text'])
            labels = []
            for _ in l['text']:
                labels.append('O')

            mscores = None
            mmapping = None
            sflag = 0
            for rawmodel in rawmodellst:
                GPmodel.load_weights(rawmodel)
                scores, mapping = NER.recognize_gplogits(GPmodel, l['text'])
                if sflag == 0:
                    mscores = np.array(scores)
                    mmapping = mapping
                else:
                    mscores += np.array(scores)
                sflag += 1

            for lessmodel in lessmodellst:
                model.load_weights(lessmodel)
                scores, mapping = NER.recognize_gpkglogits(model, l['text'], words_less, is_flat)
                if sflag == 0:
                    mscores = np.array(scores)
                    mmapping = mapping
                else:
                    mscores += np.array(scores)
                sflag += 1

            for duplessmodel in duplessmodellst:
                model.load_weights(duplessmodel)
                scores, mapping = NER.recognize_gpkglogits(model, l['text'], words_dup_less, is_flat)
                if sflag == 0:
                    mscores = np.array(scores)
                    mmapping = mapping
                else:
                    mscores += np.array(scores)
                sflag += 1

            mscores /= sflag
            entities = []
            escores = []

            for l, start, end in zip(*np.where(mscores > threshold)):
                entities.append(
                    (mmapping[start][0], mmapping[end][-1], categories[l])
                )
                escores.append(mscores[l][start][end])
            if True:
                alles = zip(entities, escores)
                alles = sorted(alles, key=lambda item: item[1], reverse=True)
                newentities = []
                for ((ns, ne, t), so) in alles:
                    for (ts, te, _) in newentities:
                        if ns < ts <= ne < te or ts < ns <= te < ne:
                            # for both nested and flat ner no clash is allowed
                            break
                        if ns <= ts <= te <= ne or ts <= ns <= ne <= te:
                            # for flat ner nested mentions are not allowed
                            break
                    else:
                        newentities.append((ns, ne, t))
                entities = newentities

            for start, end, label in entities:
                for pid in range(start, end + 1):
                    if pid == start:
                        labels[pid] = 'B-{}'.format(label)
                    else:
                        labels[pid] = 'I-{}'.format(label)

            for (tw, tl) in zip(text, labels):
                fw.write('{} {}\n'.format(tw, tl))
            fw.write('\n')

    fw.close()






# def logits_merge_BIO_back(filedir_lst, savepath, threshold=0):
#     resdict = {}
#     fw = open(savepath, encoding='utf-8', mode='w')
#     for filepath in filedir_lst:
#         lines = open(filepath, encoding='utf-8').readlines()
#         for line in lines:
#             lj = json.loads(line)
#             text = lj['text']
#             scores = lj['scores']
#             mapping = lj['mapping']
#
#
#     for text, scores in resdict.items():
#         entities = []
#         escores = []
#         mapping =
#         for l, start, end in zip(*np.where(scores > threshold)):
#             entities.append(
#                 (mapping[start][0], mapping[end][-1], categories[l])
#             )
#             escores.append(scores[l][start][end])
#         if True:
#             alles = zip(entities, escores)
#             alles = sorted(alles, key=lambda item: item[1], reverse=True)
#             newentities = []
#             for ((ns, ne, t), so) in alles:
#                 for (ts, te, _) in newentities:
#                     if ns < ts <= ne < te or ts < ns <= te < ne:
#                         # for both nested and flat ner no clash is allowed
#                         break
#                     if ns <= ts <= te <= ne or ts <= ns <= ne <= te:
#                         # for flat ner nested mentions are not allowed
#                         break
#                 else:
#                     newentities.append((ns, ne, t))
#             entities = newentities

# vote_merge([
#     '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-pseudo4W-nezha-traintest2-epo5-8e-6.txt',
#     '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-pseudo4W-nezha-traintest2-epo5-8e-6-1.txt',
#     '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-pseudo4W-nezha-traintest2-epo5-8e-6-2.txt',
#     '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-pseudo4W-nezha-traintest2-epo5-8e-6-3.txt',
#     '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-pseudo4W-nezha-traintest2-epo5-8e-6-4.txt',
#     # '/home/root1/lizheng/workspace/2022/back/GlobalPointer-finetuning2stage-fgmswa/pseudo/testA_GP-distillV2-unlabel4W-nezha-traintest2-epo5.txt'
# ], 2, '../dataSummary/pseudo/testAclue-pseudo6_4-0420.json')

logits_merge_BIO([
    './submit/testA_GP-KGtrueless128-pseudo4W-direct-8e-6-fixdecode-all-logits-0.txt',
    './submit/testA_GP-KGtrueless128-pseudo4W-direct-8e-6-fixdecode-all-logits-1.txt',
    './submit/testA_GP-pseudo4W-direct-8e-6-fixdecode-all-logits.txt',
],
'./testA_GP-pseudo4W-direct-nezha-traintest2-epo5-8e-6-logits-merge.txt')