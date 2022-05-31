import json

import matplotlib.pyplot as plt
from seqeval.metrics.sequence_labeling import classification_report, get_entities
import os
import tqdm
import pickle

def trans2view(text, label):
    nstr = ''
    entities = get_entities(label)
    pref = 0
    while len(entities) > 0:
        entity = entities.pop(0)
        nstr += text[pref: entity[1]]
        nstr += '[{}{}{}]'.format(entity[0], text[entity[1]: entity[2] + 1], entity[0])
        pref = entity[2] + 1
    return nstr

class dataParser():
    def __init__(self):
        pass

    def readfile(self, filepath):
        paras = open(filepath, encoding='utf-8').read().split('\n\n')
        datas = []
        labels = []
        for para in tqdm.tqdm(paras):
            sentences = para.split('\n')
            data = []
            label = []
            for sentence in sentences:
                if len(sentence) == 0:
                    continue
                char = sentence[0]
                bz = sentence[2:]
                data.append(char)
                label.append(bz)
            datas.append(data)
            labels.append(label)
        return datas, labels

    def trans2clue(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        datas, labels = self.readfile(filepath)

        for (data, label) in zip(datas, labels):
            inf = {}
            text = ''.join(data)
            inf['text'] = text
            inf['label'] = {}
            entities = get_entities(label)
            for entity in entities:
                tp, st, ed = entity
                et = text[st: ed + 1]
                inf['label'].setdefault(tp, {})
                inf['label'][tp].setdefault(et, [])
                inf['label'][tp][et].append([st, ed])
            fw.write('{}\n'.format(json.dumps(inf, ensure_ascii=False)))



    def countShow(self, dct, filepath):
        '''
        将记录数量的字典可视化
        :param dct:
        :param filepath:
        :return:
        '''
        dct = sorted(dct.items(), key=lambda item: item[0])
        x = [item[0] for item in dct]
        y = [item[1] for item in dct]
        fig = plt.figure(figsize=(15, 15))
        plt.bar(x, y)
        if 'label' in filepath:
            plt.xticks(range(55))
        plt.savefig(filepath)

    def showInf(self, filepath, savedir='./summary'):
        datas, labels = self.readfile(filepath)
        etlabel = {} # 记录每个实体的标签
        etleng = {} # 记录实体长度分布
        # etcount = {} # 记录实体类别数量分布
        sentleng = {}# 记录句子长度分布
        tpdetail = {}# 记录具体类别的实体
        tpdetailsentence = {} # 记录具体类别实体的句子
        tpcount = {} # 记录类别数量
        dupdata = {}
        for (data, label) in zip(datas, labels):
            entities = get_entities(label)
            sentl = len(label)
            sentleng.setdefault(sentl, 0)
            sentleng[sentl] += 1

            for et in entities:
                tp, st, ed = et
                tp = int(tp)
                word = ''.join(data[st: ed + 1])
                etl = len(word)
                etlabel.setdefault(word, {})
                etlabel[word].setdefault(tp, 0)
                etlabel[word][tp] += 1
                etleng.setdefault(etl, 0)
                etleng[etl] += 1
                tpdetail.setdefault(tp, set())
                tpdetail[tp].add(word)
                tpdetailsentence.setdefault(tp, {})
                tpdetailsentence[tp].setdefault(word, set())

                tstr = trans2view(''.join(data), label)

                tpdetailsentence[tp][word].add(tstr)
                tpcount.setdefault(tp, 0)
                tpcount[tp] += 1

        # self.countShow(etcount, os.path.join(savedir, 'entity_count.png'))
        self.countShow(etleng, os.path.join(savedir, 'entity_length.png'))
        self.countShow(sentleng, os.path.join(savedir, 'sentence_count.png'))
        self.countShow(tpcount, os.path.join(savedir, 'label_count.png'))

        tpdetailcount = {}
        lesstpfw = open(os.path.join(savedir, 'less_entities.txt'), encoding='utf-8', mode='w')
        lessets = []

        tpdetailstfw = open(os.path.join(savedir, 'label_entities_detail.txt'), encoding='utf-8', mode='w')

        for k, v in tpdetail.items():
            tpdetailcount[k] = len(v)
            if len(v) < 500:
                lessets.append(k)
                lesstpfw.write('类型:{},数量:{}\n'.format(k, len(v)))
                for w in v:
                    lesstpfw.write('{}\n'.format(w))
                lesstpfw.write('\n')


            tpdetailstfw.write('类型:{},数量:{}\n'.format(k, len(v)))
            for w in v:
                sentence = list(tpdetailsentence[k][w])[:5]
                tpdetailstfw.write('{}\n{}\n\n'.format(w, '\n'.join(sentence)))
            tpdetailstfw.write('\n')

        self.countShow(tpdetailcount, os.path.join(savedir, 'labelset_count.png'))

        dupfw = open(os.path.join(savedir, 'dupentities.txt'), encoding='utf-8', mode='w')
        duppickle = open(os.path.join(savedir, 'dupet.pickle'), mode='wb')

        for k, v in etlabel.items():
            if len(v) > 1:
                dupfw.write('{}\t{}\n'.format(k, v))

        etlabelfw = open(os.path.join(savedir, 'jdkg.pickle'), mode='wb')
        pickle.dump(etlabel, etlabelfw)
        print(lessets)

    def showClueInf(self, filepath, savedir='./summary'):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        etlabel = {} # 记录每个实体的标签
        etleng = {} # 记录实体长度分布
        # etcount = {} # 记录实体类别数量分布
        sentleng = {}# 记录句子长度分布
        tpdetail = {}# 记录具体类别的实体
        tpdetailsentence = {} # 记录具体类别实体的句子
        tpcount = {} # 记录类别数量
        dupdata = {}
        labelentity = {}
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm.tqdm(lines):
            lj = json.loads(line)
            data = lj['text']
            labels = lj['label']
            entities = []
            for tp, etdict in labels.items():
                for et, spans in etdict.items():
                    for (st, ed) in spans:
                        entities.append([tp, st, ed])
            # entities = get_entities(label)
            sentl = len(data)
            sentleng.setdefault(sentl, 0)
            sentleng[sentl] += 1

            for et in entities:
                tp, st, ed = et
                tp = int(tp)
                word = ''.join(data[st: ed + 1])
                etl = len(word)
                etlabel.setdefault(word, {})
                etlabel[word].setdefault(tp, 0)
                etlabel[word][tp] += 1
                etleng.setdefault(etl, 0)
                etleng[etl] += 1
                tpdetail.setdefault(tp, set())
                tpdetail[tp].add(word)
                tpdetailsentence.setdefault(tp, {})
                tpdetailsentence[tp].setdefault(word, set())

                # tstr = trans2view(''.join(data), label)
                tstr = data

                tpdetailsentence[tp][word].add(tstr)
                tpcount.setdefault(tp, 0)
                tpcount[tp] += 1

        # self.countShow(etcount, os.path.join(savedir, 'entity_count.png'))
        self.countShow(etleng, os.path.join(savedir, 'entity_length.png'))
        self.countShow(sentleng, os.path.join(savedir, 'sentence_count.png'))
        self.countShow(tpcount, os.path.join(savedir, 'label_count.png'))

        tpdetailcount = {}
        lesstpfw = open(os.path.join(savedir, 'less_entities.txt'), encoding='utf-8', mode='w')
        lessets = []

        tpdetailstfw = open(os.path.join(savedir, 'label_entities_detail.txt'), encoding='utf-8', mode='w')

        for k, v in tpdetail.items():
            tpdetailcount[k] = len(v)
            if len(v) < 500:
                lessets.append(k)
                lesstpfw.write('类型:{},数量:{}\n'.format(k, len(v)))
                for w in v:
                    lesstpfw.write('{}\n'.format(w))
                lesstpfw.write('\n')


            tpdetailstfw.write('类型:{},数量:{}\n'.format(k, len(v)))
            for w in v:
                sentence = list(tpdetailsentence[k][w])[:5]
                tpdetailstfw.write('{}\n{}\n\n'.format(w, '\n'.join(sentence)))
            tpdetailstfw.write('\n')

        self.countShow(tpdetailcount, os.path.join(savedir, 'labelset_count.png'))

        dupfw = open(os.path.join(savedir, 'dupentities.txt'), encoding='utf-8', mode='w')
        duppickle = open(os.path.join(savedir, 'dupentity.pickle'), mode='wb')
        for k, v in etlabel.items():
            for tp, ct in v.items():
                labelentity.setdefault(tp, {})
                labelentity[tp][k] = ct
            if len(v) > 1:
                dupfw.write('{}\t{}\n'.format(k, v))
                dupdata[k] = v
        pickle.dump(dupdata, duppickle)

        labelentityfw = open(os.path.join(savedir, 'labelentity.pickle'), mode='wb')
        pickle.dump(labelentity, labelentityfw)

        etlabelfw = open(os.path.join(savedir, 'jdkg.pickle'), mode='wb')
        pickle.dump(etlabel, etlabelfw)
        print(lessets)

dp = dataParser()
# dp.showInf('../data/train_data/train.txt')
# dp.showInf('/home/root1/lizheng/workspace/2022/京东NER/GlobalPointer-main/data/JD20/JDtrain0.txt',
#            savedir='./summary20')
# dp.trans2clue('../data/train_data/train.txt', '../data/train_data/train_clue.txt')

dp.showClueInf('/home/root1/lizheng/workspace/2022/京东NER/dataSummary/train_unlabel_6_4_clue.txt',
           savedir='./summary_unlabel_6_4')


