#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： JMXGODLZZ
# datetime： 2022/4/6 下午5:28 
# ide： PyCharm
import os
import pickle
import random
from tqdm import tqdm
import json
from seqeval.metrics.sequence_labeling import get_entities
import shutil
from ltp import LTP
import jieba

def getclue2view(filepath):
    viewmap = {}

    lines = open(filepath, encoding='utf-8').readlines()

    for line in lines:
        lj = json.loads(line)
        text = lj['text']
        label = lj['label']
        nstr = ''
        entities = []
        for tp, etdict in label.items():
            for et, spans in etdict.items():
                for (st, ed) in spans:
                    entities.append([et, st, ed, tp])

        entities = sorted(entities, key=lambda item: (item[1], item[2]))
        pref = 0
        while len(entities) > 0:
            entity = entities.pop(0)
            nstr += text[pref: entity[1]]
            nstr += '[{}{}{}]'.format(entity[3], entity[0], entity[3])
            pref = entity[2] + 1
        viewmap[text] = nstr

    return viewmap

class dataAugmenter():
    def __init__(self, savedir):
        # self.__pos = LTP()
        self.advlst = self.loadtxt('dict/副词/副词总.txt')
        self.antdict = self.loaddict('dict/0005同近反义词/反义词.txt', sep='―')
        self.simdict = self.loaddict('dict/0005同近反义词/同义词.txt', sep=' ')
        self.etpath = '../dataSummary/summary/jdkg.pickle'
        self.combinepath = '..//dataSummary/summary/combinations.pickle'
        self.jdkg = self.loadpickle(self.etpath)
        self.combinations = self.loadpickle(self.combinepath)
        self.detailnest = self.loadpickle('./nestdetail.pickle')
        self.leftcombine = self.loadpickle('./leftcombinedict.pickle')
        self.label_entity = {}
        self.highfreq_entity = {}
        self.lowfreq_entity = {}
        self.dup_entity = {}
        self.lesslabel = {}
        self.lesslabelname = []
        self.replacename = [
            6, 15, 16, 17, 19, 20, 21, 32, 33, 34, 35, 44, 50, 51, 52, 53
        ]
        self.morebutworsename = [
            2, 3, 6, 9, 10, 22, 36, 39, 47, 49,
        ]
        self.parseKG()
        self.savedir = savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    def loadtxt(self, filepath):
        res = []
        lines = open(filepath, encoding='utf-8').readlines()

        for line in lines:
            line = line.strip()
            res.append(line)
        return res

    def loaddict(self, filepath, sep):
        dt = {}

        lines = open(filepath, encoding='utf-8').readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            linesp = line.split(sep)
            dt[linesp[0]] = linesp[1]
            dt[linesp[1]] = linesp[0]
        return dt


    def loadpickle(self, filepath):
        fr = open(filepath, mode='rb')
        data = pickle.load(fr)
        return data

    def get_pos_tag(self, sentence):
        r"""
        pos tag function.

        :param str sentence: the sentence need to be ner
        :return: the triple form (tags,start,end)
        """

        assert isinstance(sentence, (list, str))
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''
            for word in sentence:
                tmp += word
            sentence = tmp

        if not sentence:
            return []

        if self.__pos is None:
            # get pos tag
            self.__pos = LTP()
        seg, hidden = self.__pos.seg([sentence])

        pos = self.__pos.pos(hidden)
        seg = seg[0]
        pos = pos[0]
        pos_tag = []
        cnt = 0
        for tag in range(len(pos)):
            pos_tag.append([pos[tag], cnt, cnt + len(seg[tag]) - 1])
            cnt += len(seg[tag])

        return pos_tag

    def resolveTags(self, seg, tags):
        '''
        根据分词对齐tags
        :param seg:
        :param tags:
        :return:
        '''
        newtags = []
        j = 0
        for i, lst in enumerate(seg):
            newtags.append(tags[j: j + len(lst)])
            j += len(lst)
        return newtags

    def insertAdvs(self, sentence, tags, seed=125, threshold=0.1):
        seg, hidden = self.__pos.seg([sentence])
        seg = seg[0]

        pos = self.__pos.pos(hidden)
        pos = pos[0]

        verbids = []
        for i, cx in enumerate(pos):
            if 'v' in cx:
                verbids.append(i)

        if len(verbids) == 0:
            return sentence, tags
        tags = self.resolveTags(seg, tags)
        n = int(len(verbids) * threshold)
        #random.seed(seed)
        sample_num = max(n, 1)
        rlst = random.sample(verbids, sample_num)
        advs = random.sample(self.advlst, sample_num)
        for i, j in enumerate(rlst):
            seg[j] = advs[i] + seg[j]
            tags[j] = ['O'] * len(advs[i]) + tags[j]
        newtags = []
        for tag in tags:
            newtags.extend(tag)
        return ''.join(seg), newtags

    def replaceAnt(self, sentence, tags, seed=126, threshold=0.1):
        seg = jieba.lcut(sentence)
        tags = self.resolveTags(seg, tags)
        n = int(len(seg) * threshold)
        # random.seed(seed)
        sample_num = max(n, 1)
        rlst = random.sample(range(len(seg)), sample_num)
        record = {}
        for i, j in enumerate(rlst):
            if seg[j] in self.antdict:
                '''
                考虑分词是否截断实体
                '''
                if tags[j] == ['O'] * len(tags[j]):
                    tags[j] = ['O'] * len(self.antdict[seg[j]])
                elif len(tags[j]) == 1 and 'B-' in tags[j][0]:
                    if len(self.antdict[seg[j]]) > 1:
                        type = tags[j][0].replace('B-', '')
                        nt = []
                        for i, c in enumerate(self.antdict[seg[j]]):
                            if i == 0:
                                nt.append('B-{}'.format(type))
                            else:
                                nt.append('I-{}'.format(type))
                        tags[j] = nt
                elif 'B-' in tags[j][0] and j < len(rlst) and 'B-' in tags[j + 1][0]:
                    type = tags[j][0].replace('B-', '')
                    if len(self.antdict[seg[j]]) == 1:
                        tags[j] = ['B-{}'.format(type)]
                    else:
                        nt = []
                        for i, c in enumerate(self.antdict[seg[j]]):
                            if i == 0:
                                nt.append('B-{}'.format(type))
                            else:
                                nt.append('I-{}'.format(type))
                        tags[j] = nt
                # elif 'B-' in tags[j][0] and 'M-' in tags[j][-1]:
                #     type = tags[j][0].replace('B-', '')
                #     if len(self.antdict[seg[j]]) == 1:
                #         tags[j] = ['B-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.antdict[seg[j]]):
                #             if i == 0:
                #                 nt.append('B-{}'.format(type))
                #             else:
                #                 nt.append('M-{}'.format(type))
                #         tags[j] = nt
                # elif 'M-' in tags[j][0] and 'E-' in tags[j][-1]:
                #     type = tags[j][0].replace('M-', '')
                #     if len(self.antdict[seg[j]]) == 1:
                #         tags[j] = ['E-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.antdict[seg[j]]):
                #             if i == len(self.antdict[seg[j]]) - 1:
                #                 nt.append('E-{}'.format(type))
                #             else:
                #                 nt.append('M-{}'.format(type))
                #         tags[j] = nt
                # elif 'M-' in tags[j][0] and 'M-' in tags[j][-1]:
                #     type = tags[j][0].replace('M-', '')
                #     if len(self.antdict[seg[j]]) == 1:
                #         tags[j] = ['M-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.antdict[seg[j]]):
                #             nt.append('M-{}'.format(type))
                #         tags[j] = nt
                else:
                    print('发生截断:{},{}'.format(seg[j], tags[j]))
                    continue # 发生实体截断

                record[seg[j]] = self.antdict[seg[j]]
                seg[j] = self.antdict[seg[j]]
        newtags = []
        for tag in tags:
            newtags.extend(tag)
        return ''.join(seg), newtags

    def replaceSim(self, sentence, tags, seed=127, threshold=0.1):
        seg = jieba.lcut(sentence)
        tags = self.resolveTags(seg, tags)
        n = int(len(seg) * threshold)
        # random.seed(seed)
        sample_num = max(n, 1)
        rlst = random.sample(range(len(seg)), sample_num)
        record = {}
        for i, j in enumerate(rlst):
            if seg[j] in self.simdict:
                '''
                考虑分词是否截断实体
                '''
                if tags[j] == ['O'] * len(tags[j]):
                    tags[j] = ['O'] * len(self.simdict[seg[j]])
                elif len(tags[j]) == 1 and 'B-' in tags[j][0]:
                    if len(self.simdict[seg[j]]) > 1:
                        type = tags[j][0].replace('B-', '')
                        nt = []
                        for i, c in enumerate(self.simdict[seg[j]]):
                            if i == 0:
                                nt.append('B-{}'.format(type))
                            else:
                                nt.append('I-{}'.format(type))
                        tags[j] = nt
                elif 'B-' in tags[j][0] and j < len(rlst) and 'B-' in tags[j + 1][0]:
                    type = tags[j][0].replace('B-', '')
                    if len(self.simdict[seg[j]]) == 1:
                        tags[j] = ['B-{}'.format(type)]
                    else:
                        nt = []
                        for i, c in enumerate(self.simdict[seg[j]]):
                            if i == 0:
                                nt.append('B-{}'.format(type))
                            else:
                                nt.append('I-{}'.format(type))
                        tags[j] = nt
                # elif 'B-' in tags[j][0] and 'M-' in tags[j][-1]:
                #     type = tags[j][0].replace('B-', '')
                #     if len(self.simdict[seg[j]]) == 1:
                #         tags[j] = ['B-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.simdict[seg[j]]):
                #             if i == 0:
                #                 nt.append('B-{}'.format(type))
                #             else:
                #                 nt.append('M-{}'.format(type))
                #         tags[j] = nt
                # elif 'M-' in tags[j][0] and 'E-' in tags[j][-1]:
                #     type = tags[j][0].replace('M-', '')
                #     if len(self.simdict[seg[j]]) == 1:
                #         tags[j] = ['E-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.simdict[seg[j]]):
                #             if i == len(self.simdict[seg[j]]) - 1:
                #                 nt.append('E-{}'.format(type))
                #             else:
                #                 nt.append('M-{}'.format(type))
                #         tags[j] = nt
                # elif 'M-' in tags[j][0] and 'M-' in tags[j][-1]:
                #     type = tags[j][0].replace('M-', '')
                #     if len(self.simdict[seg[j]]) == 1:
                #         tags[j] = ['M-{}'.format(type)]
                #     else:
                #         nt = []
                #         for i, c in enumerate(self.simdict[seg[j]]):
                #             nt.append('M-{}'.format(type))
                #         tags[j] = nt
                else:
                    print('发生截断:{},{}'.format(seg[j], tags[j]))
                    continue # 发生实体截断
                record[seg[j]] = self.simdict[seg[j]]
                seg[j] = self.simdict[seg[j]]
        newtags = []
        for tag in tags:
            newtags.extend(tag)
        return ''.join(seg), newtags

    def parseKG(self):
        '''
        首先需要得到两个分布：
        各标签下实体分布，找到低频的实体，同时也得到了标签包含实体的数量,找到低频的类别；
        多类别实体的分布，便于对该类别处理
        :return:
        '''
        for et, infdict in tqdm(self.jdkg.items()):
            if len(infdict) > 1:
                self.dup_entity[et] = sorted(infdict.items(), key=lambda item: item[1], reverse=True)
            for tp, count in infdict.items():
                self.label_entity.setdefault(tp, {})
                self.label_entity[tp][et] = count
                if count < 100:
                    self.lowfreq_entity.setdefault(tp, {})
                    self.lowfreq_entity[tp][et] = count
                else:
                    self.highfreq_entity.setdefault(tp, {})
                    self.highfreq_entity[tp][et] = count

        for k, v in self.label_entity.items():
            self.lesslabel[k] = len(v)

        self.lesslabel = sorted(self.lesslabel.items(), key=lambda item: item[1])
        for (tp, count) in self.lesslabel:
            if count < 900:
                self.lesslabelname.append(tp)



        # dupfw = open('../dataSummary/summary/dupentity.pickle', mode='wb')
        # labelfw = open('../dataSummary/summary/labelentity.pickle', mode='wb')

        # pickle.dump(self.dup_entity, dupfw)
        # pickle.dump(self.label_entity, labelfw)

    def genclue(self, sentence, tags):
        ljson = {}
        ljson['text'] = sentence
        ljson['label'] = {}
        entities = get_entities(tags)
        for entity in entities:
            (tp, st, ed) = entity
            et = sentence[st: ed + 1]
            ljson['label'].setdefault(tp, {})
            ljson['label'][tp].setdefault(et, [])
            ljson['label'][tp][et].append([st, ed])
        return json.dumps(ljson, ensure_ascii=False)

    def augmain(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if lowfreqflag:
                lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
                if lowfreq == text:
                    continue
                lowfreqjson = self.genclue(lowfreq, tagslow)
                fw.write(lowfreqjson + '\n')

    def augmainV2(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProb(text, nlabels, dupentity)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')

    def augmainV5(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.lesslabelname:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntity(text, nlabels, lessentities)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntity(text, nlabels, lessentities)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')

            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     if lowfreq == text:
            #         continue
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')


    def augmainV7(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.lesslabelname:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntity(text, nlabels, lessentities)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntityProb(text, nlabels, lessentities)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')

            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     if lowfreq == text:
            #         continue
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

    def augmainV8(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []
            replaceableentities = []

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.lesslabelname:
                            lessentities.append([tp, st, ed])
                        if int(tp) in self.replacename:
                            replaceableentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if len(replaceableentities) > 0:
                replaceableentities = sorted(replaceableentities, key=lambda item: (item[1], item[2]))
                replaceet, tagsreplace = self.replaceLessEntity(text, nlabels, replaceableentities)
                if replaceet == text > 0:
                    continue
                replacejson = self.genclue(replaceet, tagsreplace)
                fw.write(replacejson + '\n')

    def augmainV9(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity: # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label: # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    replacedict.setdefault(tp, set())
                    replacedict[tp] = replacedict[tp] | rt

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbCombine(text, nlabels, dupentity, replacedict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')

    def augmainV10(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity: # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label: # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    replacedict.setdefault(tp, {})
                    for et, count in rt.items():
                        replacedict[tp].setdefault(et, 0)
                        replacedict[tp][et] += count

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count


            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')

    def augmainV11(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.lesslabelname:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    replacedict.setdefault(tp, {})
                    for et, count in rt.items():
                        replacedict[tp].setdefault(et, 0)
                        replacedict[tp][et] += count

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntityCombine(text, nlabels, lessentities, lowfreqdict)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntityProbCombine(text, nlabels, lessentities, replacedict)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')

            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     if lowfreq == text:
            #         continue
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

    def augmainV12(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity: # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label: # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    replacedict.setdefault(tp, {})
                    for et, count in rt.items():
                        replacedict[tp].setdefault(et, 0)
                        replacedict[tp][et] += count

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count


            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')

                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail, lowfreqdict)
                if dupfreq == text:
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV13(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity: # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label: # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count


            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')

                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail, lowfreqdict)
                if dupfreq == text:
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV14(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                lowfreqdict)
                if duprpfreq == text:
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV15(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 lowfreqdict)
                # if duprpfreq == text:
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV16(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                lowfreqdict)
                if dupfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV17(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.lesslabelname:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntityCombine(text, nlabels, lessentities, lowfreqdict)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntityProbCombine(text, nlabels, lessentities, replacedict)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')


    def augmainV18(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.morebutworsename:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntityCombine(text, nlabels, lessentities, lowfreqdict)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntityProbCombine(text, nlabels, lessentities, replacedict)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')

    def augmainV19(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > 10:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                lowfreqdict)
                if dupfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV20(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.morebutworsename:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            allcontext, tagsall = self.replaceLessEntityContextEntityProbCombine(text, nlabels, [], replacedict)
            if allcontext == text:
                continue
            allcontextjson = self.genclue(allcontext, tagsall)
            fw.write(allcontextjson + '\n')

    def augmainV21(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []
            dupentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > 10:
                                dupentity.append(et)
                    for (st, ed) in spans:
                        if int(tp) in self.morebutworsename:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count
            if len(lessentities) == 0 and len(dupentity) == 0:
                allcontext, tagsall = self.replaceLessEntityContextEntityProbCombine(text, nlabels, [], replacedict)
                if allcontext == text:
                    continue
                allcontextjson = self.genclue(allcontext, tagsall)
                fw.write(allcontextjson + '\n')
    def augmainV22(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []

            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    for (st, ed) in spans:
                        if int(tp) in self.morebutworsename:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(lessentities) > 0:
                lessentities = sorted(lessentities, key=lambda item: (item[1], item[2]))
                lesset, tagsless = self.replaceLessEntityCombine(text, nlabels, lessentities, lowfreqdict)
                if lesset == text:
                    continue
                lessjson = self.genclue(lesset, tagsless)
                fw.write(lessjson + '\n')

                lesscontextet, tagslesscontext = self.replaceLessEntityContextEntityProbCombine(text, nlabels, lessentities, lowfreqdict)
                if lesscontextet == text:
                    continue
                lesscontextjson = self.genclue(lesscontextet, tagslesscontext)
                fw.write(lesscontextjson + '\n')
    def augmainV23(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            lessentities = []
            dupentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)
                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > 10:
                                dupentity.append(et)
                    for (st, ed) in spans:
                        if int(tp) in self.morebutworsename:
                            lessentities.append([tp, st, ed])
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)

            if '4' not in label: # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count
            if len(lessentities) == 0 and len(dupentity) == 0:
                allcontext, tagsall = self.replaceLessEntityContextEntityProbCombine(text, nlabels, [], lowfreqdict)
                if allcontext == text:
                    continue
                allcontextjson = self.genclue(allcontext, tagsall)
                fw.write(allcontextjson + '\n')

    def augmainV24(self, filepath, savepath):
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                lowfreqdict)
                if dupfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV25(self, filepath, savepath):
        '''
        出发点 提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                # dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                # if dupfreq == text:
                #     continue
                # dupjson = self.genclue(dupfreq, tagsdup)
                # fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                lowfreqdict)
                if duprpfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV26(self, filepath, savepath):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 lowfreqdict)
                # if duprpfreq == text: # 确保两种增强数量一致
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV27(self, filepath, savepath):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentity_maindetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)

                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentity_maindetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(dupentity_maindetail) > 0:
                dupentity_maindetail = sorted(dupentity_maindetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentity_maindetail,
                                                                                lowfreqdict)
                if duprpfreq != text: # 确保两种增强数量一致
                    duprpjson = self.genclue(duprpfreq, tagsrpdup)
                    fw.write(duprpjson + '\n')

    def augmainV28(self, filepath, savepath):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, highfreqdict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 lowfreqdict)
                # if duprpfreq == text: # 确保两种增强数量一致
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV29(self, filepath, savepath):
        '''
        出发点 提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                # dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                # if dupfreq == text:
                #     continue
                # dupjson = self.genclue(dupfreq, tagsdup)
                # fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                highfreqdict)
                if duprpfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')

    def augmainV30(self, filepath, savepath):
        '''
        出发点 提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:

                # insverb, tagsinv = self.insertAdvs(text, nlabels)
                # if insverb.replace(' ', '') != text.replace(' ', ''):
                #     insverbjson = self.genclue(insverb, tagsinv)
                #     fw.write(insverbjson + '\n')

                simrp, tagsim = self.replaceSim(text, nlabels)
                if simrp.replace(' ', '') != text.replace(' ', ''):
                    simjson = self.genclue(simrp, tagsim)
                    fw.write(simjson + '\n')

                antrp, tagant = self.replaceAnt(text, nlabels)
                if antrp.replace(' ', '') != text.replace(' ', ''):
                    antjson = self.genclue(antrp, tagant)
                    fw.write(antjson + '\n')
                # dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                # if dupfreq == text:
                #     continue
                # dupjson = self.genclue(dupfreq, tagsdup)
                # fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 highfreqdict)
                # if duprpfreq == text: # 确保两种增强数量一致
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV31(self, filepath, savepath):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, replacedict)
                if dupfreq == text:
                    continue
                dupjson = self.genclue(dupfreq, tagsdup)
                fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 lowfreqdict)
                # if duprpfreq == text: # 确保两种增强数量一致
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV32(self, filepath, savepath):
        '''
        出发点 提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:
                # dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                # if dupfreq == text:
                #     continue
                # dupjson = self.genclue(dupfreq, tagsdup)
                # fw.write(dupjson + '\n')
                dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                                                                                replacedict)
                if duprpfreq == text: # 确保两种增强数量一致
                    continue
                duprpjson = self.genclue(duprpfreq, tagsrpdup)
                fw.write(duprpjson + '\n')


    def augmainV33(self, filepath, savepath):
        '''
        出发点 提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        :param filepath:
        :param savepath:
        :return:
        '''
        jieba.load_userdict('./dict/JD_jieba.txt')
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()
        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentitydetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentitydetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0:

                # insverb, tagsinv = self.insertAdvs(text, nlabels)
                # if insverb.replace(' ', '') != text.replace(' ', ''):
                #     insverbjson = self.genclue(insverb, tagsinv)
                #     fw.write(insverbjson + '\n')

                simrp, tagsim = self.replaceSim(text, nlabels)
                if simrp.replace(' ', '') != text.replace(' ', ''):
                    simjson = self.genclue(simrp, tagsim)
                    fw.write(simjson + '\n')

                antrp, tagant = self.replaceAnt(text, nlabels)
                if antrp.replace(' ', '') != text.replace(' ', ''):
                    antjson = self.genclue(antrp, tagant)
                    fw.write(antjson + '\n')
                # dupfreq, tagsdup = self.replaceDupEntityProbLessCombine(text, nlabels, dupentity, lowfreqdict)
                # if dupfreq == text:
                #     continue
                # dupjson = self.genclue(dupfreq, tagsdup)
                # fw.write(dupjson + '\n')
                # dupentitydetail = sorted(dupentitydetail, key=lambda item: (item[1], item[2]))
                # duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentitydetail,
                #                                                                 highfreqdict)
                # if duprpfreq == text: # 确保两种增强数量一致
                #     continue
                # duprpjson = self.genclue(duprpfreq, tagsrpdup)
                # fw.write(duprpjson + '\n')

    def augmainV34(self, filepath, savepath, threshold=5000):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            dupentity_maindetail = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True
                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)

                            if int(etp) == int(tp) and eid == 0 and ecount > 10 * othercount:
                                dupentity.append(et)
                                for (st, ed) in spans:
                                    dupentity_maindetail.append([tp, st, ed])



                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0 and count1 < threshold:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    count1 += 1
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(dupentity_maindetail) > 0 and count2 < threshold:
                dupentity_maindetail = sorted(dupentity_maindetail, key=lambda item: (item[1], item[2]))
                duprpfreq, tagsrpdup = self.replaceDupEntityProbSelfLessCombine(text, nlabels, dupentity_maindetail,
                                                                                lowfreqdict)
                if duprpfreq != text: # 确保两种增强数量一致
                    count2 += 1
                    duprpjson = self.genclue(duprpfreq, tagsrpdup)
                    fw.write(duprpjson + '\n')

    def augmainV35(self, filepath, savepath, threshold=5000):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        #提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        为了避免粒度错误，增强多的长度，同样替换上下文非多类别实体为低频实体
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True

                    if et in self.leftcombine:
                        lengentity.append(et)

                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)


                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0 and count1 < threshold:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    count1 += 1
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(lengentity) > 0 and count2 < threshold:
                lengfreq, tagsleng = self.replaceDupEntityProbLessCombineV2(text, nlabels, lengentity, lowfreqdict)
                if lengfreq != text:
                    count2 += 1
                    lengjson = self.genclue(lengfreq, tagsleng)
                    fw.write(lengjson + '\n')

    def augmainV36(self, filepath, savepath, threshold=5000):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        #提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        为了避免粒度错误，增强多的长度，同样替换上下文非多类别实体为低频实体
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True



                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]
                        if et in self.leftcombine:
                            lengentity.append(et)

                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)


                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0 and count1 < threshold:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    count1 += 1
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(lengentity) > 0 and count2 < threshold:
                lengfreq, tagsleng = self.replaceDupEntityProbLessCombineV2(text, nlabels, lengentity, lowfreqdict)
                if lengfreq != text:
                    count2 += 1
                    lengjson = self.genclue(lengfreq, tagsleng)
                    fw.write(lengjson + '\n')

    def augmainV37(self, filepath, savepath, threshold=5000):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        #提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        为了避免粒度错误，增强多的长度，同样替换上下文非多类别实体为低频实体,限制长实体非多类别实体与低频实体
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True

                    if et in self.leftcombine and et not in self.dup_entity and not (et in self.label_entity[int(tp)] and self.label_entity[int(tp)][et] <100):
                        lengentity.append(et)

                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]

                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)


                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0 and count1 < threshold:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    count1 += 1
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(lengentity) > 0 and count2 < threshold:
                lengfreq, tagsleng = self.replaceDupEntityProbLessCombineV2(text, nlabels, lengentity, lowfreqdict)
                if lengfreq != text:
                    count2 += 1
                    lengjson = self.genclue(lengfreq, tagsleng)
                    fw.write(lengjson + '\n')

    def augmainV38(self, filepath, savepath, threshold=5000):
        '''
        出发点 提高多类别客体的召回率，在客体的句子中，替换上下文非多类别实体为低频实体
        #提高多类别主体的准确率，在主体的句子中，替换主体为同类别低频实体
        为了避免粒度错误，增强多的长度，同样替换上下文非多类别实体为低频实体,限制长实体非多类别实体与低频实体
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True

                    if et in self.leftcombine and et not in self.dup_entity and not (et in self.label_entity[int(tp)] and self.label_entity[int(tp)][et] <100):
                        for (st, ed) in spans:
                            lengentity.append([tp, st, ed])

                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]

                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)


                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue
            # if lowfreqflag:
            #     lowfreq, tagslow = self.replaceLowFreqEntity(text, nlabels)
            #     lowfreqjson = self.genclue(lowfreq, tagslow)
            #     fw.write(lowfreqjson + '\n')

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            if len(dupentity) > 0 and count1 < threshold:
                dupfreq, tagsdup = self.replaceDupEntityProbLessCombineV2(text, nlabels, dupentity, lowfreqdict)
                if dupfreq != text:
                    count1 += 1
                    dupjson = self.genclue(dupfreq, tagsdup)
                    fw.write(dupjson + '\n')

            if len(lengentity) > 0 and count2 < threshold:
                lengfreq, tagsleng = self.replaceDupEntityProbSelfLessCombine(text, nlabels, lengentity, lowfreqdict)
                if lengfreq != text:
                    count2 += 1
                    lengjson = self.genclue(lengfreq, tagsleng)
                    fw.write(lengjson + '\n')

    def augmainV39(self, filepath, savepath, threshold=5000):
        '''
        出发点 随机替换数据集中实体为低频实体，15%概率
        控制两者数量一致
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.lowfreq_entity[int(tp)]:
                        lowfreqflag = True

                    if et in self.leftcombine and et not in self.dup_entity and not (et in self.label_entity[int(tp)] and self.label_entity[int(tp)][et] <100):
                        for (st, ed) in spans:
                            lengentity.append([tp, st, ed])

                    if et in self.dup_entity:  # 查找出现多类别低频实体
                        inf = self.dup_entity[et]
                        othercount = sum([item[1] for item in inf[1:]])
                        maincount = inf[0][1]

                        for eid, (etp, ecount) in enumerate(inf):
                            if int(etp) == int(tp) and eid != 0 and ecount > max(5, maincount // 20):
                                dupentity.append(et)


                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            # fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue

            replacedict = {}
            combinedict = {}
            for mainet in mainentities:
                rd = self.combinations[mainet]
                for tp, rt in rd.items():
                    combinedict.setdefault(tp, {})
                    for et, count in rt.items():
                        combinedict[tp].setdefault(et, [])
                        combinedict[tp][et].append(count)

            for tp, etdict in combinedict.items():
                replacedict.setdefault(tp, {})
                for et, countlst in etdict.items():
                    if len(countlst) > 0:
                        replacedict[tp][et] = sum(countlst)

            highfreqdict = {}
            lowfreqdict = {}

            for tp, etd in replacedict.items():
                highfreqdict.setdefault(tp, {})
                lowfreqdict.setdefault(tp, {})
                for et, count in etd.items():
                    if count < 100:
                        lowfreqdict[tp][et] = count
                    else:
                        highfreqdict[tp][et] = count

            rpfreq, tagsrp = self.replaceAllEntityProbLessCombine(text, nlabels, [], lowfreqdict)
            if rpfreq != text:
                count1 += 1
                dupjson = self.genclue(rpfreq, tagsrp)
                fw.write(dupjson + '\n')

    def augmainV40(self, filepath, savepath, sampleleng=32):
        '''
        出发点 以四号实体当做
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            mainstarts = [0]
            maindict = {} # 记录主要实体起始与结束对应表
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    for (st, ed) in spans:
                        if tp == '4':
                            mainstarts.append(st)
                            maindict[st] = ed
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue

            mainstarts.append(len(text) - 1)
            mainstarts = sorted(set(mainstarts))

            nums = len(mainstarts)
            for ix, st1 in enumerate(mainstarts):
                j = ix + 1
                while j < nums: # 所以结束循环的j要么到了最后为nums 要么与前一个长度超过阈值
                    jx = mainstarts[j]
                    if jx - ix + 1 < sampleleng:
                        j += 1
                    else:
                        break
                nx = j - 1
                ed2 = maindict.get(mainstarts[nx], mainstarts[nx])
                if nx > ix:
                    rpfreq = text[st1: ed2 + 1]
                    tagsrp = nlabels[st1: ed2 + 1]
                    if rpfreq != text and len(rpfreq) > 20:
                        dupjson = self.genclue(rpfreq, tagsrp)
                        fw.write(dupjson + '\n')

    def augmainV42(self, filepath, savepath, sampleleng=64):
        '''
        出发点 按照V40采样一半的长度，之后将增强的数据打乱，逐个拼接到样本长度
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        auglines = []

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            mainstarts = [0]
            maindict = {} # 记录主要实体起始与结束对应表
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    for (st, ed) in spans:
                        if tp == '4':
                            mainstarts.append(st)
                            maindict[st] = ed
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue

            mainstarts.append(len(text) - 1)
            mainstarts = sorted(set(mainstarts))

            nums = len(mainstarts)
            for ix, st1 in enumerate(mainstarts):
                j = ix + 1
                while j < nums: # 所以结束循环的j要么到了最后为nums 要么与前一个长度超过阈值
                    jx = mainstarts[j]
                    if jx - ix + 1 < sampleleng // 2:
                        j += 1
                    else:
                        break
                nx = j - 1
                ed2 = maindict.get(mainstarts[nx], mainstarts[nx])
                if nx > ix:
                    rpfreq = text[st1: ed2 + 1]
                    tagsrp = nlabels[st1: ed2 + 1]
                    if rpfreq != text and len(rpfreq) > 10:
                        auglines.append([rpfreq, tagsrp])
        random.seed(3407)
        random.shuffle(auglines)
        nums = len(auglines)
        st = 0
        while st < nums:
            ed = st + 1
            leng = len(auglines[st][0])
            while ed < nums:
                if leng + len(auglines[ed][0]) < sampleleng:
                    leng += len(auglines[ed][0])
                    ed += 1
                else:
                    break
            ed = ed - 1
            atext = ''
            alabels = []
            for nid in range(st, ed +1):
                atext += auglines[nid][0]
                alabels.extend(auglines[nid][1])

            augjson = self.genclue(atext, alabels)
            fw.write(augjson + '\n')
            st = ed + 1

    def augmainV43(self, filepath, savepath, sampleleng=32):
        '''
        出发点 对存在粒度错误[不能划分出来]的实体 进行采样增强，目的提高准确率
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        for line in tqdm(lines):
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)
            lowfreqflag = False
            dupentity = []
            lengentity = []
            mainentities = set()
            mainstarts = [0]
            maindict = {} # 记录主要实体起始与结束对应表
            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    if tp == '4':
                        mainentities.add(et)

                    if et in self.leftcombine and et not in self.dup_entity and not (et in self.label_entity[int(tp)] and self.label_entity[int(tp)][et] <100):
                        for (st, ed) in spans:
                            lengentity.append([tp, st, ed])

                    for (st, ed) in spans:
                        if tp == '4':
                            mainstarts.append(st)
                            maindict[st] = ed
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)
            fw.write(line)
            if '4' not in label:  # 暂未考虑无主体的情况
                continue

            if len(lengentity) == 0:
                continue

            mainstarts.append(len(text) - 1)
            mainstarts = sorted(set(mainstarts))

            nums = len(mainstarts)
            for ix, st1 in enumerate(mainstarts):
                j = ix + 1
                while j < nums: # 所以结束循环的j要么到了最后为nums 要么与前一个长度超过阈值
                    jx = mainstarts[j]
                    if jx - ix + 1 < sampleleng:
                        j += 1
                    else:
                        break
                nx = j - 1
                ed2 = maindict.get(mainstarts[nx], mainstarts[nx])
                if nx > ix:
                    rpfreq = text[st1: ed2 + 1]
                    tagsrp = nlabels[st1: ed2 + 1]
                    if rpfreq != text and len(rpfreq) > 20:
                        dupjson = self.genclue(rpfreq, tagsrp)
                        fw.write(dupjson + '\n')

    def shuffleDes(self, lj, seed=42):
        random.seed(seed)
        text = lj['text']
        label = lj['label']
        nlabels = ['O'] * len(text)

        allentities = []
        for tp, etdict in label.items():
            for et, spans in etdict.items():
                for (st, ed) in spans:
                    allentities.append([st, ed, et, tp])
                    for i in range(st, ed + 1):
                        if i == st:
                            nlabels[i] = 'B-{}'.format(tp)
                        else:
                            nlabels[i] = 'I-{}'.format(tp)

        allentities = sorted(allentities, key=lambda item: (item[0], item[1]))

        augtext = ''
        auglabels = []

        prefix = 0 # 记录前缀，寻找无标注的截断
        shufflelst = [] # 存储描述实体，以便进行打乱

        for entity in allentities:
            st, ed, et, tp = entity
            addflag = False # 避免无标注后面跟着4号导致shuffle添加两次
            if st != prefix:
                random.shuffle(shufflelst)
                for (ret, rtp) in shufflelst:
                    augtext += ret
                    for i in range(len(ret)):
                        if i == 0:
                            auglabels.append('B-{}'.format(rtp))
                        else:
                            auglabels.append('I-{}'.format(rtp))
                addflag = True
                blank = text[prefix: st]
                augtext += blank
                auglabels += ['O'] * len(blank)
                shufflelst = []

            if tp == '4':
                if not addflag:
                    random.shuffle(shufflelst)
                    for (ret, rtp) in shufflelst:
                        augtext += ret
                        for i in range(len(ret)):
                            if i == 0:
                                auglabels.append('B-{}'.format(rtp))
                            else:
                                auglabels.append('I-{}'.format(rtp))
                augtext += et
                for i in range(len(et)):
                    if i == 0:
                        auglabels.append('B-{}'.format(tp))
                    else:
                        auglabels.append('I-{}'.format(tp))
                shufflelst = []
            else:
                shufflelst.append([et, tp])
            prefix = ed + 1

        if len(shufflelst) > 0:
            random.shuffle(shufflelst)
            for (ret, rtp) in shufflelst:
                augtext += ret
                for i in range(len(ret)):
                    if i == 0:
                        auglabels.append('B-{}'.format(rtp))
                    else:
                        auglabels.append('I-{}'.format(rtp))

        if prefix != len(text):
            tail = text[prefix:]
            augtext += tail
            auglabels += ['O'] * len(tail)

        augjson = self.genclue(augtext, auglabels)
        return augjson

    def augmainV45(self, filepath, savepath, threshold=5000):
        '''
        出发点 打乱形容词描述，进行数据增强
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        count1 = 0
        count2 = 0

        for line in tqdm(lines):
            fw.write(line)
            lj = json.loads(line)
            if '4' not in lj['label']:
                print(1)
            augjson = self.shuffleDes(lj)
            if augjson == lj:
                augjson = self.shuffleDes(lj, 666)

            if augjson != lj:
                fw.write('{}\n'.format(augjson))

    def augmainV46(self, filepath, savepath, sampleleng=128):
        '''
        出发点 上下文不重要的话，将样本拼接
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        record = []

        for line in tqdm(lines):
            fw.write(line)
            lj = json.loads(line)
            text = lj['text']
            label = lj['label']
            nlabels = ['O'] * len(text)

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)

            record.append([text, nlabels])
        nums = len(record)
        ix = 0
        while ix < nums:
            (text, nlabels) = record[ix]
            rtext = text
            rlabels = [x for x in nlabels]
            j = ix + 1
            while j < nums:  # 所以结束循环的j要么到了最后为nums 要么与前一个长度超过阈值
                (text2, nlabels2) = record[j]
                if len(rtext +text2) < sampleleng:
                    j += 1
                    rtext += text2
                    rlabels += nlabels2
                else:
                    break

            dupjson = self.genclue(rtext, rlabels)
            fw.write(dupjson + '\n')
            ix = j


    def augmainV48(self, filepath, savepath, sampleleng=128):
        '''
        出发点 上下文不重要的话，将样本拼接，shuffle几次获取不同拼接样本
        :param filepath:
        :param savepath:
        :return:
        '''
        fw = open(savepath, encoding='utf-8', mode='w')
        lines = open(filepath, encoding='utf-8').readlines()

        record = []
        result = set()
        for line in tqdm(lines):
            # fw.write(line)
            lj = json.loads(line)
            text = lj['text']
            label = lj.get('label', {})
            nlabels = ['O'] * len(text)

            for tp, etdict in label.items():
                for et, spans in etdict.items():
                    for (st, ed) in spans:
                        for i in range(st, ed + 1):
                            if i == st:
                                nlabels[i] = 'B-{}'.format(tp)
                            else:
                                nlabels[i] = 'I-{}'.format(tp)

            record.append([text, nlabels])
        for _ in range(20):
            random.shuffle(record)
            nums = len(record)
            ix = 0
            while ix < nums:
                (text, nlabels) = record[ix]
                rtext = text
                rlabels = [x for x in nlabels]
                j = ix + 1
                while j < nums:  # 所以结束循环的j要么到了最后为nums 要么与前一个长度超过阈值
                    (text2, nlabels2) = record[j]
                    if len(rtext +text2) < sampleleng:
                        j += 1
                        rtext += text2
                        rlabels += nlabels2
                    else:
                        break

                dupjson = self.genclue(rtext, rlabels)
                result.add(dupjson)
                ix = j

        for dupjson in result:
            fw.write('{}\n'.format(dupjson))


    def augdir(self, filedir):
        for filename in os.listdir(filedir):
            if 'JDtrain' in filename:
                filepath = os.path.join(filedir, filename)
                savepath = os.path.join(self.savedir, filename.replace('.txt', '_augV48.txt'))
                self.augmainV48(filepath, savepath)
            else:
                filepath = os.path.join(filedir, filename)
                savepath = os.path.join(self.savedir, filename)
                shutil.copy(filepath, savepath)

    def replaceLessEntity(self, sentence, tags, entities):
        '''
        思路 替换句子中较少类别的实体 为同类的实体
        :param sentence:
        :param tags:
        :return:
        '''
        res = ''
        restags = []
        prefix = 0
        for entity in entities:
            rentity = random.choice(list(self.label_entity[int(entity[0])].keys()))
            count = 0
            while rentity == sentence[entity[1]: entity[2] + 1]:  # 如果实体相同 重新选择
                rentity = random.choice(list(self.label_entity[int(entity[0])].keys()))
                count += 1
                if count > 50:
                    return sentence, tags

            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceLessEntityCombine(self, sentence, tags, entities, replacedict):
        '''
        思路 替换句子中较少类别的实体 为搭配的同类的实体
        :param sentence:
        :param tags:
        :return:
        '''
        res = ''
        restags = []
        prefix = 0
        for entity in entities:
            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity == sentence[entity[1]: entity[2] + 1]:  # 如果实体相同 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags

            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceLessEntityContextEntity(self, sentence, tags, lessentities):
        '''
        思路 句子中较少类别的实体 替换上下文中的一个实体为低频实体
        :param sentence:
        :param tags:
        :return:
        '''
        res = ''
        restags = []
        prefix = 0
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        entity = random.choice(entities)
        count = 0
        while entity in lessentities:
            entity = random.choice(entities)
            count += 1
            if count > 50:
                return sentence, tags

        rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
        count = 0
        while rentity == sentence[entity[1]: entity[2] + 1]:  # 如果实体相同 重新选择
            rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
            count += 1
            if count > 50:
                return sentence, tags

        rtags = ['M-{}'.format(entity[0])] * len(rentity)
        if len(rtags) == 1:
            rtags[0] = 'S-{}'.format(entity[0])
        else:
            rtags[0] = 'B-{}'.format(entity[0])
            rtags[-1] = 'E-{}'.format(entity[0])
        res += sentence[prefix: entity[1]] + rentity
        restags += tags[prefix: entity[1]] + rtags
        prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceLessEntityContextEntityProb(self, sentence, tags, lessentities):
        '''
        思路 句子中较少类别的实体 替换上下文中的10%实体为低频实体
        :param sentence:
        :param tags:
        :return:
        '''
        res = ''
        restags = []
        prefix = 0
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags

        for i, entity in enumerate(entities):
            if entity in lessentities:
                continue

            prob = random.random()
            if prob > 0.1:
                continue

            rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
            count = 0
            while rentity == sentence[entity[1]: entity[2] + 1]:  # 如果实体相同 重新选择
                rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
                count += 1
                if count > 50:
                    return sentence, tags

            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceLessEntityContextEntityProbCombine(self, sentence, tags, lessentities, replacedict):
        '''
        思路 句子中较少类别的实体 替换上下文中的10%实体为低频实体
        :param sentence:
        :param tags:
        :return:
        '''
        res = ''
        restags = []
        prefix = 0
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags

        for i, entity in enumerate(entities):
            if entity in lessentities:
                continue

            prob = random.random()
            if prob > 0.1:
                continue

            if entity[0] == '4':
                continue

            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity == sentence[entity[1]: entity[2] + 1]:  # 如果实体相同 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags

            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceDupEntity(self, sentence, tags, dupentities):
        '''
        思路 将存在多类别低频实体句子中的实体替换为低频同类实体，并且该实体不具有多类别
        一次增强一个，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0
        entity = random.choice(entities)
        count = 0
        while sentence[entity[1]: entity[2] + 1] in self.highfreq_entity.get(int(entity[0]), {}) or sentence[entity[1]: entity[2] + 1] in dupentities:
            entity = random.choice(entities)
            count += 1
            if count > 50:
                return sentence, tags

        rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
        count = 0
        while len(rentity) == 1 or rentity in self.dup_entity: # 如果长度为1 或者 属于多类别实体的 重新选择
            rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
            count += 1
            if count > 50:
                return sentence, tags
        rtags = ['M-{}'.format(entity[0])] * len(rentity)
        if len(rtags) == 1:
            rtags[0] = 'S-{}'.format(entity[0])
        else:
            rtags[0] = 'B-{}'.format(entity[0])
            rtags[-1] = 'E-{}'.format(entity[0])
        res += sentence[prefix: entity[1]] + rentity
        restags += tags[prefix: entity[1]] + rtags
        prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags


    def replaceDupEntityProb(self, sentence, tags, dupentities):
        '''
        思路 将存在多类别低频实体句子中的实体替换为低频同类实体，并且该实体不具有多类别
        一次增强15%，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in self.highfreq_entity.get(int(entity[0]), {}) or sentence[entity[1]: entity[2] + 1] in dupentities:
                continue

            prob = random.random()
            if prob > 0.15:
                continue

            rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
            count = 0
            while len(rentity) == 1 or rentity in self.dup_entity: # 如果长度为1 或者 属于多类别实体的 重新选择
                rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceDupEntityProbCombine(self, sentence, tags, dupentities, replacedict):
        '''
        思路 将存在多类别低频实体句子中的实体替换为 组合内同类实体，并且该实体不具有多类别
        一次增强15%，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in dupentities:
                continue

            prob = random.random()
            if prob > 0.30:
                continue
            if len(replacedict[entity[0]]) < 2:
                continue
            rentity = random.choice(list(replacedict[entity[0]]))
            count = 0
            while rentity in self.dup_entity: # 如果 属于多类别实体的 重新选择
                rentity = random.choice(list(replacedict[entity[0]]))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceDupEntityProbLessCombine(self, sentence, tags, dupentities, replacedict):
        '''
        思路 将存在多类别低频实体句子中的少数实体替换为 组合内同类低频少于100实体

        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in dupentities or int(entity[0]) not in self.lesslabelname: # 去除多类别实体自身与 高频类别实体
                continue

            # prob = random.random()
            # if prob > 0.15:
            #     continue
            if len(replacedict[entity[0]]) < 2:
                continue
            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity in self.dup_entity: # 如果 属于多类别实体的 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceAllEntityProbLessCombine(self, sentence, tags, dupentities, replacedict):
        '''
        思路 将存在多类别低频实体句子中的低频实体替换为 组合内同类低频少于100实体，并且该实体不具有多类别
        一次增强15%，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in dupentities or sentence[entity[1]: entity[2] + 1] in self.dup_entity: # 去除多类别实体自身与 高频类别实体
                continue

            prob = random.random()
            if prob > 0.15:
                continue
            if len(replacedict[entity[0]]) < 2:
                continue
            if entity[0] == '4':
                continue
            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity in self.dup_entity: # 如果 属于多类别实体的 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceDupEntityProbLessCombineV2(self, sentence, tags, dupentities, replacedict):
        '''
        思路 将存在多类别低频实体句子中的低频实体替换为 组合内同类低频少于100实体，并且该实体不具有多类别
        一次增强15%，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in dupentities or sentence[entity[1]: entity[2] + 1] in self.dup_entity: # 去除多类别实体自身与 高频类别实体
                continue

            prob = random.random()
            if prob > 0.15:
                continue
            if len(replacedict[entity[0]]) < 2:
                continue
            if entity[0] == '4':
                continue
            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity in self.dup_entity: # 如果 属于多类别实体的 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceDupEntityProbSelfLessCombine(self, sentence, tags, dupentities, replacedict):
        '''
        思路 将存在多类别低频实体句子中的多类别实体替换为 组合内同类低频少于100实体，并且该实体不具有多类别
        :param filepath:
        :param savepath:
        :return:
        '''
        # entities = get_entities(tags)
        # if len(entities) == 0:
        #     return sentence, tags
        res = ''
        restags = []
        prefix = 0

        for i, entity in enumerate(dupentities):
            # prob = random.random()
            # if prob > 0.15:
            #     continue
            if entity[0] == '4':
                continue

            if len(replacedict[entity[0]]) < 2:
                continue
            rentity = random.choice(list(replacedict[entity[0]].keys()))
            count = 0
            while rentity in self.dup_entity: # 如果 属于多类别实体的 重新选择
                rentity = random.choice(list(replacedict[entity[0]].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags

    def replaceLowFreqEntity(self, sentence, tags):
        '''
        思路 将存在低频实体句子中的实体替换为低频同类实体，并且该实体不具有多类别
        一次增强一个，该比赛实体较为密集
        :param filepath:
        :param savepath:
        :return:
        '''
        entities = get_entities(tags)
        if len(entities) == 0:
            return sentence, tags

        nums = max(1, len(entities) // 5)

        res = ''
        restags = []
        prefix = 0


        for i, entity in enumerate(entities):
            if sentence[entity[1]: entity[2] + 1] in self.highfreq_entity.get(int(entity[0]), {}):
                continue

            prob = random.random()
            if prob > 0.15:
                continue

            rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
            count = 0
            while len(rentity) == 1 or rentity in self.dup_entity: # 如果长度为1 或者 属于多类别实体的 重新选择
                rentity = random.choice(list(self.lowfreq_entity[int(entity[0])].keys()))
                count += 1
                if count > 50:
                    return sentence, tags
            rtags = ['M-{}'.format(entity[0])] * len(rentity)
            if len(rtags) == 1:
                rtags[0] = 'S-{}'.format(entity[0])
            else:
                rtags[0] = 'B-{}'.format(entity[0])
                rtags[-1] = 'E-{}'.format(entity[0])
            res += sentence[prefix: entity[1]] + rentity
            restags += tags[prefix: entity[1]] + rtags
            prefix = entity[2] + 1

        res += sentence[prefix:]
        restags += tags[prefix:]
        return res, restags




da = dataAugmenter('../GlobalPointer-main/data/JD5-augV48')
# da.getCombineLeft()
# da.checkNestInf('../data/train_data/train_clue.txt', 'nestinf.txt')
# da.checkNestCombine()
# da.checknested()
# da.augmain('../GlobalPointer-main/data/JD10/JDtrain0.txt')
# print(1)
# da.augdir('../GlobalPointer-main/data/JD5/')
# da.datasummary()

da.augmainV48('/home/root1/lizheng/workspace/2022/京东NER/dataSummary/unlabelclue.txt', '/home/root1/lizheng/workspace/2022/京东NER/dataSummary/unlabelclue_aug-time20.txt')