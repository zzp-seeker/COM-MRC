import re
from constant import *

class RawData:
    def __init__(self, data_dir):
        self.s = 0
        self.ma_s = 0
        self.t = 0
        self.ma_t = 0

        self.data = []
        self.multi_aspect_id = []
        self.data_dir = data_dir

        data_lines = open(data_dir)
        for i, text in enumerate(data_lines):
            self.s += 1
            sentence = re.search(r'.+(?=####)',text)[0]
            labels = eval(re.search(r'(?<=####).+',text)[0])
            data_dict = {}
            data_dict['sentence']=sentence
            aspect_span_list,opinion_span_list,sentiment_list=[],[],[]

            for j in labels:
                as_l,as_r = j[0][0],j[0][-1]
                os_l,os_r = j[1][0],j[1][-1]
                if [as_l,as_r] in aspect_span_list:
                    index = aspect_span_list.index([as_l,as_r])
                    opinion_span_list[index].append([os_l,os_r])
                    if j[2]!=sentiment_list[index]:
                        print('inconsistent sentiment')
                else:
                    aspect_span_list.append([as_l,as_r])
                    opinion_span_list.append([[os_l,os_r]])
                    sentiment_list.append(j[2])
                self.t += 1
            if len(aspect_span_list) > 1: # multiple aspect
                self.multi_aspect_id.append(i)
                self.ma_s += 1
                self.ma_t += len(labels)
            data_dict['triplets']=[(aspect_span_list[i],opinion_span_list[i],sentiment_list[i])
                                   for i in range(len(aspect_span_list))]
            self.data.append(data_dict)
        self.len = len(self.data)

        self.all_triplets,self.all_multi_triplets,self.all_single_triplets = self.get_all_triplets_set()

        sample_t,sample_2t,sample_2et = 0,0,0
        for d in self.data:
            aspect_num = len(d['triplets'])
            sample_t += 1
            sample_2t += 2*aspect_num
            sample_2et += 2**aspect_num

        print()
    def __iter__(self):
        for i in range(self.len):
            yield {'sentence':self.data[i]['sentence'],'triplets':self.data[i]['triplets']}

    #  i-al-ar-ol-or-s set
    def get_all_triplets_set(self):
        r = ['','','','','','']
        golden_set,multi_set,single_set = set(),set(),set()
        data_lines = open(self.data_dir)
        for i, text in enumerate(data_lines):
            sentence = re.search(r'.+(?=####)',text)[0]
            labels = eval(re.search(r'(?<=####).+',text)[0])
            r[0]=str(i)
            for j in labels:
                r[1],r[2] = str(j[0][0]),str(j[0][-1])
                r[3],r[4] = str(j[1][0]),str(j[1][-1])
                r[5] = str(s2id[j[2]])
                golden_set.add('-'.join(r))
                if i in self.multi_aspect_id:
                    multi_set.add('-'.join(r))
                else:
                    single_set.add('-'.join(r))
        return golden_set,multi_set,single_set


    #  i-al-ar set
    def get_all_aspect_set(self):
        triplets_golden_set = self.all_triplets
        golden_set = set()
        for i in triplets_golden_set:
            t = i.split('-')[:3]
            golden_set.add('-'.join(t))
        return golden_set

    # i-ol-or set
    def get_all_opinion_set(self):
        triplets_golden_set = self.all_triplets
        golden_set = set()
        for i in triplets_golden_set:
            t = i.split('-')[0:1]
            t.extend(i.split('-')[3:5])
            golden_set.add('-'.join(t))
        return golden_set

    # i-al-ar-ol-or set
    def get_all_pair_set(self):
        triplets_golden_set = self.all_triplets
        golden_set = set()
        for i in triplets_golden_set:
            t = i.split('-')[0:-1]
            golden_set.add('-'.join(t))
        return golden_set

    # i-al-ar-s set
    def get_all_as_set(self):
        triplets_golden_set = self.all_triplets
        golden_set = set()
        for i in triplets_golden_set:
            t = i.split('-')[0:3]
            t.append(i.split('-')[-1])
            golden_set.add('-'.join(t))
        return golden_set

    # dict {i':'al-ar-ol-or-s'}
    def get_triplets_dict(self):
        triplets_golden_list = list(self.all_triplets)
        r = {}
        for t in triplets_golden_list:
            key = t.split('-')[0]
            value = '-'.join(t.split('-')[1:])
            if key not in r.keys():
                r[key] = [value]
            else:
                r[key].append(value)
        return r

    def __len__(self):
        return self.len


