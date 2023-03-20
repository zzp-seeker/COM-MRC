import torch,math
import numpy as np
from constant import *

from utils.sampler import BatchSampler,RandomSampler,SequentialSampler

class Instance:
    def __init__(self,data,tokenizer,max_length,plus_text):
        self.tokenizer = tokenizer
        self.sentence = data['sentence']
        self.plus_text = plus_text
        self.sentence_token = tokenizer.encode(self.sentence)[1:-1]
        self.plus_text_token = tokenizer.encode(self.plus_text)[1:-1]

        self.sentence_token_range = self.get_sen_tokenrange(tokenizer)

        self.triplets = []
        for t in data['triplets']:
            triplet = {"aspect_span":[],"opinion_span_list":[],'sentiment':0}
            triplet["aspect_span"]=[self.sentence_token_range[t[0][0]][0],
                                    self.sentence_token_range[t[0][-1]][1]]
            for o in t[1]:
                triplet["opinion_span_list"].append([
                    self.sentence_token_range[o[0]][0],
                    self.sentence_token_range[o[-1]][1]
                ])
            triplet['sentiment'] = s2id[t[2]]
            self.triplets.append(triplet)


    def get_sen_tokenrange(self,tokenizer):
        token_range,token_start = [],0
        for i,t in enumerate(self.sentence.strip().split()):
            token_end = token_start + len(tokenizer.encode(t,add_special_tokens=False))
            token_range.append([token_start,token_end-1])
            token_start = token_end
        return token_range


class DataLoader:
    def collate_fn(self,args,batch):
        sentence,plus_text = [],[]
        sentence_token_range = []
        triplets_all = []
        sentence_all_token_len = []
        sentence_token_len = []

        for ins in batch:
            sentence.append(ins.sentence)
            plus_text.append(ins.plus_text)
            sentence_all_token_len.append(len(self.tokenizer.encode(ins.plus_text,ins.sentence)))
            sentence_token_len.append(len(ins.sentence_token))
            sentence_token_range.append(ins.sentence_token_range)
            triplets_all.append(ins.triplets)

        inputs = self.tokenizer(plus_text,sentence,padding=True,return_tensors='pt')
        inputs_plus,label_plus,inputs_plus_for_test = self.get_plus(args,inputs,triplets_all,sentence_token_len)


        for cuda_var in [inputs_plus,label_plus,inputs_plus_for_test]:
            for k,v in cuda_var.items():
                cuda_var[k] = v.to(args.device)

        extra_info = {
            'sentence': sentence
        }


        return {
            'inputs_plus':inputs_plus,'label_plus':label_plus,'inputs_plus_for_test':inputs_plus_for_test,
            'sentence_token_range':sentence_token_range,'extra_info':extra_info
        }


    def get_plus(self,args,inputs,triplets_all,sentence_token_len):
        B,S = inputs['input_ids'].size(0),inputs['input_ids'].size(1)

        def get_opinion(i,j):
            return self.opinion_span_list2SE(triplets_all[i][j]['opinion_span_list'],
                                             S-args.sen_pre_len)

        inputs_plus = {
            'input_ids':inputs['input_ids'][0:1].clone(),
            'token_type_ids':inputs['token_type_ids'][0:1].clone(),
            'attention_mask':inputs['attention_mask'][0:1].clone()
        }
        label_plus = {
            'a_s':torch.tensor([triplets_all[0][0]['aspect_span'][0]]),
            'a_e':torch.tensor([triplets_all[0][0]['aspect_span'][1]]),
            'a_s_':torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0),
            'a_e_':torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0),
            'mask':torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0),
            'is_on': torch.tensor([1]),
            'o_s':torch.tensor(get_opinion(0,0)[0]).unsqueeze(0),
            'o_e':torch.tensor(get_opinion(0,0)[1]).unsqueeze(0),
            's':torch.tensor([triplets_all[0][0]['sentiment']])
        }

        label_plus['a_s_'][0][triplets_all[0][0]['aspect_span'][0]]=1
        label_plus['a_e_'][0][triplets_all[0][0]['aspect_span'][1]]=1
        label_plus['mask'][0][:sentence_token_len[0]]=1

        inputs_plus_for_test = {
            'input_ids':inputs['input_ids'][0:1].clone(),
            'token_type_ids':inputs['token_type_ids'][0:1].clone(),
            'attention_mask':inputs['attention_mask'][0:1].clone()
        }

        for i in range(B):
            triplets = triplets_all[i]
            t_mask = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0)
            t_mask[0][:sentence_token_len[i]] = 1
            for j in range(len(triplets)+1):
                l = self.get_all_bin(len(triplets)-j-1)

                if l==-1:
                    if args.use_context_augmentation==-1:
                        continue
                    inputs_plus['token_type_ids'] = torch.cat((inputs_plus['token_type_ids'],inputs['token_type_ids'][i:i+1].clone()))
                    inputs_plus['input_ids'] = torch.cat((inputs_plus['input_ids'],inputs['input_ids'][i:i+1].clone()))

                    t = inputs_plus['attention_mask'][-1:].clone()
                    aspect_span = triplets[j-1]['aspect_span']
                    t[0][args.sen_pre_len+aspect_span[0]:args.sen_pre_len+aspect_span[1]+1]= 0
                    inputs_plus['attention_mask'] = torch.cat((inputs_plus['attention_mask'],t))

                    label_plus['a_s']=torch.cat((label_plus['a_s'],torch.tensor([s2id['Invalid']])))
                    label_plus['a_e']=torch.cat((label_plus['a_e'],torch.tensor([s2id['Invalid']])))
                    label_plus['a_s_'] = torch.cat((label_plus['a_s_'],torch.tensor(
                        [0] * (S-args.sen_pre_len)).unsqueeze(0)))
                    label_plus['a_e_'] = torch.cat((label_plus['a_e_'],torch.tensor(
                        [0] * (S-args.sen_pre_len)).unsqueeze(0)))

                    label_plus['is_on']=torch.cat((label_plus['is_on'],torch.tensor([0])))
                    label_plus['o_s'] = torch.cat((label_plus['o_s'],torch.tensor(
                        [0] * (S-args.sen_pre_len)).unsqueeze(0)))
                    label_plus['o_e'] = torch.cat((label_plus['o_e'],torch.tensor(
                        [0] * (S-args.sen_pre_len)).unsqueeze(0)))
                    label_plus['s']=torch.cat((label_plus['s'],torch.tensor([s2id['Invalid']])))

                    label_plus['mask']=torch.cat((label_plus['mask'],t_mask))

                    continue

                for k in range(len(l)-1,-1,-1):
                    if args.use_context_augmentation==0 and (not (l[k].count('0')==0 or l[k].count('1')==0)):
                        continue
                    if args.use_context_augmentation==-1 and not (j==0 and l[k].count('1')==0):
                        continue

                    inputs_plus['token_type_ids'] = torch.cat((inputs_plus['token_type_ids'],inputs['token_type_ids'][i:i+1].clone()))
                    inputs_plus['input_ids'] = torch.cat((inputs_plus['input_ids'],inputs['input_ids'][i:i+1].clone()))

                    t = inputs['attention_mask'][i:i+1].clone()
                    if j!=0:
                        for index in range(j):
                            aspect_span = triplets[index]['aspect_span']
                            t[0][args.sen_pre_len+aspect_span[0]:args.sen_pre_len+aspect_span[1]+1]= 0
                    for index,mask_i in enumerate(l[k]):
                        if mask_i=='1':
                            aspect_span = triplets[index+j+1]['aspect_span']
                            t[0][args.sen_pre_len+aspect_span[0]:args.sen_pre_len+aspect_span[1]+1]= 0
                    inputs_plus['attention_mask'] = torch.cat((inputs_plus['attention_mask'],t))

                    triplet = triplets[j]
                    label_plus['a_s']=torch.cat((label_plus['a_s'],torch.tensor([triplet['aspect_span'][0]])))
                    label_plus['a_e']=torch.cat((label_plus['a_e'],torch.tensor([triplet['aspect_span'][1]])))
                    label_plus['is_on']=torch.cat((label_plus['is_on'],torch.tensor([1])))
                    label_plus['o_s'] = torch.cat((label_plus['o_s'],torch.tensor(get_opinion(i,j)[0]).unsqueeze(0)))
                    label_plus['o_e'] = torch.cat((label_plus['o_e'],torch.tensor(get_opinion(i,j)[1]).unsqueeze(0)))
                    label_plus['s']=torch.cat((label_plus['s'],torch.tensor([triplet['sentiment']])))

                    t_as = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0)
                    t_as[0][triplet['aspect_span'][0]] = 1
                    label_plus['a_s_']=torch.cat((label_plus['a_s_'],t_as))

                    t_ae = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0)
                    t_ae[0][triplet['aspect_span'][1]] = 1
                    label_plus['a_e_']=torch.cat((label_plus['a_e_'],t_ae))

                    label_plus['mask']=torch.cat((label_plus['mask'],t_mask))

        return inputs_plus,label_plus,inputs_plus_for_test


    def opinion_span_list2BIO(self,args,opinion_span_list,max_length,token_length):
        r = [term2id['Invalid']] * max_length
        r[args.sen_pre_len:token_length-1] = [term2id['O']] * (token_length-1-args.sen_pre_len)
        for span in opinion_span_list:
            for index in range(span[0],span[1]+1):
                if index==span[0]:
                    r[args.sen_pre_len+index]=term2id['B']
                else:
                    r[args.sen_pre_len+index]=term2id['I']
        return r

    def get_all_bin(self,l):
        if l<0: return -1
        r = []
        x = int(math.pow(2,l))
        for i in  range(x):
            v = f'{i:b}'.rjust(l,'0')
            r.append(v)
        return r

    def opinion_span_list2SE(self,opinion_span_list,max_length):

        s = [0] * max_length
        e = [0] * max_length
        for span in opinion_span_list:
            s[span[0]]=1
            e[span[1]]=1
        return s,e

    def __init__(self, instances, tokenizer, args,  shuffle=False, drop_last=False , is_test=False):
        self.instances = instances
        self.args = args
        self.tokenizer = tokenizer
        if shuffle:
            if is_test:
                self.batch_sampler = BatchSampler(1, SequentialSampler(len(self.instances)), drop_last)
            else:
                self.batch_sampler = BatchSampler(args.batch_size, RandomSampler(len(self.instances)), drop_last)

        else:
            if is_test:
                self.batch_sampler = BatchSampler(1, SequentialSampler(len(self.instances)), drop_last)
            else:
                self.batch_sampler = BatchSampler(args.batch_size, SequentialSampler(len(self.instances)), drop_last)


    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.collate_fn(self.args,[self.instances[idx] for idx in batch_indices])

    def __len__(self):
        return len(self.batch_sampler)




