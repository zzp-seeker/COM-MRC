import json,os
valueTrans = {'positive':'POS','negative':'NEG','neutral':'NEU'}

def BIO_2_SE(x):
    x_list = x.split(' ')
    r,s,e = [],-1,-1
    for i,xi in enumerate(x_list):
        if xi.find(r'\B')!=-1:
            s = i
        if s!=-1 and (i==len(x_list)-1 or x_list[i+1].find(r'\I')==-1):
            e = i
            if s==e:
                r.append([s])
            else:
                r.append([s,e])
            s,e=-1,-1
    return r

data_name = ['14lap','14res','15res','16res']

for name in data_name:
    data_dir = r'./'+ name +r'/'
    modes = ['train','dev','test']
    for mode in modes:
        read_dir = data_dir + mode + r'.json'
        write_dir = data_dir + mode + r'_triplets.txt'

        if os.path.exists(write_dir):
            os.remove(write_dir)

        if not os.path.exists(write_dir):
            os.mknod(write_dir)


        with open(read_dir,'r') as f:
            data = json.load(f)
            for index,data_i in enumerate(data):
                sentence,triplets = data_i['sentence'],data_i['triples']
                r = []
                if(len(data_i['triples'])>4):
                    print(f'{name}-{mode}-{index}-num:{len(data_i["triples"])}')
                for triplet in triplets:


                    a_span = BIO_2_SE(triplet['target_tags'])
                    o_spans = BIO_2_SE(triplet['opinion_tags'])
                    sentiment = valueTrans[triplet['sentiment']]
                    if len(a_span)!=1:
                        raise ValueError
                    for o_span in o_spans:
                        # r.append((triplet['uid'],a_span[0],o_span,sentiment))
                        g = (a_span[0],o_span,sentiment)
                        if g not in r:
                            r.append(g)
                        else:
                            r.append(g)
                            # print("repeat multi times!")
                            pass
                result = sentence+'####'+str(r)+'\n'
                with open(write_dir,'a+') as f:
                    f.write(result)








