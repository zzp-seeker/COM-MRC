import torch,time,copy,os
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup,\
    get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
from model.comodel import comModel
from utils.logger import now

from utils.strategy import logits2aspect,logits2aspect_aspect

def train(args,train_data_loader,dev_data_loader,test_data_loader,three_goden_set_dev,three_goden_set_test,logger):
    logger.info(f'seed:{args.seed}')
    logger.info(f'dataset:{args.data_dir}')

    model = comModel(args).to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
            # "lr": 2e-5
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            # "lr": 1e-3
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * args.train_epochs * len(train_data_loader)), args.train_epochs * len(train_data_loader))

    best_t_f1_dev,t_f1_test = 0,0

    topk = 10
    k,min_loss = 0,100

    for i in range(args.train_epochs):
        t_s = time.time()
        loss_sum = []
        is_on_loss_sum,a_loss_sum,o_loss_sum,s_loss_sum = [],[],[],[]
        logger.info(f'Epoch:{i}')

        for batch in train_data_loader:
            model.train()
            optimizer.zero_grad()

            inputs_plus,label_plus = batch['inputs_plus'],batch['label_plus']
            B = inputs_plus['input_ids'].size(0) # 40

            y = model(inputs_plus,label_plus['a_s_'],label_plus['a_e_'],args)
            is_on_logits,as_p,ae_p = y['is_on_logits'],y['as_p'],y['ae_p']
            os_p,oe_p,s_logits = y['os_p'],y['oe_p'],y['s_logits']

            a_loss = (F.cross_entropy(as_p,label_plus['a_s'],ignore_index=-1)+
                      F.cross_entropy(ae_p,label_plus['a_e'],ignore_index=-1))/2
            o_loss = -torch.sum(F.log_softmax(os_p,dim=1).reshape([-1])*label_plus['o_s'].reshape([-1]))- \
                     torch.sum(F.log_softmax(oe_p,dim=1).reshape([-1])*label_plus['o_e'].reshape([-1]))
            o_loss = o_loss / (label_plus['o_s'].reshape([-1])!=0).nonzero().size(0)

            is_on_loss = F.cross_entropy(is_on_logits,label_plus['is_on'],ignore_index=-1)
            s_loss = F.cross_entropy(s_logits,label_plus['s'],ignore_index=-1)

            loss = is_on_loss*8 + a_loss * 3.2 + o_loss + s_loss

            loss_sum.append(loss)
            is_on_loss_sum.append(is_on_loss)
            a_loss_sum.append(a_loss)
            o_loss_sum.append(o_loss)
            s_loss_sum.append(s_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_sum = torch.sum(torch.stack(loss_sum)).item()/len(loss_sum)
        logger.info(f"aver loss:{loss_sum}")
        logger.info(f"is_on_loss:{torch.sum(torch.stack(is_on_loss_sum)).item()}")
        logger.info(f"a_loss_sum:{torch.sum(torch.stack(a_loss_sum)).item()}")
        logger.info(f"o_loss_sum:{torch.sum(torch.stack(o_loss_sum)).item()}")
        logger.info(f"s_loss_sum:{torch.sum(torch.stack(s_loss_sum)).item()}")
        logger.info(f"time:{time.time()-t_s}")

        if i>15:

            as_f,p_f,t_f = test(args,model,dev_data_loader,three_goden_set_dev,logger,is_test=False)
            if t_f >= best_t_f1_dev:
                logger.info(f'saved {i}th model')

                if os.path.exists(f"{args.model_dir}/{now}-{args.a}-{args.a_ww}-{args.b}-{args.b_ww}"):
                    torch.save(model.state_dict(), f'{args.model_dir}/{now}-{args.a}-{args.a_ww}-{args.b}-{args.b_ww}/model.pth')
                else:
                    torch.save(model.state_dict(), f'{args.model_dir}/model.pth')

                best_t_f1_dev = max(best_t_f1_dev,t_f)

    model = comModel(args).to(args.device)

    if os.path.exists(f"{args.model_dir}/{now}-{args.a}-{args.a_ww}-{args.b}-{args.b_ww}"):
        mt_f,st_f,t_f1_test = test(args,model, test_data_loader, three_goden_set_test,logger,
                                 is_test=True,
                                 model_dir=f'{args.model_dir}/{now}-{args.a}-{args.a_ww}-{args.b}-{args.b_ww}/model.pth')
        logger.info("BEST DEV Triplets F1:" + str(best_t_f1_dev))
        logger.info("Multi Aspect Triplets F1:" + str(mt_f))
        logger.info("Single Aspect Triplets F1:" + str(st_f))
        logger.info("TEST Triplets F1:" + str(t_f1_test))
    else:
        mt_f,st_f,t_f1_test = test(args,model, test_data_loader, three_goden_set_test,logger,
                             is_test=True,
                             model_dir=f'{args.model_dir}/model.pth')
        logger.info("BEST DEV Triplets F1:" + str(best_t_f1_dev))
        logger.info("Multi Aspect Triplets F1:" + str(mt_f))
        logger.info("Single Aspect Triplets F1:" + str(st_f))
        logger.info("TEST Triplets F1:" + str(t_f1_test))


def f_SE(s_,e_):
    r = []
    if s_.shape[0]!=0:
        e_ = e_[e_>=s_[0]]
    while s_.shape[0]!=0 and e_.shape[0]!=0:
        i = s_[s_<=e_[0]][-1]
        r.append((i,e_[0]))
        s_ = s_[s_>i]
        if s_.shape[0]!=0:
            e_ = e_[e_>=s_[0]]
    return r

def f_check(se1,se2):
    if se2[0]==se1[1] or se2[0]==se1[1]+1:
        return (se1[0],se2[1])
    elif se2[1]==se1[0] or se2[1]==se1[0]-1:
        return (se2[0],se1[1])
    return False

def f_join(t,se):
    r = []
    is_join = False
    flag = False
    for i,se_ in enumerate(se):
        check_r = f_check(t,se_)
        if check_r:
            t = check_r
            is_join = True
            flag = True
        elif is_join:
            r.append(t)
            r.append(se_)
            is_join = False
        else:
            r.append(se_)
    if not flag or (flag and is_join):
        r.append(t)
    return r

# inference algorithm 2
def test(args,model,test_data_loader,three_goden_set,logger,is_test=False,model_dir=''):
    model.eval()
    if is_test:
        model.load_state_dict(torch.load(model_dir))
    aspect_set,opinion_set,triplets_set,pair_set,as_set = set(),set(),set(),set(),set()
    multi_set,single_set = set(),set()

    multi_aspect_id = three_goden_set[2][3]

    with torch.no_grad():

        for i,batch in enumerate(test_data_loader):
            is_on,count = True,0
            inputs_plus,sentence_token_range = batch['inputs_plus_for_test'],batch['sentence_token_range'][0]
            extra_info = batch['extra_info']
            S = inputs_plus['input_ids'].size(1)

            token2word_idx = [] # token2word_idx['token_idx']=word_idx
            for ii,t in enumerate(sentence_token_range):
                for j in range(t[0],t[1]+1):
                    token2word_idx.append(ii)

            if is_test:
                logger.info(f'{i}-th golden:{three_goden_set[-1][str(i)]}')

            se_r = []
            inputs_plus_clone = copy.deepcopy(inputs_plus)
            while is_on:
                plus = model.get_aspect(inputs_plus_clone,args)
                as_p,ae_p,_,_,_ = plus

                a_spans = logits2aspect(as_p.squeeze(),ae_p.squeeze(),3,1,args.a,args.a_ww,args.a,token2word_idx,is_test)

                if len(a_spans)==0:
                    is_on = False
                else:
                    # cal is_on
                    as_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                    as_input[0][a_spans[0][0]]=1
                    ae_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                    ae_input[0][a_spans[0][1]]=1

                    y = model(inputs_plus_clone,as_input,ae_input,args,plus)
                    is_on_logits = y['is_on_logits']

                    if count!=0:
                        is_on = torch.argmax(is_on_logits,dim=1).squeeze().item()==1
                        if not is_on:
                            break

                    if args.use_c == 1:
                        se_r  = f_join((a_spans[0][0],a_spans[0][1]),se_r)
                    else:
                        se_r.append((a_spans[0][0],a_spans[0][1]))

                    inputs_plus_clone['attention_mask'][0][args.sen_pre_len+a_spans[0][0]:args.sen_pre_len+a_spans[0][1]+1]=0
                    count += 1

                    if count>10:
                        break

            for a_spans in se_r:
                inputs_plus['attention_mask'][0][args.sen_pre_len+a_spans[0]:args.sen_pre_len+a_spans[1]+1]=0

            for a_spans in se_r:
                as_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                as_input[0][a_spans[0]]=1
                ae_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                ae_input[0][a_spans[1]]=1

                inputs_plus['attention_mask'][0][args.sen_pre_len+a_spans[0]:args.sen_pre_len+a_spans[1]+1]=1
                y = model(inputs_plus,as_input,ae_input,args)
                inputs_plus['attention_mask'][0][args.sen_pre_len+a_spans[0]:args.sen_pre_len+a_spans[1]+1]=0

                os_p,oe_p,s_logits = y['os_p'],y['oe_p'],y['s_logits']

                a_span =  get_aspect(i,a_spans[0],a_spans[1],sentence_token_range)
                aspect_set.add(a_span)

                o_spans = logits2aspect(os_p.squeeze(),oe_p.squeeze(),3,3,args.b,args.b_ww,args.b,token2word_idx,is_test)


                o_spans = get_opinion_spans(i,o_spans,sentence_token_range)
                for o in o_spans:
                    opinion_set.add(o)

                s = torch.argmax(s_logits,dim=1).squeeze().item()
                as_pair,pair,triplets = get_as_pair_and_triplets(a_span,o_spans,s)
                for t in pair:
                    pair_set.add(t)
                for t in triplets:
                    triplets_set.add(t)
                    if i in multi_aspect_id:
                        multi_set.add(t)
                    else:
                        single_set.add(t)

                for t in as_pair:
                    as_set.add(t)

                if is_test:
                    logger.info(f'a_span:{a_span}')
                    logger.info(f'o_spans:{o_spans}')
                    logger.info(f'triplets:{triplets}')


    a_p, a_r, a_f = score_set(aspect_set,three_goden_set[0])
    o_p, o_r, o_f = score_set(opinion_set,three_goden_set[1])
    t_p, t_r, t_f = score_set(triplets_set,three_goden_set[2][0])
    mt_p,mt_r,mt_f = score_set(multi_set,three_goden_set[2][1])
    st_p,st_r,st_f = score_set(single_set,three_goden_set[2][2])
    # p_p,p_r,p_f = score_set(pair_set,three_goden_set[3])
    # as_p,as_r,as_f = score_set(as_set,three_goden_set[4])


    # logger.info("aspect p,r,f1: " + '%.5f'%a_p +' '+ '%.5f'%a_r +' '+ '%.5f'%a_f)
    # logger.info("opinion p,r,f1: " + '%.5f'%o_p +' '+ '%.5f'%o_r +' '+ '%.5f'%o_f)
    logger.info("multi a p,r,f1: " + '%.5f'%mt_p +' '+ '%.5f'%mt_r +' '+ '%.5f'%mt_f)
    logger.info("single a p,r,f1: " + '%.5f'%st_p +' '+ '%.5f'%st_r +' '+ '%.5f'%st_f)
    logger.info("triplets p,r,f1: " + '%.5f'%t_p +' '+ '%.5f'%t_r +' '+ '%.5f'%t_f)


    return mt_f,st_f,t_f

def get_aspect(i,a_s,a_e,sentence_token_range):
    r_s,r_e = 0,0
    for index,span in enumerate(sentence_token_range):
        if a_s>=span[0] and a_s<=span[1]:
            r_s = index
        if a_e>=span[0] and a_e<=span[1]:
            r_e = index
            break
    return '-'.join([str(i),str(r_s),str(r_e)])


def get_opinion_spans(idx,spans,sentence_token_range):
    result = []
    l,r=0,0
    for span in spans:
        for i,t in enumerate(sentence_token_range):
            if span[0]>=t[0] and span[0]<=t[1]:
                l=i
            if span[1]>=t[0] and span[1]<=t[1]:
                r=i
                result.append('-'.join([str(idx),str(l),str(r)]))
                break

    return result

# a: i-al-ar  o: [i-ol-or]  s: s
def get_as_pair_and_triplets(a,o,s):
    pair_result = []
    trip_result = []
    as_result = []
    for o_i in o:
        tt = a.split('-')
        tt.append(str(s))
        as_result.append('-'.join(tt))

        t = a.split('-')
        t.extend(o_i.split('-')[1:])
        pair_result.append('-'.join(t))
        t.append(str(s))
        trip_result.append('-'.join(t))
    return as_result,pair_result,trip_result

def score_set(predict_set , golden_set):
    correct_num = len(golden_set & predict_set)
    precision = correct_num / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# inference algorithm 1
def test2(args,model,test_data_loader,three_goden_set,logger,is_test=False,model_dir=''):

    model.eval()
    if is_test:
        # model.load_state_dict(torch.load(model_dir, map_location={'cuda:2': 'cuda:3'}))
        model.load_state_dict(torch.load(model_dir))
    aspect_set,opinion_set,triplets_set = set(),set(),set()
    multi_set,single_set = set(),set()
    multi_aspect_id = three_goden_set[2][3]
    with torch.no_grad():

        for i,batch in enumerate(test_data_loader):
            is_on,count = True,0
            inputs_plus,sentence_token_range = batch['inputs_plus_for_test'],batch['sentence_token_range'][0]
            extra_info = batch['extra_info']
            S = inputs_plus['input_ids'].size(1)

            token2word_idx = [] # token2word_idx['token_idx']=word_idx
            for ii,t in enumerate(sentence_token_range):
                for j in range(t[0],t[1]+1):
                    token2word_idx.append(ii)

            if is_test:
                logger.info(f'{i}-th golden:{three_goden_set[-1][str(i)]}')

            se_r = []
            inputs_plus_clone = copy.deepcopy(inputs_plus)
            # inputs_plus_clone = inputs_plus.clone()
            while is_on:
                plus = model.get_aspect(inputs_plus_clone,args)
                as_p,ae_p,_,_,_ = plus

                # a_spans =  logits2aspect_aspect(as_p.squeeze(),ae_p.squeeze(),3,1,4,0.5,4,token2word_idx,is_test,i)
                a_spans = logits2aspect(as_p.squeeze(),ae_p.squeeze(),3,1,args.a,args.a_ww,args.a,token2word_idx,is_test)


                if len(a_spans)==0:
                    is_on = False
                else:


                    as_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                    as_input[0][a_spans[0][0]]=1
                    ae_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                    ae_input[0][a_spans[0][1]]=1

                    y = model(inputs_plus_clone,as_input,ae_input,args,plus)
                    is_on_logits = y['is_on_logits']

                    if count!=0:
                        is_on = torch.argmax(is_on_logits,dim=1).squeeze().item()==1
                        if not is_on:
                            break


                    # se_r  = f_join((a_spans[0][0],a_spans[0][1]),se_r)


                    if args.use_c == 1:
                        se_r  = f_join((a_spans[0][0],a_spans[0][1]),se_r)
                    else:
                        se_r.append((a_spans[0][0],a_spans[0][1]))

                    inputs_plus_clone['attention_mask'][0][args.sen_pre_len+a_spans[0][0]:args.sen_pre_len+a_spans[0][1]+1]=0
                    count += 1

                    if count>10:
                        break

            for a_spans in se_r:
                as_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                as_input[0][a_spans[0]]=1
                ae_input = torch.tensor([0] * (S-args.sen_pre_len)).unsqueeze(0).to(args.device)
                ae_input[0][a_spans[1]]=1

                y = model(inputs_plus,as_input,ae_input,args)
                os_p,oe_p,s_logits = y['os_p'],y['oe_p'],y['s_logits']

                a_span =  get_aspect(i,a_spans[0],a_spans[1],sentence_token_range)
                aspect_set.add(a_span)


                # o_spans = logits2aspect(os_p.squeeze(),oe_p.squeeze(),3,3,5,1,5,token2word_idx,is_test)
                o_spans = logits2aspect(os_p.squeeze(),oe_p.squeeze(),3,3,args.b,args.b_ww,args.b,token2word_idx,is_test)

                o_spans = get_opinion_spans(i,o_spans,sentence_token_range)
                for o in o_spans:
                    opinion_set.add(o)

                s = torch.argmax(s_logits,dim=1).squeeze().item()
                as_pair,pair,triplets = get_as_pair_and_triplets(a_span,o_spans,s)

                for t in triplets:
                    triplets_set.add(t)
                    if i in multi_aspect_id:
                        multi_set.add(t)
                    else:
                        single_set.add(t)

                if is_test:
                    logger.info(f'a_span:{a_span}')
                    logger.info(f'o_spans:{o_spans}')
                    logger.info(f'triplets:{triplets}')

                inputs_plus['attention_mask'][0][args.sen_pre_len+a_spans[0]:args.sen_pre_len+a_spans[1]+1]=0

    t_p, t_r, t_f = score_set(triplets_set,three_goden_set[2][0])
    mt_p,mt_r,mt_f = score_set(multi_set,three_goden_set[2][1])
    st_p,st_r,st_f = score_set(single_set,three_goden_set[2][2])

    logger.info("multi a p,r,f1: " + '%.5f'%mt_p +' '+ '%.5f'%mt_r +' '+ '%.5f'%mt_f)
    logger.info("single a p,r,f1: " + '%.5f'%st_p +' '+ '%.5f'%st_r +' '+ '%.5f'%st_f)
    logger.info("triplets p,r,f1: " + '%.5f'%t_p +' '+ '%.5f'%t_r +' '+ '%.5f'%t_f)


    return mt_f,st_f,t_f

