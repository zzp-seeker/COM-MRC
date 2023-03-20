import argparse,time,torch,random,numpy as np

from model.comodel import comModel
from utils.loader_v2 import RawData
from utils.data2 import DataLoader,Instance
from fastNLP import cache_results
from transformers import  BertTokenizer
from utils.logger import Logger

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument('--mode',type=str,choices=["train", "test"],
                    default='train')
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--data_dir',type=str,
                    default='../Dataset/ASTE-Data-V2/14lap')
parser.add_argument('--cache_path', type=str,
                    default="../main_cache",
                    help='pretrained bert model cache path')
parser.add_argument('--bert_type', type=str,default="bert-base-uncased",)
parser.add_argument("--lr", type=float, default=9e-5)
parser.add_argument('--use_query', type=int, default=1)
parser.add_argument('--use_context_augmentation', type=int, default=-1)
parser.add_argument('--a', type=float, default=5) # inference parameter
parser.add_argument('--a_ww', type=float, default=0.5)
parser.add_argument('--b', type=float, default=11)
parser.add_argument('--b_ww', type=float, default=1)
parser.add_argument('--use_c', type=int, default=1)
parser.add_argument('--max_length', type=int, default=100)

args = parser.parse_args()
save_dir = '../COM-MRC-code/result'


from utils.logger import now
import os
args.model_dir = f"{save_dir}/COMMRC-{args.seed}"
filepath = f"{save_dir}/COMMRC-{args.seed}"
if os.path.exists(filepath):
    filepath = f"{save_dir}/COMMRC-{args.seed}/{now}-{args.a}-{args.a_ww}-{args.b}-{args.b_ww}"
logger_class = Logger(filename='COMMRC',filepath=filepath)
logger = logger_class.logger
seed_everything(args.seed)

from trainer import train,test,test2
tokenizer = BertTokenizer.from_pretrained(args.bert_type, cache_dir=args.cache_path)
args.text = 'Find the first aspect term and corresponding opinion term in the text' # 14
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
if args.use_query == 0: args.text = '' # 2
args.sen_pre_len = len(tokenizer.encode(args.text))

# @cache_results(
#     f"./caches/{args.data_dir[-5:]}.pkl",_refresh=False
# )
def load_data_instances():
    raw_train_data = RawData(f'{args.data_dir}/train_triplets.txt')
    instances_train = []
    for data in raw_train_data:
        instances_train.append(Instance(data,tokenizer,args.max_length,
                        plus_text=args.text))

    raw_dev_data = RawData(f'{args.data_dir}/dev_triplets.txt')
    aspect_golden_set = raw_dev_data.get_all_aspect_set()
    opinion_goden_set = raw_dev_data.get_all_opinion_set()

    triplets_goden_set = raw_dev_data.all_triplets
    multi_set = raw_dev_data.all_multi_triplets
    single_set = raw_dev_data.all_single_triplets
    multi_aspect_id = raw_dev_data.multi_aspect_id

    pair_goden_set = raw_dev_data.get_all_pair_set()
    as_goden_set = raw_dev_data.get_all_as_set()
    triplets_goden_dict = raw_dev_data.get_triplets_dict()

    instances_dev = []
    for data in raw_dev_data:
        instances_dev.append(Instance(data,tokenizer,args.max_length,
                                       plus_text=args.text))

    raw_test_data = RawData(f'{args.data_dir}/test_triplets.txt')
    aspect_golden_set2 = raw_test_data.get_all_aspect_set()
    opinion_goden_set2 = raw_test_data.get_all_opinion_set()
    triplets_goden_set2 = raw_test_data.all_triplets

    multi_set2 = raw_test_data.all_multi_triplets
    single_set2 = raw_test_data.all_single_triplets
    multi_aspect_id2 = raw_test_data.multi_aspect_id

    pair_goden_set2 = raw_test_data.get_all_pair_set()
    as_goden_set2 = raw_test_data.get_all_as_set()
    triplets_goden_dict2 = raw_test_data.get_triplets_dict()

    instances_test = []
    for data in raw_test_data:
        instances_test.append(Instance(data,tokenizer,args.max_length,
                                       plus_text=args.text))

    return instances_train, instances_dev,instances_test,\
           [aspect_golden_set,opinion_goden_set,(triplets_goden_set,multi_set,single_set,multi_aspect_id),pair_goden_set,as_goden_set,raw_dev_data.multi_aspect_id,triplets_goden_dict],\
           [aspect_golden_set2,opinion_goden_set2,(triplets_goden_set2,multi_set2,single_set2,multi_aspect_id2),pair_goden_set2,as_goden_set2,raw_test_data.multi_aspect_id,triplets_goden_dict2],

instances_train,instances_dev,instances_test,three_goden_set_dev,three_goden_set_test = load_data_instances()
train_data_loader = DataLoader(instances_train,tokenizer,args)
dev_data_loader = DataLoader(instances_dev,tokenizer,args,is_test=True)
test_data_loader = DataLoader(instances_test,tokenizer,args,is_test=True)
train(args,train_data_loader,dev_data_loader,test_data_loader,three_goden_set_dev,three_goden_set_test,logger)

# model = comModel(args).to(args.device)
# test(args,model, test_data_loader, three_goden_set_test, logger,
#      is_test=True,
#      model_dir='***.pth')

