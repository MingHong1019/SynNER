import random
import numpy as np
import torch
import os
from transformers import BertTokenizer
from src.trainer_sentence_index_wexample import Trainer_sentence_index_wexample
from src.trainer_batch_sentence_index_wexample import Trainer_batch_sentence_index_wexample
from src.trainer_batch_sentence_noindex_wexample import Trainer_batch_sentence_noindex_wexample
from src.trainer_only_llm import Trainer_llm
from src.train_dataloader import get_train_loader
from src.test_dataloader import  get_test_loader
from utils.data_reader import germ, split_data, nerel, genia, ace04
from utils.data_reader import ace05, ace05_chinese, vlsp18, vlsp16,label_num
from utils.get_llm_example import get_llm_example

from args import argparser
from openai import OpenAI
import httpx
import time
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparser()
    args = parser.parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda:" + str(args.select_gpu))
    args.device = device
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    args.tokenizer = tokenizer
    model_path = f'checkpoint/{args.model_name}.pkl'
    N, K = args.train_ways, args.train_shots
    if args.llm_revise_level == 'sentence_index_we':
        trainer = Trainer_sentence_index_wexample(args)
    elif args.llm_revise_level == 'batch_sentence_noindex_we':
        trainer = Trainer_batch_sentence_noindex_wexample(args)
    elif args.llm_revise_level == 'batch_sentence_index_we':
        trainer = Trainer_batch_sentence_index_wexample(args)
    elif args.llm_revise_level == 'only_llm':
        trainer = Trainer_llm(args)
    else:
        print("[ERROR] llm_revise_level wrong!")
        assert(0)
    if args.do_train:
        print('training......')
        train_path = f'{args.train_path}/train_{N}_{K}.jsonl' 
        dev_path = f'{args.train_path}/dev_{N}_{K}.jsonl'
        train_data_loader = get_train_loader(train_path, args)
        val_data_loader = get_train_loader(dev_path, args, neg_sample=False)
        trainer.train(train_data_loader, val_data_loader, model_path=model_path)
    
    if args.do_predict:
        print('test......')
        # model_path = f'checkpoint\conbsr_10.pkl'
        result = []
        if args.do_llm_revise:
            client = OpenAI(
            base_url="https://api.xty.app/v1",
            api_key="sk-ENSwczEtjQe1r3Ue9d46E7C77758488aB99d8a3bC69e7e34",
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True),
            )
            args.llm = client

        if args.test_data == 'fewnerd':
            print('test on fewnerd')
            test_path = f'{args.train_path}/test_{N}_{K}.jsonl' 
            test_data_loader = get_train_loader(test_path, args, neg_sample=False)
            for seed in range(10):
                set_seeds(seed)
                f1 = trainer.test_fewnerd(test_data_loader,model_path= model_path)
                result.append(f1)
            
        else:
            test_path = args.test_path
            if args.test_data == 'genia':
                if test_path == '':
                    test_path = r"../../data/genia/GENIAcorpus3.02.xml"
                sentence, span, label, total_type = genia(test_path)

            elif args.test_data == 'nerel':
                if test_path == '':
                    test_path = r"../../data/NEREL/NEREL-v1.1/test"
                sentence, span, label, total_type = nerel(test_path)

            elif args.test_data == 'germ':
                if test_path == '':
                    test_path = r"../../data/GermEval"
                sentence, span, label, total_type = germ(test_path)

            elif args.test_data == 'ace04':
                if test_path == '':
                    test_path = r"../../data/ACE2004"
                sentence, span, label, total_type = ace04(test_path)

            elif args.test_data == 'ace05':
                if test_path == '':
                    test_path = r"../../data/ACE2005"
                sentence, span, label, total_type = ace05(test_path, args.ace05_type)
            
            elif args.test_data == 'ace05_chinese':
                if test_path == '':
                    test_path = r"../../data/ACE2005_Chinese"
                sentence, span, label, total_type = ace05_chinese(test_path, args.ace05_type)

            elif args.test_data == 'vlsp18':
                if test_path == '':
                    test_path = r"../../data/VLSP2018"
                sentence, span, label, total_type = vlsp18(test_path)

            elif args.test_data == 'vlsp16':
                if test_path == '':
                    test_path = r"../../data/VLSP2016"
                sentence, span, label, total_type = vlsp16(test_path)

            else:
                print("[ERROR] test data must be genia, nerel, germ, ace04, ace05, vlsp16 or vlsp18")
                assert (0)

            
            total_label_num = label_num(label, total_type)
            ignore_label = [key for key, value in total_label_num.items() if value < args.test_shots]
            total_type = [i for i in total_type if i not in ignore_label]
            args.test_class = len(total_type) 



            set_seeds(args.random_seed)
            train_sen, train_span, train_label, test_sen, \
            test_span, test_label  = split_data(sentence, span, label, total_type, args.test_shots)
            args.llm_example = get_llm_example(train_sen, train_span, train_label, total_type, args.test_shots)
            total_type_witho = ['O'] + total_type
            
        
            finetune_dataloader = get_test_loader(args=args, sentence=train_sen, span=train_span,
                                                    label=train_label, total_type=total_type_witho, 
                                                    batch_size=args.finetune_batchsize,neg_sample=True)
            test_dataloader = get_test_loader(args=args, sentence=test_sen, span=test_span,
                                                    label=test_label, total_type=total_type_witho, 
                                                    batch_size=1,shuffle=False, neg_sample=False )

            test_start_time = time.perf_counter()
            f1 = trainer.test(finetune_dataloader, test_dataloader, model_path=model_path)


            test_end_time = time.perf_counter()
            testrunTime = test_end_time - test_start_time
            print("测试时间", testrunTime / 60, ' 分')



if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('log'):
        os.mkdir('log')
    main()