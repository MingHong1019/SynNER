from utils.evaluate import Evaluate_onlyLLM
from src.prompt_generate import get_llm_revise_prompt_we
from src.trainer import Trainer

from torch.optim import AdamW
from tqdm import tqdm
import copy
import torch
import jsonlines
import logging
import os
import json
import re

class HttpFilter(logging.Filter):
    def filter(self, record):
        # 排除包含'HTTP'的日志记录
        return 'HTTP' not in record.getMessage()

class Trainer_llm(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)
        

    def test(self, finetune_data_loader, target_test_loader, model_path):

        print('================  start   test    ======================')
        logging.basicConfig(filename=f'log/{self.args.model_name}_{self.args.test_data}_{self.args.test_shots}shots_{self.args.llm_refine_batch_size}batchsize_{self.args.llm_refine_batch_threshold}threshold_{self.args.llm_revise_level}_{self.args.llm_model}.log', level=logging.INFO, \
            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
         # 添加自定义过滤器
        http_filter = HttpFilter()
        for handler in logger.handlers:
            handler.addFilter(http_filter)

        print(f'only_llm')
        print(f'test task: {self.args.test_data}_{self.args.test_shots}shots')

        it = 0
        eval_iter =  min(5000, len(target_test_loader))
        # eval_iter =  3000
        evaluate = Evaluate_onlyLLM()

        revise_sentence = []
        before_revise = {}
        batch_true, batch_query = [], []
        revise_batch_size = self.args.llm_refine_batch_size
        result_file_name = f'result/onlyLLM_{self.args.test_data}_{self.args.test_shots}shots_{self.args.llm_refine_batch_size}batchsize.jsonl'

        already_refine_idx = []
        already_refine_pred_word = []
        already_refine_pred_label = []
 
        already_refine_true_word = []
        already_refine_true_label = []


        if os.path.exists(result_file_name):
            with open(result_file_name, 'r+') as f:
                for items in jsonlines.Reader(f):
                    already_refine_idx.append(items['idx'])
                    already_refine_pred_word.append(items['pred_word'])
                    already_refine_pred_label.append(items['pred_type'])
                    already_refine_true_word.append(items['true_word'])
                    already_refine_true_label.append(items['true_type'])

        for query in tqdm(target_test_loader, total=eval_iter, desc=f'testing {self.args.test_data}', ncols=80, ascii=True):
            if  query['data_idx'][0] in already_refine_idx:
                idx = already_refine_idx.index(query['data_idx'][0])
                pred_word = already_refine_pred_word[idx]
                pred_type = already_refine_pred_label[idx]
                true_word = already_refine_true_word[idx]
                true_type = already_refine_true_label[idx]
                evaluate.collect_data( pred_word, true_word, pred_type, true_type)

            else:
        
                origin_sentence = query['sentence'][0]
                revise_sentence.append(origin_sentence)
                true_label = torch.cat(query['entity_types'], 0)
                
                batch_true.append(true_label)
                batch_query.append(query)


            if revise_sentence!=[] and len(revise_sentence) % revise_batch_size == 0:
                batch_result = self.batch_llm_revise_wo_sentence(revise_sentence, batch_query, \
                                                                batch_true, query['label2tag'], logger)
                for key, i in batch_result.items():
                    pred_word, true_word = i['pred_word'], i['true_word']
                    pred_type, true_type = i['pred_type'], i['true_type']
                    evaluate.collect_data(pred_word, true_word, pred_type, true_type)
                    evaluate.save_jsonline(result_file_name, i)
         
                revise_sentence = []
                before_revise = {}
                batch_true, batch_query = [], []

            if it >= eval_iter:
                break
            it += 1


        if revise_sentence:
            batch_result = self.batch_llm_revise_wo_sentence(revise_sentence, batch_query, \
                                                                batch_true, query['label2tag'], logger)
            for key, i in batch_result.items():
                pred_word, true_word = i['pred_word'], i['true_word']
                pred_type, true_type = i['pred_type'], i['true_type']
                evaluate.collect_data(pred_word, true_word, pred_type, true_type)
                evaluate.save_jsonline(result_file_name, i)

        
        f1 = evaluate.get_f1()
        print(f'{self.args.test_data}_{self.args.test_shots}shot')
        print(round(f1*100,2))
        # print(num_change_label)
        return f1

    def batch_llm_revise_wo_sentence(self, revise_sentence, query, true_label, label2tag, logging):
        num_changed = 0
        batch_result = {}
        idx = 1
        revise_result = {}
        prompt = get_llm_revise_prompt_we(self.args.llm_example, self.args.test_data)
        prompt += f'Now, for these test sentences: \n'

        for idx, sentence in enumerate(revise_sentence):
            temp_sentence = ' '.join(sentence)
            prompt += f'{idx + 1}: {temp_sentence}\n'
            prompt += '\n'
            # revise_result[idx] = query[idx]['data_idx'][0]
            revise_result[query[idx]['data_idx'][0]] = idx
        

        keys = list(self.args.llm_example.keys())
        total_type = '[O, '
        for i in keys:
            total_type += str(i)
            total_type += ', '
        total_type = total_type[:-2]
        total_type += ']'

        prompt += f'Please help me identify possible entities and there types in each sentence. For each test sentence, use index number and  brackets around what you think are the possible pairs of entities and types. If there are multiple test sentences, use \n to separate each bracket. For example, 1:[we:PER, car: VEH] \n 2:[ the secretary of homeland security:ORG, people:PER]. The number of output pairs must match the number of test sentences, no more, no less. Result word must in the test sentence, and the type must be among {total_type}. ONLY return the results list, do NOT return any other information.'

        logging.info(f'prompt为:  {prompt}')
        completion = self.args.llm.chat.completions.create(
                        model=f"{self.args.llm_model}",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
        result = completion.choices[0].message.content
        logging.info(f'LLM输出结果为: {str(result)}')
        result = result.split('\n')

        idx_total, pred_words, pred_types = [], [], []
        true_words, true_types = [], []
        
        for sub_result in result:
            if '[' not in sub_result:
                continue
            index = sub_result.index('[')
            if ':' in sub_result[:index]:
                first_colon = sub_result.index(':')
                idx = int(sub_result[:first_colon])
                idx_total.append(idx)
                sub_string = sub_result[first_colon+1:]
                matches = re.findall(r'\s*([\w\s]+)\s*:\s*([\w\s]+)\s*', sub_string)
                words = [match[0].strip() for match in matches]
                types = [match[1].strip() for match in matches]
                pred_words.append(words)
                pred_types.append(types)

        for i in range(len(query)):
            true = query[i]
            data_idx = true['data_idx'][0]
            if revise_result[data_idx] + 1 in idx_total:
                llm_idx = idx_total.index(revise_result[data_idx] + 1 )
                pred_word, pred_type = pred_words[llm_idx], pred_types[llm_idx]
            else:
                pred_word, pred_type = [], []

            token_idx = true['token_idx'][0]
            reverse_tokenidx = {i:idx for idx, i in enumerate(token_idx)}
            true_pos_span = true['pos_span'][0]
            origin_sentence = true['sentence'][0]
            true_type = true['entity_types'][0]
            true_type = true_type[true_type>0].tolist()
            tag2label = {value:key for key, value in true['label2tag'].items() }
            true_type = [tag2label[i] for i in true_type]
            words = []
            for i in range(len(true_pos_span)):
                temp = true_pos_span[i]
                temp = [reverse_tokenidx[temp[0]], reverse_tokenidx[temp[1]]]
                words.append(origin_sentence[temp[0]:temp[1]])
            
            words = [' '.join(i) for i in words]
            
            batch_result[data_idx] = {'idx':data_idx, 'pred_word':pred_word, 'true_word':words, 'pred_type':pred_type, 'true_type':true_type}
       
        return batch_result
    


