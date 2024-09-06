
from utils.evaluate import Evaluate
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

class HttpFilter(logging.Filter):
    def filter(self, record):
        # 排除包含'HTTP'的日志记录
        return 'HTTP' not in record.getMessage()

class Trainer_batch_sentence_index_wexample(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)
        

    def test(self, finetune_data_loader, target_test_loader, model_path):

        print('================  start   test    ======================')
        self.model = self.load_model().to(self.args.device)
        if self.args.model_name != 'nerdp':
            self.model.load_state_dict(self.__load_ckpt__(model_path))
        
        logging.basicConfig(filename=f'log/{self.args.model_name}_{self.args.test_data}_{self.args.test_shots}shots_{self.args.llm_refine_batch_size}batchsize_{self.args.llm_refine_batch_threshold}threshold_{self.args.llm_revise_level}_{self.args.llm_model}.log', level=logging.INFO, \
            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
         # 添加自定义过滤器
        http_filter = HttpFilter()
        for handler in logger.handlers:
            handler.addFilter(http_filter)

        print(f'test task: {self.args.test_data}_{self.args.test_shots}shots')

        if self.args.do_fintune:
            print(f'fintuning{self.args.finetune_iter}.....')
            parameters_to_optimize = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            learning_rate = self.args.test_lr
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate)
            eval_iter = self.args.test_iter

            self.model.train()

            for i in range(self.args.finetune_iter):
                self.model.reset_spt()
                for support in finetune_data_loader:
                    support = self._switch_support(support)
                    # self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)
                    loss, _, _ = self.model(support, support)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()


        it = 0
        eval_iter =  len(target_test_loader)
        eval_iter =  min(5000, len(target_test_loader))
        # eval_iter =  1000
        evaluate = Evaluate()
        self.model.eval()

        with torch.no_grad():
            self.model.reset_spt()
            for support in finetune_data_loader:
                support = self._switch_support(support)
                self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)


            spt_total = self.model.get_spt_proto()
            revise_sentence = []
            before_revise = {}
            batch_pred, batch_true, batch_query, batch_wrongidx = [], [], [], []
            revise_batch_size = self.args.llm_refine_batch_size
            result_file_name = f'main_result/{self.args.model_name}_{self.args.test_data}_{self.args.test_shots}shots_{self.args.llm_refine_batch_size}batchsize_{self.args.llm_refine_batch_threshold}threshold.jsonl'
            num_change_label = 0

            already_refine_idx = []
            already_refine_pred = []
            already_refine_refine = []
            already_refine_true = []


            if self.args.do_llm_revise and os.path.exists(result_file_name):
                with open(result_file_name, 'r+') as f:
                    for items in jsonlines.Reader(f):
                        already_refine_idx.append(items['idx'])
                        already_refine_pred.append(items['pred_label'])
                        already_refine_true.append(items['true'])
                        already_refine_refine.append(items['refine_label'])

            # total_idx, total_pred, total_true, total_worng_idx = [], [], [], []

            for query in tqdm(target_test_loader, total=eval_iter, desc=f'testing {self.args.test_data}', ncols=80, ascii=True):
                if self.args.do_llm_revise and query['data_idx'][0] in already_refine_idx:
                    idx = already_refine_idx.index(query['data_idx'][0])
                    if already_refine_refine[idx]:
                        evaluate.collect_data( already_refine_refine[idx], already_refine_true[idx], query)
                    else:
                        evaluate.collect_data( already_refine_pred[idx], already_refine_true[idx], query)
                else:
                    query = self._switch_support(query)
                    probability, pred = self.model.test(query, spt_total)
                    true_label = torch.cat(query['entity_types'], 0)
                    index = torch.where(probability<self.args.llm_refine_batch_threshold)[0]

                    # total_pred.append(pred.tolist())
                    # total_true.append(true_label.tolist())
                    # total_worng_idx.append(index.tolist())
                    # total_idx.append(query['data_idx'][0])

                    if self.args.do_llm_revise and index.tolist() != []:
                        origin_sentence = query['sentence'][0]
                        revise_sentence.append(origin_sentence)
                        unusual_pred_label = pred[index].tolist()
                        corresponding_true = true_label[index].tolist()
                        entity_masks, token_idx = query['entity_masks'][0].tolist(), query['token_idx'][0]
                        reverse_tokenidx = {i:idx for idx, i in enumerate(token_idx)}
                        origin_sentence = query['sentence'][0]
                        test_sentence = ' '.join(origin_sentence)
                        tag2label = {value:key for key, value in query['label2tag'].items() }
                        for i in range(len(corresponding_true)):
                            corresponding_true[i] = tag2label[corresponding_true[i]]
                        
                        for i in range(len(index.tolist())):
                            unusual_pred = entity_masks[index[i]]
                            unusual_pred = [reverse_tokenidx[unusual_pred[0]], reverse_tokenidx[unusual_pred[1]]]
                            unusual_pred_word = [origin_sentence[w] for w in range(unusual_pred[0], unusual_pred[1])]
                            unusual_pred_word = ' '.join(unusual_pred_word)
                            unusual_pred_word_label = tag2label[unusual_pred_label[i]]
                            current_sentence_num = len(revise_sentence)-1
                            if current_sentence_num not in before_revise.keys():
                                before_revise[current_sentence_num] = [[unusual_pred_word, unusual_pred_word_label, index.tolist()[i]]]
                            else:
                                before_revise[current_sentence_num].append([unusual_pred_word, unusual_pred_word_label, index.tolist()[i]])
                        
                        batch_pred.append(pred)
                        batch_true.append(true_label)
                        batch_query.append(query)
                        batch_wrongidx.append(index.tolist())
                    else:
                        evaluate.collect_data(pred, true_label, query)
                        if self.args.do_llm_revise:
                            write_data = {'idx': query['data_idx'][0], 'pred_label': pred.tolist(), 'refine_label':[], 'llm_golden':[], 'true':true_label.tolist()}
                            evaluate.save_jsonline(result_file_name, write_data)

                if self.args.do_llm_revise and revise_sentence!=[] and len(revise_sentence) % revise_batch_size == 0:
                    batch_refine, batch_golden_true, num_changed = self.batch_llm_revise_wo_sentence(revise_sentence, before_revise, \
                                                                    batch_pred, batch_true, query['label2tag'], logger)
                    for p, r, g, t, q, widx in zip(batch_pred, batch_refine, batch_golden_true, batch_true, batch_query, batch_wrongidx):
                        evaluate.collect_data( r, t, q)
                        write_data = {'idx': q['data_idx'][0], 'pred_label': p.tolist(), 'refine_label':r.tolist(), 'llm_golden':g.tolist(), 'true':t.tolist(), 'refine_idx':widx}
                        evaluate.save_jsonline(result_file_name, write_data)
                    num_change_label += num_changed
                    revise_sentence = []
                    before_revise = {}
                    batch_pred, batch_true, batch_query, batch_wrongidx = [], [], [], []

                if it >= eval_iter:
                    break
                it += 1


            if revise_sentence:
                batch_refine, batch_golden_true, num_changed = self.batch_llm_revise_wo_sentence(revise_sentence, before_revise, \
                                                                    batch_pred, batch_true, query['label2tag'], logger)
                for p, r, g, t, q, widx in zip(batch_pred, batch_refine, batch_golden_true, batch_true, batch_query, batch_wrongidx):
                    evaluate.collect_data( r, t, q)
                    write_data = {'idx': q['data_idx'][0], 'pred_label': p.tolist(), 'refine_label':r.tolist(), 'llm_golden':g.tolist(), 'true':t.tolist(), 'refine_idx':widx}
                    evaluate.save_jsonline(result_file_name, write_data)
            
            # with open(f'result/{self.args.model_name}_{self.args.test_data}_{self.args.test_shots}shots.jsonl', 'a+') as f:
            #     for idx, pred, true, worng_idx in zip(total_idx, total_pred, total_true, total_worng_idx):
            #         write_data = {'idx': idx, 'pred_label': pred,  'true':true, 'refine_idx':worng_idx}
            #         json_item = json.dumps(write_data)
            #         f.write(json_item + '\n')


            (micro, macro), (nested_micro, nested_macro), (flat_micro, flat_macro) = evaluate.metrics_by_entity( query['label2tag'])
            print(f'{self.args.test_data}_{self.args.test_shots}shot')
            print(round(micro*100,2), round(macro*100,2), round(nested_micro*100,2), round(nested_macro*100,2), round(flat_micro*100,2), round(flat_macro*100,2))
            # print(num_change_label)
        

        return micro, macro, nested_micro, nested_macro, flat_micro, flat_macro

    def batch_llm_revise_wo_sentence(self, revise_sentence, before_revise, pred_label, true_label, label2tag, logging):
        num_changed = 0

        idx = 1
        revise_result = {}
        prompt = get_llm_revise_prompt_we(self.args.llm_example, self.args.test_data)
        prompt += f'Now, \n'
        for key, value in before_revise.items():
            temp_sentence = ' '.join(revise_sentence[key])
            prompt += f'For the sentence {temp_sentence} \n'
            for i in value:
                assert(i[0] in temp_sentence)
                prompt += f'{idx}. I guess words span [ {i[0]} ] is a {i[1]} entity,\n'
                revise_result[idx] = [key, i[-1]]

                idx +=1
            prompt += '\n'
        

        keys = list(self.args.llm_example.keys())
        total_type = '[O, '
        for i in keys:
            total_type += str(i)
            total_type += ', '
        total_type = total_type[:-2]
        total_type += ']'

        prompt += f'But I am not very sure. Please help me, If you think my guess is correct, please return my guess type. If you think my guess is wrong, return the entity type you believe it to be, or if you think the guessed words are not entities, please return O. Please output the list of results and separate by commas if there are multiple guesses, for example, 1:PER, 2:ORG. The number of output pairs must match the number of my guesses, no more, no less. Results must be be among {total_type}. ONLY return the results list, do NOT return any other information.'

        logging.info(f'prompt为:  {prompt}')
        completion = self.args.llm.chat.completions.create(
                        model=f"{self.args.llm_model}",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
        result = completion.choices[0].message.content
        result = result.split(',')
        logging.info(f'LLM输出结果为: {str(result)}')

        total_need_revice = len(revise_result.keys())


        refine_label = copy.deepcopy(pred_label)
        llm_golde_true = copy.deepcopy(pred_label)


        target_type = list(label2tag.keys())

        if len(result) == total_need_revice:
            logging.info(f'开始替换')
            temp = []
            for i in result:
                if ':' in i:
                    colon_idx = i.index(':')
                    entity_type = i[colon_idx+1:].replace('\'','')
                    
                else:
                    entity_type = i.replace('\'','')

                entity_type = entity_type.replace(' ','')
                entity_type = entity_type.replace('.','')
                entity_type = entity_type.replace('\n','')
                entity_type = entity_type.replace('.','')
                entity_type = entity_type.replace('\'','')
                entity_type = entity_type.replace(':','')
                temp.append(entity_type)

            result = temp

            befor_revise, after_revise, corr_true = [], [], []
            llm_results = []

            for key, value in revise_result.items():
                befor_revise.append(result[key-1])
                if result[key-1] in label2tag.keys() and pred_label[value[0]][value[1]] != label2tag[result[key-1]]:
                    num_changed += 1
                    refine_label[value[0]][value[1]] = label2tag[result[key-1]]
                corr_true.append(true_label[value[0]][value[1]].item())
                llm_golde_true[value[0]][value[1]] = true_label[value[0]][value[1]]

            logging.info(f'模型输出结果为: {str(befor_revise)}')
            logging.info(f'对应的真实标签为: {str(corr_true)}')

        return refine_label, llm_golde_true, num_changed

        # return 0, 0, 0, len(prompt)