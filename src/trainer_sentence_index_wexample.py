from utils.evaluate import Evaluate
from src.prompt_generate import get_llm_revise_prompt_we
from src.trainer import Trainer

from torch.optim import AdamW
from tqdm import tqdm

import torch




import logging

class HttpFilter(logging.Filter):
    def filter(self, record):
        # 排除包含'HTTP'的日志记录
        return 'HTTP' not in record.getMessage()

class Trainer_sentence_index_wexample(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args)
        

    def test(self, finetune_data_loader, target_test_loader, model_path):

        print('================  start   test    ======================')
        self.model = self.load_model().to(self.args.device)
        if self.args.model_name != 'nerdp':
            self.model.load_state_dict(self.__load_ckpt__(model_path))

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
        # eval_iter =  5000
        evaluate = Evaluate()
        self.model.eval()
        with torch.no_grad():
            self.model.reset_spt()
            for support in finetune_data_loader:
                support = self._switch_support(support)
                self.model.get_total_spt(support, len(finetune_data_loader), no_O=False)


            spt_total = self.model.get_spt_proto()

            for query in tqdm(target_test_loader, total=eval_iter, desc=f'testing {self.args.test_data}', ncols=80, ascii=True):
                query = self._switch_support(query)
                probability, pred = self.model.test(query, spt_total)
                true_label = torch.cat(query['entity_types'], 0)
                index = torch.where(probability<0.9)[0]

                if self.args.do_llm_revise and index.tolist() != []:
                    unusual_pred_label = pred[index].tolist()
                    corresponding_true = true_label[index].tolist()
                    entity_masks, token_idx = query['entity_masks'][0].tolist(), query['token_idx'][0]
                    reverse_tokenidx = {i:idx for idx, i in enumerate(token_idx)}
                    origin_sentence = query['sentence'][0]
                    test_sentence = ' '.join(origin_sentence)
                    tag2label = {value:key for key, value in query['label2tag'].items() }
                    for i in range(len(corresponding_true)):
                        corresponding_true[i] = tag2label[corresponding_true[i]]

                    prompt = get_llm_revise_prompt_we(self.args.llm_example)
                    prompt += f'Now for the test sentence [{test_sentence}], I guess: \n'

                    total_replace_span_after = []
                    for i in range(index.size()[0]):
                        unusual_pred = entity_masks[index[i]]
                        unusual_pred = [reverse_tokenidx[unusual_pred[0]], reverse_tokenidx[unusual_pred[1]]]
                        unusual_pred_word = [origin_sentence[w] for w in range(unusual_pred[0], unusual_pred[1])]
                        unusual_pred_word = ' '.join(unusual_pred_word)
                        unusual_pred_word_label = tag2label[unusual_pred_label[i]]
                        prompt += f'{i+1}. the words [ {unusual_pred_word} ] is a {unusual_pred_word_label} entity \n'

                    
                    prompt += f'But I am not very sure. Please help me, If you think my guess is correct, please return my guess, separated by commas if there are multiple guesses. If you think my guess is wrong, return the entity category you believe it to be, separated by commas if there are multiple guesses, or if you think the guessed words are not entities, please return O. Please output the result in dictionary format in python. For example, {{1:PER, 2:ORG}}. Note that the key in the output dictionary is the the indices I guessed above, and the corresponding value is the result you think should be. The number of key-value pairs must match the number of my guesses, no more, no less. The values must be within the range. Values must be be among [O, GPE, ORG, PER, LOC, WEA, VEH, FAC]. ONLY return the dictionary, do NOT return any other information.'
                    
                    completion = self.args.llm.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                    result = completion.choices[0].message.content
                    result = result.split('\n')
                    for i in result:
                        if '{' in i and '}' in i and ':' in i:
                            break
                    result = i.replace('{', '')
                    result = result.replace('}', '')
                    result = result.split(',')
                    if len(result) == index.size()[0]:
                        temp = []
                        for i in result:
                            colon_idx = i.index(':')
                            entity_type = i[colon_idx+1:].replace('\'','')
                            entity_type = entity_type.replace(' ','')
                            temp.append(entity_type)
                        result = temp
                        logging.info(f'第 {it} 个例子, prompt为: {prompt}')
                        logging.info(f'第 {it} 个例子, LLM输出结果为: {str(result)}')

                        index = index.tolist()
                        for i in range(len(index)):
                            temp = result[i]
                            if temp in query['label2tag'].keys():
                                if pred[index[i]] != query['label2tag'][temp]:
                                    num_change_label += 1
                                    pred[index[i]] = query['label2tag'][temp]
                                total_replace_span_after.append(temp)

                        logging.info(f'第 {it} 个例子, LLM输出替换结果为: {str(total_replace_span_after)}')
                        logging.info(f'第 {it} 个例子, 对应的真实标签为: {str(corresponding_true)}')



                evaluate.collect_data(pred, true_label, query)

                if it >= eval_iter:
                    break
                it += 1

            (micro, macro), (nested_micro, nested_macro), (flat_micro, flat_macro) = evaluate.metrics_by_entity( query['label2tag'])
            print(f'{self.args.test_data}_{self.args.test_shots}shot')
            print(round(micro*100,2), round(macro*100,2), round(nested_micro*100,2), round(nested_macro*100,2), round(flat_micro*100,2), round(flat_macro*100,2))
        

        return micro, macro, nested_micro, nested_macro, flat_micro, flat_macro
    


