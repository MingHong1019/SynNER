import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import json

class Evaluate():
    def __init__(self, ):
        self.pred_label = []
        self.true_label = []
        
        self.nested_pred_label = []
        self.nested_true_label = []
        self.flat_pred_label = []
        self.flat_true_label = []
        
        self.total_classify_num = 0
        self.correct_O_num = 0
        self.correct_entity_num = 0

        self.O_classify2_label = 0
        self.labe_lclassify2_O = 0
        self.labe_lclassify2_otherlabel = 0

    def collect_data(self, pred, label, query=None):
        if type(pred) == list:
            pred = torch.tensor(pred)
            label = torch.tensor(label)
        
        self.pred_label.extend(pred.view(-1).cpu().tolist())
        self.true_label.extend(label.view(-1).cpu().tolist())
        
        
        if query!=None:
        
            total_spans = query['pos_span'][0]
            flat_idx, nested_idx = [], []
            for i in range(len(total_spans)):
                current_span = total_spans[i]
                nested = False
                for j in range(len(total_spans)):
                    temp_span = total_spans[j]
                    if current_span == temp_span:
                        continue
                    if self.weather_nested(current_span, temp_span):
                        nested = True
                
                if nested:
                    nested_idx.append(i)
                else:
                    flat_idx.append(i)
                    
                    
            if nested_idx!=[]:        
                nested_result = [i for i in range(len(label)) if i not in flat_idx]
            
                nested_result = torch.tensor(nested_result)
                pred_ = pred[nested_result]
                label_ = label[nested_result]
                
                self.nested_pred_label.extend(pred_.view(-1).cpu().tolist())
                self.nested_true_label.extend(label_.view(-1).cpu().tolist())

            
            if flat_idx!=[]:
                flat_result = [i for i in range(len(label)) if i not in nested_idx]
            
                flat_result = torch.tensor(flat_result)
                pred_ = pred[flat_result]
                label_ = label[flat_result]
                self.flat_pred_label.extend(pred_.view(-1).cpu().tolist())
                self.flat_true_label.extend(label_.view(-1).cpu().tolist())
        

    def metrics_by_entity(self, label2tag=None, clean=False):
        nested_micro, nested_macro, flat_micro, flat_macro = 0, 0, 0, 0
        if self.nested_pred_label:
            nested_micro, nested_macro = self.get_f1(self.nested_pred_label,  self.nested_true_label, label2tag)
            flat_micro, flat_macro = self.get_f1(self.flat_pred_label,  self.flat_true_label, label2tag)
        micro, macro = self.get_f1(self.pred_label,  self.true_label, label2tag)
        if clean:
            self.pred_label = []
            self.true_label = []
            
            self.nested_pred_label = []
            self.nested_true_label = []
            self.flat_pred_label = []
            self.flat_true_label = []
        

        return (micro, macro), (nested_micro, nested_macro), (flat_micro, flat_macro)

    def analysis(self,):
        pred_label = self.pred_label
        true_label = self.true_label
        self.total_classify_num = len(pred_label)


        for i in range(len(pred_label)):
            if pred_label[i] != true_label[i]:
                if true_label[i] == 0:
                    self.O_classify2_label += 1

                else:
                    if pred_label[i] == 0:
                        self.labe_lclassify2_O += 1

                    else:
                        self.labe_lclassify2_otherlabel += 1

            else:
                if true_label[i] == 0:
                    self.correct_O_num +=1

                else :
                    self.correct_entity_num +=1
    
    def show_result(self):
        print(f'total predicted {self.total_classify_num} spans')
        print(f'    predict true, but the label is O: {self.correct_O_num} ')
        total_num = self.correct_entity_num + self.O_classify2_label + \
                    self.labe_lclassify2_O + self.labe_lclassify2_otherlabel
        print(f'    predict true, and the label is entity: {self.correct_entity_num} , '
              f'rate: {round(100*self.correct_entity_num/total_num, 3)}')
        print(f'    predict wrong, the true is O but predict to entity: {self.O_classify2_label}, '
              f'rate: {round(100*self.O_classify2_label/total_num, 3)} ')
        print(f'    predict wrong, the true is entity but predict to O: {self.labe_lclassify2_O} ,'
              f' rate: {round(100*self.labe_lclassify2_O/total_num, 3)}')
        print(f'    predict wrong, the true is entity but predict to other entity entity:'
              f' {self.labe_lclassify2_otherlabel} , rate: {round(100*self.labe_lclassify2_otherlabel/total_num, 3)}')

    
    def reset_evaluate(self):
        self.pred_label = []
        self.true_label = []
        
    def weather_nested(self, current_span, temp_span):
        result= False
        if current_span[1]>temp_span[0] and current_span[1]<=temp_span[1]:
            if current_span[0]<=temp_span[0]:
                result = True
        
        if current_span[0]>=temp_span[0] and current_span[0]<temp_span[1]:
            if current_span[1]>=temp_span[1]:
                result = True
        
        if current_span[0]>=temp_span[0] and current_span[1]<=temp_span[1]:
            result = True   
        
        if current_span[0]<=temp_span[0] and current_span[1]>=temp_span[1]:
            result = True    
                
        return result

    def get_f1(self, pred, label, label2tag=None):

        total_label = list(set(label))
        total_label.remove(0)
        if label2tag:
            target_names = list(label2tag.keys())
            target_names.remove('O')

        # print(classification_report(label, pred, labels=total_label, target_names=target_names, digits=4))
        micro = prfs(label, pred, labels=total_label, average='micro')[2]
        macro = prfs(label, pred, labels=total_label, average='macro')[2]
        return micro, macro


    def save_jsonline(self, path, data):
        with open(path, 'a+') as f:
            json_item = json.dumps(data)
            f.write(json_item + '\n')
        return
    
class Evaluate_onlyLLM():

    def __init__(self, ):
        self.correct_cnt = 0
        self.label_cnt = 0
        self.pred_cnt = 0

    def save_jsonline(self, path, data):
        with open(path, 'a+') as f:
            json_item = json.dumps(data)
            f.write(json_item + '\n')
        return

    def collect_data(self, pred_word, true_word, pred_type, true_type):

        pred = {}
        true = {}

        if len(pred_word) == len(pred_type):
            for pw, pt in zip(pred_word, pred_type):
                if pt not in pred.keys():
                    pred[pt] = [pw]
                else:
                    pred[pt].append(pw)
        
        if len(true_word) == len(true_type):
            for tw, tt in zip(true_word, true_type):
                if tt not in true.keys():
                    true[tt] = [tw]
                else:
                    true[tt].append(tw)


        
        for label in pred.keys():
            self.pred_cnt += len(pred[label])
        
        for label in true.keys():
            self.label_cnt += len(true[label])

        for label in true.keys():
            self.correct_cnt += len(list(set(true[label]).intersection(set(pred.get(label,[])))))


        return 

    def get_f1(self):

        precision = self.correct_cnt / (self.pred_cnt + 1e-8)
        recall = self.correct_cnt / (self.label_cnt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1




