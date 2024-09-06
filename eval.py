import jsonlines
from utils.evaluate import Evaluate
import copy
import json

def eval(path):
    pred_evaluate = Evaluate()
    refine_evaluate = Evaluate()
    golden_evaluate = Evaluate()
    threshold_golden_evaluate = Evaluate()
    idx, pred_label, refine_label, llm_golden, threshold_golden, true = [], [], [], [], [], []
    with open(path, 'r+') as f:
        for items in jsonlines.Reader(f):
            pred_label.append(items['pred_label'])
            refine_label.append(items['refine_label'])
            llm_golden.append(items['llm_golden'])
            true.append(items['true'])
            idx.append(items['idx'])
            if 'refine_idx' in items.keys():
                threshold_golden.append(items['refine_idx'])
            else:
                threshold_golden.append([])
    
    for i in range(len(idx)):

        current_pred_label = pred_label[idx[i]]
        current_true_label = true[idx[i]]
        current_refine_label = refine_label[idx[i]]
        current_golden_label = llm_golden[idx[i]]
        current_threshold_golden_label = threshold_golden[idx[i]]
        current_threshold = copy.deepcopy(current_pred_label)

        pred_evaluate.collect_data(current_pred_label, current_true_label)
        if len(current_refine_label) != len(current_true_label):
            refine_evaluate.collect_data( current_pred_label, current_true_label)
        else:
            refine_evaluate.collect_data( current_refine_label, current_true_label)
        
        if len(current_golden_label) != len(current_true_label):
            golden_evaluate.collect_data( current_pred_label, current_true_label)
        else:
            golden_evaluate.collect_data( current_golden_label, current_true_label)

        if current_threshold_golden_label:
            for j in current_threshold_golden_label:
                current_threshold[j] = current_true_label[j]
            threshold_golden_evaluate.collect_data( current_threshold, current_true_label)
        else:
            threshold_golden_evaluate.collect_data( current_pred_label, current_true_label)
        


    (pred_micro, pred_macro), (_, _), (_, _) = pred_evaluate.metrics_by_entity()
    (refine_micro, refine_macro), (_, _), (_, _) = refine_evaluate.metrics_by_entity()
    (golden_micro, golden_macro), (_, _), (_, _) = golden_evaluate.metrics_by_entity()
    (threshold_golden_micro, threshold_golden_macro), (_, _), (_, _) = threshold_golden_evaluate.metrics_by_entity()

    
    return pred_micro, pred_macro, refine_micro, refine_macro, golden_micro, golden_macro, threshold_golden_micro, threshold_golden_macro



def compare_file(path1, path2):

    idx1, pred_label1, true1, golden1, refine_idx1 = [], [], [], [], []
    idx2, pred_label2, true2, golden2, refine_idx2 = [], [], [], [], []
    
    with open(path1, 'r+') as f:
        for items in jsonlines.Reader(f):
            golden1.append(items['llm_golden'])
            true1.append(items['true'])
            idx1.append(items['idx'])
            pred_label1.append(items['pred_label'])
            if 'refine_idx' in items.keys():
                refine_idx1.append(items['refine_idx'])
            else:
                refine_idx1.append([])



    with open(path2, 'r+') as f:
        for items in jsonlines.Reader(f):
            golden2.append(items['llm_golden'])
            true2.append(items['true'])
            idx2.append(items['idx'])
            pred_label2.append(items['pred_label'])
            if 'refine_idx' in items.keys():
                refine_idx2.append(items['refine_idx'])
            else:
                refine_idx2.append([])

    for i in idx1:
        if golden1[i] != golden2[i]:
            print(i)
        if pred_label1[i] != pred_label2[i]:
            print(i)
        if true1[i] != true2[i]:
            print(i)
    return

def compare_with_golden(path1, path2):
    idx1, pred_label1, true1,  refine_idx1 = [], [], [], []
    idx2, pred_label2, true2, refine_idx2 = [], [], [], []
    
    with open(path1, 'r+') as f:
        for items in jsonlines.Reader(f):
            true1.append(items['true'])
            idx1.append(items['idx'])
            pred_label1.append(items['pred_label'])
            if 'refine_idx' in items.keys():
                refine_idx1.append(items['refine_idx'])
            else:
                refine_idx1.append([])

    with open(path2, 'r+') as f:
        for items in jsonlines.Reader(f):
            idx2.append(items['idx'])
            pred_label2.append(items['pred_label'])
            true2.append(items['true'])
            refine_idx2.append(items['refine_idx'])

    for i in range(len(idx1)):
        idx, pred_label, true = idx1[i], pred_label1[idx1[i]], true1[idx1[i]]
        refine_idx = refine_idx1[idx1[i]]

        new_idx = idx2.index(idx1[i])
        pred_label_, true_ = pred_label2[new_idx], true2[new_idx]
        refine_idx_ = refine_idx2[new_idx]

        if pred_label != pred_label_:
            print(i)
        if true != true_:
            print(i)
        if refine_idx != refine_idx_:
            print(i)


    return



def revise_file(path1, path2, path3):
    idx1, pred_label1, refine_label1, llm_golden1, true1 = [], [], [], [], []
    with open(path1, 'r+') as f:
        for items in jsonlines.Reader(f):
            pred_label1.append(items['pred_label'])
            refine_label1.append(items['refine_label'])
            llm_golden1.append(items['llm_golden'])
            true1.append(items['true'])
            idx1.append(items['idx'])

    idx2, pred_label2, true2, wrong_idx2 = [], [], [], []
    with open(path2, 'r+') as f:
        for items in jsonlines.Reader(f):
            pred_label2.append(items['pred_label'])
            wrong_idx2.append(items['refine_idx'])
            true2.append(items['true'])
            idx2.append(items['idx'])

    for i in range(len(idx1)):
        idx, pred_label, refine_label = idx1[i], pred_label1[idx1[i]], refine_label1[idx1[i]]
        llm_golden, true = llm_golden1[i], true1[idx1[i]]

        new_idx = idx2.index(idx1[i])
        pred_label_, true_ = pred_label2[new_idx], true2[new_idx]
        wrong_idx = wrong_idx2[new_idx]

        if pred_label != pred_label_:
            print()


    
    
    with open(path3, 'a+') as f:
        for i in range(len(idx1)):
            idx, pred_label, refine_label = idx1[i], pred_label1[idx1[i]], refine_label1[idx1[i]]
            llm_golden, true = llm_golden1[i], true1[idx1[i]]

            new_idx = idx2.index(idx1[i])
            pred_label_, true = pred_label2[new_idx], true2[new_idx]
            wrong_idx = wrong_idx2[new_idx]

            write_data = {'idx': idx, 'pred_label': pred_label, 'refine_label':refine_label, \
                'llm_golden':llm_golden, 'true':true, 'refine_idx':wrong_idx}
            
            json_item = json.dumps(write_data)
            f.write(json_item + '\n')




    




if __name__ == '__main__':

    model_name = 'protobert'
    data = 'ace05'
    shots = '10'
    batch_size = '6'
    threshhold = '0.75'
    path = f'{model_name}_{data}_{shots}shots_{batch_size}batchsize_{threshhold}threshold.jsonl'

    pred_micro, pred_macro, refine_micro, refine_macro, golden_micro, golden_macro, \
        threshold_golden_micro, threshold_golden_macro = eval('result/'+path)

    print(f'当前实验设置为: {path}')
    print(f'仅小语言模型输出结果：micro: {pred_micro}, macro: {pred_macro}')
    print(f'大语言模型微调结果：micro: {refine_micro}, macro: {refine_macro}')
    print(f'大语言模型预测格式中若全对结果：micro: {golden_micro}, macro: {golden_macro}')
    print(f'大语言模型若全对结果：micro: {threshold_golden_micro}, macro: {threshold_golden_macro}')


    # path1 = 'result/protobert_ace05_10shots_1batchsize_0.9threshold.jsonl'
    # path2 = 'result/protobert_ace05_10shots_2batchsize_0.9threshold.jsonl'
    # compare_file(path1, path2)

    # path1 = 'result/protobert_ace05_10shots_2batchsize_0.9threshold.jsonl'
    # path2 = 'result/protobert_ace05_10shots.jsonl'
    # compare_with_golden(path1, path2)




    # path1 = 'result/protobert_ace05_10shots_1batchsize_0.9threshold.jsonl'
    # path2 = 'result/protobert_ace05_10shots.jsonl'
    # path3 = 'result/protobert_ace05_10shots_1batchsize_0.9threshold_new.jsonl'
    # revise_file(path1, path2, path3)



