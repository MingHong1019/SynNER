

def get_llm_example(sentences, span, label, total_type, shots):
    result = {}

    for current_type in total_type:
        temp_result = []
        for i in range(len(label)):
            idx = [m for m, n in enumerate(label[i]) if n == current_type]
            if idx != []:
                for j in idx:
                    span_ = span[i][j]
                    words = sentences[i][span_[0]:span_[1]]
                    words = ' '.join(words)
                    if words not in temp_result:
                        temp_result.append(words) 
            
        max_len = min(shots, len(temp_result))
        result[current_type] = temp_result[:max_len]




    return result