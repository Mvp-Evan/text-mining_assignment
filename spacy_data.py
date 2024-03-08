import spacy
import json

def tokenize_data_spacy(text, title, output_file):
    # 加载spaCy的英文模型
    nlp = spacy.load("en_core_web_sm")
    
    # 处理文本
    doc = nlp(text)
    
    # 初始化数据结构
    sentences = []
    vertexSet = []

    for sent in doc.sents:
        sentence = [token.text for token in sent if not token.is_stop]  # 分词且去除停用词
        sentences.append(sentence)
        
        vertexs = []
        for ent in sent.ents:  # 遍历句子中的命名实体
            vertex = {
                'name': ent.text,
                'pos': [ent.start_char, ent.end_char],  # 使用字符级别的位置
                'sent_id': len(sentences) - 1,  # 当前句子的索引
                'type': ent.label_  # 实体类型
            }
            vertexs.append(vertex)
        if vertexs:
            vertexSet.append(vertexs)
    
    # 构建数据结构
    data = {
        'title': title,
        'sents': sentences,
        'vertexSet': vertexSet
    }
    
    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 示例文本和调用函数
text = "odel Performance. Table 4 shows the experimental results under the supervised and weakly supervised settings, from which we have the following observations: (1) Models trained with human annotated data generally outperform their counter parts trained on distantly supervised data."
title = "Example Document"
output_file = "spacy_formatted_data.json"
tokenize_data_spacy(text, title, output_file)
print(f"Data has been formatted and saved to '{output_file}'.")