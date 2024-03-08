import argparse
import json
import random
import spacy
import sys

from nltk import word_tokenize, PorterStemmer, download, pos_tag
from nltk.corpus import stopwords

from configs import generate_data
from configs import generate_test_predict
from test import test, predict
from train import train

nlp = spacy.load("en_core_web_sm")

# # def tag_to_type(word, sentences):
#     word_types = ["ORG", "LOC", "TIME", "PER", "MISC", "NUM"]
#     if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
#         return random.choice(word_types)

#     else:
#         return 'BLANK'
def tag_to_type(word,sentences,tag):
    # Process sentences with spaCy to identify entities
    checked = False
    docs = nlp(sentences)

    for sent_id, sent in enumerate(docs.sents):
        for ent in sent.ents:
            if word == ent and ent.label_ in ["ORG", "LOC", "TIME", "PER", "MISC", "NUM"]:
                checked = True
                return ent.label_
            
    if checked == False:
        # Define a simplistic and hypothetical mapping from POS tag to a NER type
        word_types = ["ORG", "LOC", "TIME", "PER", "MISC", "NUM"]
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return random.choice(word_types)

        else:
            return 'BLANK'
            

def tokenize_data(text, title, output_file):
    download('punkt')
    download('stopwords')
    download('averaged_perceptron_tagger')  # download the tagger

    # Load ner2id.json
    with open('./data/ner2id.json', 'r') as f:
        ner2id = json.load(f)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # separate sentences
    sentences = []
    sentence = []
    for word in words:
        if word == '.'or word =='?'or word =='!':
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    # new added 
    if len(sentences)==0:
        print("it is not a proper sentence!!!")
        sys.exit(-1)
    # separate vertexSet, and add pos[start, end], sent_id, type to each vertex
    vertexSet = []
    for i, sentence in enumerate(sentences):
        # tag the sentence
        tagged_sentence = pos_tag(sentence)
        vertexs = []
        for j, (word, tag) in enumerate(tagged_sentence):
            # find pos end
            if j == 0:
                pos = [0, len(word)]
            else:
                pos = [vertexs[-1]['pos'][1], vertexs[-1]['pos'][1] + len(word)]

            type_ = tag_to_type(word,text,tag)
            vertex = {
                'name': word,
                'pos': pos,
                'sent_id': i,
                'type': type_
            }
            vertexs.append(vertex)
        vertexSet.append(vertexs)

    data = {
        'sents': sentences,
        'title': title,
        'vertexSet': vertexSet
    }

    with open(output_file, 'w') as f:
        json.dump([data], f)


def data_preprocess(text, title, output_file='./data/test_predict.json', device='mps'):
    # tokenize data
    tokenize_data(text, title, output_file)

    # generate data
    generate_test_predict(device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='predict')
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--need_generate_data', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    # if you want to generate data, run this line
    if args.need_generate_data:
        generate_data(device=args.device)

    # model_name = 'BERT' or 'LSTM'
    # device = 'mps', 'cuda' or 'cpu'
    if args.run_type == 'train':
        train(args.model_name, args.device)
    elif args.run_type == 'test':
        # if you want to predict, run this line
        test(args.model_name, args.device)
    elif args.run_type == 'predict':
        output_file = './data/test_predict.json'
        title = input('Please input the title of the text you want to predict: ')
        text = input('Please input the text you want to predict: ')
        # title = "sample title"
        # text = "odel Performance. Table 4 shows the experimental results under the supervised and weakly supervised settings, from which we have the following observations: (1) Models trained with human annotated data generally outperform their counter parts trained on distantly supervised data."
        data_preprocess(text, title, output_file, args.device)
        predict(args.model_name, args.device)


if __name__ == '__main__':
    main()
