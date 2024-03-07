import argparse
import json

from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

from configs import generate_data
from configs import generate_test_predict
from test.test import test
from test.test_predict import predict
from train import train


def tokenize_data(text, output_file):
    words = word_tokenize(text)

    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 词干提取
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    with open(output_file, 'w') as f:
        json.dump(words, f)


def data_preprocess(text, device='mps'):
    # tokenize data
    tokenize_data(text, '.data/test_predict.json')

    # generate data
    generate_test_predict(device='mps')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='predict')
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--need_generate_data', type=bool, default=False)
    parser.add_argument('--device', type=str, default='mps')

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
        text = input('Please input the text you want to predict: ')
        data_preprocess(text, args.device)
        predict(args.model_name, args.device)


if __name__ == '__main__':
    main()
