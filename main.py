from configs import generate_data
from evaluation import evaluate
from train import train


def main():
    generate_data()
    # model_name = 'BERT' or 'LSTM'
    # device = 'mps', 'cuda' or 'cpu'
    train('BERT', 'mps')

    evaluate()


if __name__ == '__main__':
    main()
