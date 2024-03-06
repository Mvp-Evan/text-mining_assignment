from configs import generate_data
from evaluation import evaluate
from train import train


def main():
    # if you want to generate data, run this line
    # generate_data()

    # model_name = 'BERT' or 'LSTM'
    # device = 'mps', 'cuda' or 'cpu'
    train('BERT', 'mps')

    # if you want to evaluate the model, run this line
    evaluate()


if __name__ == '__main__':
    main()
