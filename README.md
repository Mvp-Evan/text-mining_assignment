# Package Requirements
Python 3.8
Before runing the code, you need to install the following packages:
```terminal
pip install -r requirements.txt
```

# Data and model preparation
Please download the data from Google Drive and put it in the `./data` folder. The data can be downloaded from [here](https://drive.google.com/drive/folders/1AC6sN87-FcfLcr_9SWqVO4yX49wPXKKl?usp=sharing)

Please download the model weights from Google Drive and put it in the `./checkpoint` folder. The model weights can be downloaded from [here](https://drive.google.com/drive/folders/1kZN6OfedEjPhP8DG_yclB5_BM02LNKmv?usp=share_link)

# Run the code
This project data provide two updated methods of LSTM and BERT. You can follow the instructions below to train, test and predict via run `./main.py`.
There are four arguments you can use for main.py:
- `--model`: the model you want to use, you can choose from `LSTM_UP`, `LSTM` and `BERT`
- `--need_generate_data`: whether you want to generate the data, you can choose from `True` and `False`
- `--run_type`: the mode you want to use, you can choose from `train`, `test` and `predict`
- `--device`: the device you want to use, you can choose from `cpu`, `cuda` and `mps`  

For `--model`, you can choose `LSTM_UP` to use **the updated version** of LSTM model, `LSTM` to use **the original version** of LSTM model, and `BERT` to use **the updated version** of BERT model.

Typically, the data we provided in Googl Drive **has been preprocessed**, so you can set `--need_generate_data` to `False` or ignore this argument.

For example, if you want to train the LSTM model on GPU, you can run the following command:
```terminal
python main.py --model LSTM --run_type train --device cuda
```

If you want to train the BERT model on MPS and generate the data in advance, you can run the following command:
```terminal
python main.py --model BERT --need_generate_data True --run_type train --device mps
```

If you want to test the BERT model on CPU, you can run the following command:
```terminal
python main.py --model BERT --run_type test --device cpu
```

If you want to predict the LSTM model on MPS, you can run the following command:
```terminal
python main.py --model LSTM --run_type predict --device mps
```
When you run predict mode, you will be required to input the sentence you want to predict. After you input the sentence, the model will print the prediction result. In prediction result, `r` is the Relation Extraction result.

# Possible issues
- If you meet issues with `***.npy file not found` or `***.json file not found`, please run `main.py` with parameter `--need_generate_data True` to generate the data in advance.
- Because of test set is big, it may take a long time and more memory to test the model.
