# Keras implementation of SRCNN

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

## ESPCN
CVPR2016: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
![espcn](/uploads/0f6cf4b968ed5c2adce7efba861b4678/espcn.png)

## Structure of directory
### sub directory
```
geosr-keras
├── data
│   ├── test
│   │   └──
│   └── train
│       └──
├── logs
│   ├── model_zoo
│   │   └──
│   └── statistic
│       └──
├── main.py
├── prepare_data.py
├── README.md
└── result
    └──
```

My implementation have some difference with the original paper, include:

* use Adam alghorithm for optimization, with learning rate 0.0003 for all layers.
* Use the opencv library to produce the training data and test data, not the matlab library. This difference may caused some deteriorate on the final results.
* I did not set different learning rate in different layer, but I found this network still work.
* The color space of YCrCb in Matlab and OpenCV also have some difference. So if you want to compare your results with some academic paper, you may want to use the code written with matlab.

## Usage:
### Create your own data directory, and prepare training and test dataset.
`cd ./data`  
`mkdir train test`

### Example:
`python main.py -t True -i 100 -u 2`

### Help
```
usage: main.py [-h] [--train_data_path TRAIN_DATA_PATH]
               [--test_data_path TEST_DATA_PATH] [--model_path MODEL_PATH]
               [--model_name_train MODEL_NAME_TRAIN]
               [--model_name_predict MODEL_NAME_PREDICT]
               [--result_path RESULT_PATH]
               [--result_stats_path RESULT_STATS_PATH] [-t TRAIN_MODE]
               [-i NEPOCHS] [-u UPSCALE_FACTOR]

Keras Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --train_data_path TRAIN_DATA_PATH
                        training data path
  --test_data_path TEST_DATA_PATH
                        testing data path
  --model_path MODEL_PATH
                        model path
  --model_name_train MODEL_NAME_TRAIN
                        trained model name
  --model_name_predict MODEL_NAME_PREDICT
                        model used to predict
  --result_path RESULT_PATH
                        model path
  --result_stats_path RESULT_STATS_PATH
                        trained model name
  -t TRAIN_MODE, --train_mode TRAIN_MODE
                        train the model or not
  -i NEPOCHS, --nEpochs NEPOCHS
                        number of epochs to train for
  -u UPSCALE_FACTOR, --upscale_factor UPSCALE_FACTOR
                        super resolution upscale factor
```

## Result(training for 200 epoches on 91 images, with upscaling factor 2):
Results on Set5 dataset:
sdaad
