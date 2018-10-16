# Keras implementation of Super Resolution

## Model: SRCNN
The original paper is [ECCV2014: Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)  
![srcnn__1_](/uploads/14bfd601cecd49cd317eaea81f0c7974/srcnn__1_.png)

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
* prepare_data.py is an utils python script
* model_zoo contains trained models
* statistic contains statistic result such as PSNR for each test image
* result contains obtained lr images, related sr images, and result comparison image

## Prequirements
* Python                             3.5.2
* Keras                              2.1.6
* tensorflow-gpu                     1.8.0

## Usage:
### Step 1: Data Preparation
Create data directory, and prepare your own training and test dataset.  
`cd ./data`  
`mkdir train test`

### Step 2: Conduct
#### Train and Test
`python main.py --model_name_train model_name -t True -i 10 -u 2`  
* Trained a model named model_name, and use it to test

#### Test only
`python main.py --model_name_predict model_name -t False -i 10 -u 2`  
* Use trained model named model_name to test

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

## Result(training for 100 epoches on 91 images, with upscaling factor 2):
### Peak signal-to-noise ratio (PSNR) comparison:
|  Image | Low Resolution | High Resolution |
|:------:|:--------------:|:---------------:|
| comic  | 23.7955677227  | 26.0434704104   |
| face   | 32.7865527256  | 33.9367435428   |
| baboon | 23.1091089408  | 23.9472346226   |

### Visualization:
comic![comic](/uploads/77e4cfcd8735acb23b044b9f4dcd7fdc/comic.png)  
face![face](/uploads/dd7521d0d0e94cced8535a1325c81a22/face.png)  
baboon![baboon](/uploads/933f4c906a1d20f9fec3d2fb50dcd8c0/baboon.png)

License
----

MIT

