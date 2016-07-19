# Kaggle Ultrasound Nerve Segmentation

Content:
```
.
├── data
│   ├── processed -- temp and final results will be here
│   └── raw -- place kaggle data here
├── docker
│   └── Dockerfile -- python/Theano/keras with CUDA/CUDNN
├── ker.sh -- script for running nvidia-cuda
├── Makefile -- control panel for the whole project
├── models -- folder for storing weights
├── README.md
└── src
    ├── data.py -- preprocessing and data utils
    ├── model.py -- u-net model on keras
    ├── submission.py
    ├── test_gpu.py -- Theano use GPU or not
    └── train.py
```
Usage:

1. Download data into `data/raw` and unzip
2. Prepare data: `make prepare`. Now you have processed data into `data/processed`
3. Prepare docker image: `make docker`
4. Make `ker.sh` executable and run train procedure:
    ```
    chmod +x ker.sh
    ./ker.sh python /src/train.py
    ```
    Be aware about path: local dirs are mounted into root in docker container: `src/->/src`, `data/->/data`, `models/->/models`
5. Build submission `make submission`



Current status:

1. Docker ready for python2, not for python3
2. Model runnable
3. Train runnable

TODO:

1. Add opencv3 into Dockerfile
2. Make a submission
3. Enhance the model

Solution code based on https://github.com/jocicmarko/ultrasound-nerve-segmentation solution.

Main idea: rich augmentation technics and model tuning: BN and so on.
