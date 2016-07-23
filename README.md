# Kaggle Ultrasound Nerve Segmentation

Based on https://github.com/jocicmarko/ultrasound-nerve-segmentation solution.

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
2. Under active development. Be careful with kludges

TODO:

1. Reread and refactor
2. Investigate errors
3. Modify network
4. Modify preprocessing

Infrastructure TODO:

1. Add opencv for python3 into Dockerfile (for future usage)
1. Move all processes into docker



Main idea: rich augmentation technics and model tuning: BN and so on.
