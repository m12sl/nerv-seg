# Kaggle Ultrasound Nerve Segmentation

Usage:

1. Download data into `data/raw` and unzip
2. Prepare data: `make prepare`. Now you have processed data into `data/processed`
3. Prepare docker image: `make docker`
4. Make `ker.sh` executable and run train procedure:
```
chmod +x ker.sh
./ker.sh python /src/train.py
```
Be aware about path:
local dirs are mounted into root in docker container:
`src/->/src`, `data/->/data`, `models/->/models`



Current status:
1. Docker ready for python2, not for python3
2. Model runnable
3. Train runnable

TODO:
0. Add submission builder
1. Add opencv3 into Dockerfile
2. Make a submission!

Solution code based on https://github.com/jocicmarko/ultrasound-nerve-segmentation solution.

Main idea: rich augmentation technics and model tuning: BN and so on.


