# Kaggle Bengali.AI Handwritten Grapheme Classification

[**WandB Report**](https://app.wandb.ai/masterscrat/bengaliai/reports/BengaliAI-Kaggle--Vmlldzo2MzY0Mw)

## To try

- [x] Weights & Biases integration

- [ ] Try Fleuret paper for sample prioritization!

- [ ] Optimize [fit_generator](https://keras.io/models/model/#fit_generator) parameters: more workers? May need to [reduce queue size in this case](https://stackoverflow.com/a/45539517/318557). Will need `use_multiprocessing`.

- [ ] Cutmix augmentation (see below). [Keras kernel](https://www.kaggle.com/code1110/mixup-cutmix-in-keras). Target [224x224](https://www.kaggle.com/c/bengaliai-cv19/discussion/123198#719100)?

- [ ] Train with OneCycle learning rate (https://github.com/titu1994/keras-one-cycle#training-with-onecyclelr)

- [ ] Estimate optimal LR with LRFinder (https://github.com/surmenok/keras_lr_finder#usage)

## Setup

API credentials: https://github.com/Kaggle/kaggle-api#api-credentials

Download the dataset:

```
# vim ~/.kaggle/kaggle.json
pip install kaggle --upgrade
kaggle competitions download -c bengaliai-cv19
unzip bengaliai-cv19.zip -d bengaliai-cv19
```

The model consists of an EfficientNet B3 pre-trained (on Imagenet) model with a generalized mean pool and custom head layer.
For image preprocessing I just invert, normalize and scale the image... nothing else. No form of augmentation is used.

Create from existing environments:

```
# Update conda
conda update -n root conda

# Linux w/ GPU
conda env create -f linux-gpu-env.yml

# MacOS wo/ GPU
conda env create -f mac-env.yml
```

Backup solution (using `--from-history` which doesn't capture pip packages):

```
conda env create -f env-from-history.yml 
pip install -U iterative-stratification efficientnet tqdm
```

Creating the env from scratch:

```
conda create --name bengaliai-env python=3.6 --yes
conda activate bengaliai-env
echo $CONDA_DEFAULT_ENV
conda install -c anaconda tensorflow-gpu==2 --yes
# without GPU:
#conda install tensorflow==2 --yes
conda install keras==2.3.1 --yes
conda install -c menpo opencv --yes
conda install -c trent-b iterative-stratification --yes
# on linux had to use pip:
# pip install iterative-stratification
pip install -U efficientnet
conda install -c conda-forge fastparquet --yes
conda install python-snappy
# on MacOS may need snappy decompression for fastparquet:
# brew install snappy
# CPPFLAGS="-I/usr/local/include -L/usr/local/lib" pip install python-snappy
```

TODO add wandb to setup

Set the following path in `train.py`:
- DATA_DIR (directory with the Bengali.AI dataset)
- TRAIN_DIR (directory to store the generated training images) 
- GENERATE_IMAGES (whether to extract images from parquet files, should be done initially)

Training:

`clear; time python train.py`

Inference kernel: https://www.kaggle.com/rsmits/keras-efficientnet-b3-training-inference

## Papers

### EfficientNet

https://arxiv.org/abs/1905.11946

>  In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.

### Generalized-Mean (GeM) pooling

https://arxiv.org/abs/1711.02512

> A novel trainable Generalized-Mean (GeM) pooling layer that generalizes max and average pooling and boosts retrieval performance

### Cutmix augmentation (to test!)

https://arxiv.org/abs/1905.04899

> CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features

Mentioned a lot in top models: https://www.kaggle.com/c/bengaliai-cv19/discussion/123198

Implementations:
- https://github.com/DevBruce/CutMixImageDataGenerator_For_Keras

### Contrast Limited Adaptive Histogram Equalization

https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE

Used eg here: https://www.kaggle.com/devbruce/kakr-2019-3rd-cutmix-ensemble-keras#Image-Preprocessing---CLAHE

Supported by `albumentations`

## Packages

### iterative-stratification

https://github.com/trent-b/iterative-stratification

> iterative-stratification is a project that provides scikit-learn compatible cross validators with stratification for multilabel data.

## Training

(Comments from original author below)

I've trained the model for 80 epochs and picked some model weight files todo an ensemble in the inference kernel. 

I first tested the training part by using 5/6th of the training data for training and 1/6th for validation. Based on the validation and some leaderboard submissions I found that the highest scoring epochs were between epoch 60 - 70. In the final training (as is used in this code) I use a different distribution of the training data for every epoch. The downside of this is that validation doesn't tell you everything anymore. The major benefit is however that it increases the score about 0.005 to 0.008 compared to the use of the fixed training set. This way it get close to what a Cross Validation ensemble would do...just without the training needed for that.

The model weight files as used in the inference kernel are available in folder 'KaggleKernelEfficientNetB3\model_weights'. It contains the following 4 files (for each file mentioned the LB score when used as a single file to generate the submission):
- Train1_model_59.h5     LB 0.9681
- Train1_model_64.h5     LB 0.9679
- Train1_model_66.h5     LB 0.9685
- Train1_model_68.h5     LB 0.9691

