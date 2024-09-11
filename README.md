# MLCAS Corn Yield Prediction Using Satellite Data

## Project Introduction

The project is using deep learning methods to predict the final corn yield on the MLCAS Dataset.
In addition to using satellite data, we not only integrate the text information and additional information about the captured area block, but we also specifically use the histogram information on each band as our input feature. We use a multi-model architecture to train the models.

## Dataset & Pre-trained Models

Dataset:
Please download the dataset from the link and unzip it -> https://drive.google.com/file/d/1lICfveVrGHaqTcaM0LG6HLZHorbZdgMh/view?usp=share_link
After unzipping, replace 'MLCAS24_Competition_data' folder as the unzipped folder.

Pre-trained models:
https://drive.google.com/file/d/1pcaV-NY5Y8oOBszs3EFDaf1h4Gpbjr7j/view?usp=sharing
https://drive.google.com/file/d/1MnWUkyy6myQQ1BGLAdJ00iTVwkynwHeu/view?usp=sharing

## Directory Structure

HI123
├── data_latest\
│   │   data0.npz\
│   │   data0.txt\
│   │   data1.npz\
│   │   data1.txt\
│   │   ...\
│\
└───MLCAS24_Competition_data/ <-*Dataset*\
│   └───test\
│   │   │...\
│   └───train\
│   │   │...\
│   └───validation\
│   │   │...\
│\
└───models\
│   │   model_L1L2_ADP_stat_lstm_64nf_4rb_64h_stepLR_3ai_epoch-200.pth\
│   │   ...\
│\
│   dataset_dul_latest.py\
│   dataset_fast_latest.py\
│   dataset_test.py\
│   ensemble.py\
│   generate_data.py\
│   model_adp_hist.py\
│   model_transformer_adp_stat.py\
│   predict_latest.py\
│   README.md\
│   train_normal_hist.py\
│   train_normal.py\

## Environment

torch=1.12.0+cu116, torchvision=0.13.0+cu116, torchaudio=0.12.0+cu116
numpy=1.21.6
pandas=1.3.5
tqdm=4.66.4
matplotlib=3.5.3
openai-clip=1.0.1
rasterio=1.3.11

## Instruction

### 0. Check the Directory Structure is Correct as Above

### 1. Generate Data

Run `generate_data.py` to generate our training data using the 2022 and 2023 dataset from the 'MLCAS24_Competition_data' directory.

```bash
python generate_data.py
```

### 2. Training Each Model, Seperately

```bash
python train_normal_hist.py
python train_normal.py
```

### 4. Ensemble All Prediction Results

```bash
python ensemble.py
```

### 5. Results

The finally ensemble result is in the 'result_Ensemble' folder. As for each models' output results are in the 'results_test'.
# MLCAS_HI123
