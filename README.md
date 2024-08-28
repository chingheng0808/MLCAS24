# MLCAS Corn Yield Prediction Using Satellite Data

## Project Introduction

The project is using deep learning methods to predict the final corn yield on the MLCAS Dataset.
In addition to using satellite data, we not only integrate the text information and additional information about the captured area block, but we also specifically use the histogram information on each band as our input feature. We use a multi-model architecture to train the models.

## Dataset & Pre-trained Models

Dataset:
Please download the dataset from the link and unzip it -> https://drive.google.com/file/d/1lICfveVrGHaqTcaM0LG6HLZHorbZdgMh/view?usp=share_link
After unzipping, make the  

Pre-trained models:
https://drive.google.com/file/d/1lgi_t6Eq8fbN0rLBrC-GiTVQFr4A2gNj/view?usp=share_link
https://drive.google.com/file/d/1IkHpf7c1WCs0d7VV1u8wSwYfyCMmKYfA/view?usp=share_link

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
│   │   model_L1L2_ADP_stat_lstm_64nf_4rb_64h_stepLR_3ai_epoch-800.pth\
│   │   ...\
│\
│   dataset_dul_latest.py\
│   dataset_fast_latest.py\
│   dataset_test.py\
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
clip=1.0
rasterio=1.2.10

## Instruction

### 0. Download the pretrained weight and data

https://drive.google.com/drive/folders/1-U2i5f-7DOhYtO2xu2sUo-jNGEri-MvJ?usp=sharing

### 1. Generate Data

Run `generate_data.py` to generate our training data using the 2022 and 2023 dataset from the 'MLCAS24_Competition_data' directory.

```bash
python generate_data.py
```

### 2. Training Each Model, Seperately

```bash
python train_adpW_hist.py
python train_normal_hist.py
python train_normal.py
```

### 4. Ensemble ALl Prediction Results

```bash
python ensemble.py
```

### 5. Results

The finally ensemble result is in the 'result_Ensemble' folder. As for each models' output results are in the 'results_test'.
# MLCAS_HI123
