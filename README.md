# Image-and-Label Conditioning Latent Diffusion Model ：Synthesizing Aβ-PET from MRI for Detecting Amyloid Status

This repo contains the official Pytorch implementation of the paper: Image-and-Label Conditioning Latent Diffusion Model ：Synthesizing Aβ-PET from MRI for Detecting Amyloid Status

## Contents

1. [Summary of the Model](#1-summary-of-the-model)
2. [Setup instructions and dependancies](#2-setup-instructions-and-dependancies)
3. [Running the model](#4-running-the-model)
4. [Some results of the paper](#5-some-results-of-the-paper)
5. [Contact](#6-contact)
6. [License](#7-license)

## 1. Summary of the Model

The following shows the training procedure for our proposed model

<img src= image\framework.png>

The image-and-label conditioning latent diffusion model (IL-CLDM) is proposed to synthesize Aβ-PET scans from MRI scans for providing a reliable MRI-based alternative of classifying Aβ as positive or negative. The proposed model incorporates both image and label conditioning modules into the denoising network for preserving subject-specific and diagnosis-specific information.

## 2. Setup instructions and dependancies

For training/testing the model, you must first download ADNI dataset. You can download the dataset [here](https://adni.loni.usc.edu/data-samples/access-data/). Also for storing the results of the validation/testing datasets, checkpoints and loss logs, the directory structure must in the following way:

    ├── data                # Follow the way the dataset has been placed here
    │   ├── whole_Abeta        # Here Abeta-PET images must be placed
    │   └── whole_MRI          # Here Abeta-PET images must be placed
    │   └── latent_Abeta       # This can be empty since the encoding process will generate these latent representations of original Abeta-PET images
    ├── data_info          # Follow the way the data info has been placed here
    │   ├── data_info.csv       # This file contains labels for each ID
    │   ├── train.txt           # This file contains the ID of training dataset, like '037S6046'
    │   └── validation.txt      # This file contains the ID of validation dataset
    │   └── test.txt            # This file contains the ID of test dataset
    ├── result             # Follow the way the result has been placed here
    │   ├── exp_1              # for experiment 1
    │   │   └── CHECKPOINT_AAE.pth.tar      # This file is the trained checkpoint for AAE in the first stage
    │   │   └── CHECKPOINT_Unet.pth.tar     # This file is the trained checkpoint for Unet in the second stage
    │   │   └── loss_curve.csv              # This file is the loss curve for two stages
    │   │   └── validation.csv              # This file is the indicator files for two stages in the validation set
    │   │   └── test.csv                    # This file is the indicator files for two stages in the test set
    ├── config.py          # This is the configuration file, containing some hyperparameters
    ├── dataset.py         # This is the dataset file used to preprocess and load data
    ├── main.py            # This is the main file used to train and test the proposed model
    ├── model.py           # This is the model file, containing two models (AAE and Unet)
    ├── README.md
    ├── utils.py           # This file stores the helper functions required for training

## 3. Running the model

Users can modify the setting in the config.py to specify the configurations for training/validation/testing. For training/validation/testing the our proposed model:

```
python main.py
```

## 4. Some results of the paper

Some of the results produced by our proposed model and competitive models are as follows. *For more such results, consider seeing the main paper and also its supplementary section*

<img src=image\result.png>

## 5. Contact

If you have found our research work helpful, please consider citing the original paper.

If you have any doubt regarding the codebase, you can open up an issue or mail at ouzx2022@shanghaitech.edu.cn

## 6. License

This repository is licensed under MIT license
