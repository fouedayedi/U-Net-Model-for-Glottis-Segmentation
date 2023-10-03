# Glottis Segmentation with U-Net

## Overview

This project focuses on leveraging the U-Net model for the segmentation of the glottis area in high-speed videoendoscopy frames derived from the BAGLS dataset. The BAGLS dataset comprises a substantial and diverse multihospital dataset that includes 59,250 high-speed videoendoscopy frames, each accompanied by individually annotated segmentation masks.

The process of analyzing these recordings traditionally requires a meticulous segmentation of the glottal area by trained experts, which is notably time-consuming. This project aims to automate this process using the U-Net model to effectively and accurately perform glottis segmentation.

## Dataset

The utilized BAGLS dataset is a large, multihospital collection, offering 59,250 high-speed videoendoscopy frames, each with an individually annotated segmentation mask. Laryngeal videoendoscopy is a prominent tool in clinical examinations concerning voice disorders and voice research. While high-speed videoendoscopy enables the complete capture of vocal fold oscillations, processing the recordings often entails a laborious segmentation of the glottal area by trained experts.

## Prerequisites

Ensure you have the following prerequisites installed to execute the model training:

- Python 3.7 or later
- Dependencies from `requirements.txt`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Training

The U-Net model utilized in this project was trained on a subset of the available data, specifically a randomly selected portion of the training dataset. The training was conducted over multiple epochs and allowed for modification of parameters like batch size and learning rate. This is due to resource constraints, but you can modify subset=1 to train on the entire dataset if desired.
```bash
python main.py
```

## Results

For a detailed exploration of our results please refer to our Kaggle Notebook.

📘 [View the Results on Kaggle](https://www.kaggle.com/fouedayedi/u-net-model-for-glottis-segmentation-analysis)

## Acknowledgements

We extend our sincere appreciation to the creators and curators of the BAGLS dataset, which served as the pivotal foundation for this project. Their meticulous efforts in assembling and maintaining such a valuable resource have profoundly enabled and enriched our exploration into the realm of automated glottis segmentation.

For those interested in delving deeper into the dataset’s creation and specifics, we recommend consulting the original publication:
- [BAGLS: A Multihospital Benchmark for Automatic Glottis Segmentation](https://doi.org/10.1038/s41597-020-0526-3) | DOI: 10.1038/s41597-020-0526-3


