# SentimentAnalysis
sentiment analysis of Yelp data using Keras


## Attention Model
Attention is good at predicting time-series data.

However, it can also be used for sentiment analysis which showed a relatively good performance but lacked in efficiency.

## RCNN
RCNN implementation from reference to a [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552).

It shows a relatively good performance but also lacks in efficiency (running time).

However, if finely-tuned, it is expected to show one of the best accuracy.

## Model Generator
It is the model geneartor of the final model selected (1 Layer of 1D CNN + 2 Layers of Bidirectional LSTM).

This has running time of about 50 seconds per epoch which has optimum learning at 10 ~ 15 epochs.

It is used to generate as many models possible for Ensembling. Ensemble method can improve the accuracy by about 2%.

It runs in Google colab's environment using Google Drive APIs.

## Calculate Average
This is used to load models to predict each input and used for ensembling.

This should also run in same environment as model generator (due to GPU usage) and it uses Google Drive APIs.

## HAN
Hierarchical Attention Network

## LGBM
Light Gradient Boosting Model (Built by Microsoft)
