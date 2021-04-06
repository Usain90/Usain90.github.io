---
title: "Detecting Fake News With Python and Machine Learning"
date: 2021-04-01
tags: [data analysis, data science,Python,Machine Learning]
header:
  image: "/images/photos/fake_news_cover.jpg"
excerpt: "Fake News, Data Science, Python, Data Analysis"
mathjax: "true"
---
### Detekting Fake News With Python and Machine Learning:
In this project I will try to answer some basics questions related to the titanic tragedy using Python.

A king of yellow journalism, fake news is false information and hoaxes spread through social media and other online media to achieve a political agenda. In this data science project idea, we will use Python to build a model that can accurately detect whether a piece of news is real or fake. We’ll build a *TfidfVectorizer* and use a *PassiveAggressiveClassifier* to classify news into “Real” and “Fake”. We’ll be using a dataset of shape 7796×4 and execute everything in Jupyter Notebook.

## Let's start with defining some related terms:
TfidfVectorizer: Transforms text to feature vectors that can be used as input to estimator when TF: is term frequency and IDF: is Inverse Document Frecuency.

PassiveAggressiveClassifier: are generally used for large-scale learning. It is one of the few ‘online-learning algorithms‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example.
