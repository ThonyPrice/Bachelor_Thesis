# Manuscript for presentation

## Agenda

I'm Thony, this is Niklas. In this presentation we'll present a method which we've found increase the accuracy when it comes to diagnosing breast cancer.

During the presentation we'll address *Why* we have conducted the study.

Then, *How* we achieved our results, what methods we used and how it validates our findings.

Lastly we will present *what* these results conclude, how they are put context and *what* you could use them for.

## About breast cancer

To explain why we have done this study we need some background. Today breast cancer is the leading cause of cancer deaths among women. It origins from tumours which can be either benign or malignant. Where the former is relatively harmless and the latter could be fatal.

Currently there is no cure to prevent breast cancer but early detection drastically increases the chances of recovery.

The common way of detecting breast cancer is by mammography screenings. To classify a detected tumour a method is sampling cells from the tumour, staining a view under a microscope. Such as the image here. Medical experts can then use these images to examine if the tumour is benign or malignant.

Problem is, humans are slow and there is a shortage of this expertise. Studies has estimated an accuracy at about 70% in detection by radiologists.

The solution is, Computer aided diagnostics


## CAD and Machine Learning

Computer aided diagnostics utilises machine learning to do the classification and assist medial experts in their decisions.

We will not go into detail on machine learning but illustrate the high level concept. It begins with data.

Lets take the example of a the image we saw in the previous slide. The data is the image and the label is if the sample was benign or malignant. Lots of such images-key pairs are collected and is then used as the input.

Next we select a model. A machine learning model is the design of the algorithm that will make the predictions. These are a four commonly used in Computer Aided Diagnostics.

Then we can train this model on the input data. It predicts a classification based on an image and by having the key it iteratively can fine tune its parameters to become more accurate.

This trained model can then be used to classify new images and in such a way be utilised as an predictor.

So, each the input is an image. What *is* an image?

## Feature selection

We can explain an image as constituting of features. If the resolution is 256 by 256 the total dimension of the image is 256 squared.

Each pixel represents a dimension of the input to a machine learning model. As some features (in this case pixels) might not contain any useful information they can be considered redundant and don't add to the learning of a model.

Previous studies has shown reducing the amount of features can in cases improve the accuracy and significantly reduce the computation time.

We studied two types of feature selection methods.

### Filter methods

Where all features are ranked based on some condition, such as Entropy. The ranking is made as a preprocessing step and is independent of what Machine Learning classifier are to be used.

### Wrapper methods

Features are evaluated in conjunction with a classifier. That is each feature is tested with an calssifier and are selected based on some condition, such as highest scoring accuracy.

## Why?

Adding these pices of information we naturally asked, can feature selection improve the accuracy of classification of breast cancer?

And in this case one would like to know, in which classification methods is this improvement present?

And the why of our thesis motivated by answering these questions.

## Method

To make sure that our results were on a more universal level we chose to use four different datasets. Only using one dataset would only conclude that feature selection is improving accuracy on a single dataset for a certain classifier.     

Data was collected from the four different databases. They all contained different types of features on breast cancer from various patients. An example is the Wisconsin database, displayed here, that contained 30 different features from a Fine Needle Aspiration. (Fine Needle Aspiration is when you extracts tissue from the breast tumor with the help of a needle.)

The data was then preprocess with the help of four different feature selection methods. Two filter methods, Entropy and chi2, and two wrapper methods, Sequential Forward Selection and Sequential Backward Selection. An example is the Wisconsin data that is preprocessed by the Entropy method generating 30 different results depending on how many features we choose to select.

Lastly, the data is used as input in all the machine learning classifiers for training. The algorithms work there magic and can after training create prediction on test data from the same database as the input data originated from. Now we can calculate the accuracy of the predictions for eace combination of dataset, feature selection method and machine learning classifier and create a accuracy table.  Our Wisconsin dataset that used the Entropy filter method, can now use the artificial neural network to contribute with its part in the accuracy table.

This makes it a combination of 800 classification accuracy results.

## Classification improvements

Comparing the accuracy ahieved be the best feature selection method to the accuracy when using all features these are the results. Each bar is a distict dataset and the hight represents the increase or decrease in percent compared to using all features.

Its evident that ANN benefits the consistently by feature selection, performing students t-test also ensures the increase is significant.

The other classifiers benefit on some datasets and worsens the results on others, t-test concluded none neither an increase or decrase can be concluded on theese classifiers.

With the t-test results of the t-test and obvious appearence of this bar chart the conclusion seems kinda obvious? This is not the case, further analysis is needed.

## Further analysis

Consider we have three factors at play here. The datasets, feature selection methods and classifiers. The results could me a manifistation of prefrerence to ANN of the datasets or an interaction of datasets and feature selection methods.

We ran an Analysis of Variance, ANOVA, test to conclude the between which of these factors the interction is significant in respect to accuracy.

As we found that the interaction between classifier and feature selection method indeed has a significant interaction with expected accuracy. Thus rendering our result trustworthy.