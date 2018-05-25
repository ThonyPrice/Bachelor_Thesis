# Manuscript for presentation

## Agenda

I'm Thony, this is Niklas. In this presentation we'll present a method which we've found increases the accuracy when it comes to diagnosing breast cancer.

During the presentation we'll address Why we've conducted the study.

Then, How we achieved our results, by what methods and proof of thier trustworthiness.

And last what these results conclude and should be put into context.

## About breast cancer

Something about tumours... They can be either benign or malignant, that is fatal or harmless(?)

Today breast cancer is the leading cause of cancer deaths among women. However, early detection increases the chaces of survival and recovery.

The common way of classifying breast cancer mammography screenings. An image as the one here is produced and radiologist examine it to determine if a tumour is benign or malignant.

The process is both time consuming and studies has shown there is a large shortage of radiologists as mammographies are increasingly used.

Other studies has measured the accuracy of radiologist and estimated it to 70%. There has been work made to make this process more efficient.

That is, Computer aided diagnostics


## CAD and Machine Learning

Computer aided diagnostics utilizes machine learning to the classification and assist medial experts in their decision.

The high level concept of machine lerning begins with data. Lets take the example of a mammographic scan. Lots of images are collected and labeled by radiologists to have a key for each datapoint.

A Machine Learning algorithm is trained on this data. It predicts a classification based on an image and by having the key it iteratively can fine tune its parameters to become more accurate.

This trained model can then be used to classify new images and in such a way be utilized as an predictor.

Overgang!

## Feature selection

To explain feature selection let's continue with the example of the image. If the resolution is 256 by 256 the total dimension of the image is 256 squared.

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

## Further analysis

More text...
































## Conclusion an Discussion

Applying feature selection methods to a Artificial Neural Network provides an improved classification accuracy of benign or malignant breast cancer.

This is an important result since it shows that Feature selection is an significant part of optimizing ANN for breast cancer. However, we have to remember that we did not use all possible feature- data for breast cancer. There are of course other methods for extracting data then the once used in our datasets. On new promising method is extracting RNA- data from patients blood samples. The RNA data contains over 1900 features for each patient. Our methodology could be used for evaluating the effect of feature selection on that type of data as well or be expanded to trying to getting state of the art results in regard to breast cancer classification. However, that would require a lot of computational power which brings us to our next subject.

Should you use wrapper or filter methods?
well, the methods that improve our result the most where generally the wrappers, if looking at the number of combinations. But you have to take into a count the demand large computational time for wrappers compared to filter methods. Basically 99%-100% of our computational time went to the wrapper methods. If we where to only use filter methods we could have expanded the research to more datasets.  


Before wrapping up this presentation we want you to remember one more thing:

Using all the data does not always bring the beat results.

<!-- 

When using classifiers Decision Tree, Na\"ive Bayes and Support Vector Machine no increase, or decrease of accuracy could be proven using feature selection. -->
