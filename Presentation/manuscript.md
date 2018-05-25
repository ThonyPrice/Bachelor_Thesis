# Manuscript for presentation

## Agenda

I'm Thony, this is Niklas. In this presentation we'll present a method which we've found increase the accuracy when it comes to diagnosing breast cancer.
--
During the presentation we'll address *Why* we have conducted the study.
--
Then, *How* we achieved our results.
--
And lastly, *what* these results conclude, how they are put context and *what* you could use them for.

To begin explain *why* we did this study we need some background, staring with breast cancer.
--
## About breast cancer

Today breast cancer is the leading cause of cancer deaths among women. It origins from tumours which can be either benign or malignant. The benign form is relatively harmless. Malignant could be fatal.
--
Currently there is no cure to prevent breast cancer, but early detection drastically increases the chances of recovery.
--
The common way of diagnosing breast cancer is first detecting a tumour by mammography screenings. To classify a detected tumour *one* method is sampling cells from the tumour, staining them, and view under a microscope. Such as the image here. Medical experts can then use these images to examine if the tumour is benign or malignant.
--
Problem is, humans are slow and there is a shortage of this expertise. Studies has estimated an accuracy at about 70% in detection by radiologists.
--
But there is a solution: Computer aided diagnostics
--
## CAD and Machine Learning

Computer aided diagnostics utilises machine learning to do the classification and assist medial experts in their decisions.

We will not go into detail on machine learning but illustrate the high level concept. It begins with data.
--
Lets take the example of a the image we saw in the previous slide. The data is the image and the label is if the sample was benign or malignant. Lots of such images-key pairs are collected and is then used as the input.
--
Next we select a model. A machine learning model is the design of the algorithm that will make the predictions. These four are models which is commonly used in Computer Aided Diagnostics.
--
Then we can train this model on the input data. It predicts a classification based on an image, and by having the key it iteratively can fine tune its parameters to become more accurate.
--
This trained model can then be used to classify new images and in such a way be utilised as an predictor!
--
So, each the input is an image. What *is* an image really?
What does the input data look like?
--
## Input and its features

So, the image is built up by pixels. In this example we have 256 by 200 pixels. This results in an input dimension of 51 200. But do all of these pixels include valuable information?
--
Let's remove the pixels which do not include any cells, our input dimension is now vastly reduced.
--
Previous studies has shown reducing the amount of features can in some cases improve the accuracy, and significantly reduce the computation time. But how do we decide on what features to keep?
By *Feature selection*!
--
## Feature selection
In this study we decided to study feature selection methods from 2 types of "families".
--
### Filter methods
First filter methods. Here all features are ranked based on some condition, such as Entropy. It creates a ordered list of the most valuable features. The ranking is made as a preprocessing step and is independent of what Machine Learning classifier are to be used.
--
### Wrapper methods

Secondly Wrapper methods. Here features are evaluated in conjunction with a classifier. That is, each subset of features is tested with a classifier and is selected based on some condition, such as highest scoring accuracy.
--
## Why?

So adding these pieces of information we naturally asked, can feature selection improve the accuracy of classification on breast cancer?

In which classification methods is this suggested improvement present?

Niklas will tell you *How* we approached these questions.
--
## Method

To make sure that our results were on a more universal level we chose to use four different datasets. Only using one dataset would only conclude that feature selection is improving accuracy on a single dataset for a certain classifier.     

Data was collected from the four different databases. They all contained different types of features on breast cancer from various patients. An example is the Wisconsin database, displayed here, that contained 30 different features from a Fine Needle Aspiration.

The data was then preprocess with the help of four different feature selection methods. Two filter methods, Entropy and chi2, and two wrapper methods, Sequential Forward Selection and Sequential Backward Selection. An example is the Wisconsin data that is preprocessed by the Entropy method generating 30 different results depending on how many features we choose to select.

Lastly, the data is used as input in all the machine learning classifiers for training. The algorithms work there magic and can after training create prediction on test data from the same database as the input data originated from. Now we can calculate the accuracy of the predictions for eace combination of dataset, feature selection method and machine learning classifier and create a accuracy table.  Our Wisconsin dataset that used the Entropy filter method, can now use the artificial neural network to contribute with its part in the accuracy table.

This makes it a combination of 800 classification accuracy results. But how should be interpreted?

## Classification improvements

Comparing the accuracy achieved be the best feature selection method to the accuracy when using all features these are the results. Each bar is a distinct dataset. The height represents the ratio of increase or decrease in accuracy compared to using all features.

Its evident that ANN benefits consistently by feature selection on every dataset, performing students t-test also ensures this increase is significant.

The other classifiers benefit on some datasets and worsens the results on others. However, neither an increase or decrease can be statistically concluded on these classifiers.

With these results the conclusion seems kinda obvious? Not at all, further analysis is needed!
--
## Further analysis

Consider we have three factors at play here. The datasets, feature selection methods and classifiers. The results could me a manifistation of prefrerence to ANN of the datasets or an interaction of datasets and feature selection methods.

We ran an Analysis of Variance test to conclude the between which of these factors the interaction is significant in respect to accuracy.
--
We found that the interaction between classifier and feature selection method indeed has a significant significant effect on what accuracy to expect. Because of this, we're confident our results are trustworthy.

With these results proven, Niklas will tell you *what* these results conclude and how to put them into context.
--
## Conclusion an Discussion

Applying feature selection methods to a Artificial Neural Network provides an improved classification accuracy of benign or malignant breast cancer.

This is an important result since it shows that Feature selection is an significant part of optimizing ANN for breast cancer. However, we have to remember that we did not use all possible feature- data for breast cancer. There are of course other methods for extracting data then the once used in our datasets. On new promising method is extracting RNA- data from patients blood samples. The RNA data contains over 1900 features for each patient. Our methodology could be used for evaluating the effect of feature selection on that type of data as well or be expanded to trying to getting state of the art results in regard to breast cancer classification. However, that would require a lot of computational power which brings us to our next subject.

Should you use wrapper or filter methods?
well, the methods that improve our result the most where generally the wrappers, if looking at the number of combinations. But you have to take into a count the demand large computational time for wrappers compared to filter methods. Basically 99% of our computational time went to the wrapper methods. If we where to only use filter methods we could have expanded the research to more then two filter methods four datasets.

To summarize:

Why did we conduct this study, well CAD, Machine learning and Feature selection all shows great potential to help in the fight agains breast cancer.

We tried 64 combinations of methodology for classification and extracted 800 results. Did rigorous evaluation of the results to prove there reliability and could draw the conclusion:   

what, Applying feature selection methods to a Artificial Neural Network provides an improved classification accuracy of benign or malignant breast cancer.

Before wrapping up this presentation we want you to remember one more thing:

Using all the data does not always bring the best results.

<!--

When using classifiers Decision Tree, Na\"ive Bayes and Support Vector Machine no increase, or decrease of accuracy could be proven using feature selection. -->
