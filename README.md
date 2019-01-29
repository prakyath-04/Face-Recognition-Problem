# Face-Recognition-Problem
Classified set of images, belonging to different classes, using Naive Bayes and Softmax Classifiers


### Principal Component Analysis

- Performed PCA over all images to the reduce the size of the input images. 

- Plotted a graph showing the total mean square error over all train images
  vs the number of principal components used to reconstruct.


### Naive Bayes Multi-class Classifier

- Given train images-labels and test images, reduced the images to a vector
  representation of length 32 using PCA.
  
- Trained a Naive bayes classifier using train images and then predicted the
  labels of test images.
  
- Code will run as : `python naive_bayes.py <path-to-train-file> <path-to-test-file>`

### Linear Multi-class Classifier

- Given train images-labels and test images, reduced the images to a vector
  representation of length 32 using PCA.
  
- Trained  the linear classifiers with gradient descent using softmax probabilies to calculate loss.
- For each test image, predicted its label.

- Code will run as `python linear_classifier.py <path-to-train-file> <path-to-test-file>`

#### Structure of train file
```
<absolute-path-to-train-1> <label-1>
<absolute-path-to-train-2> <label-2>
.
.
.
<absolute-path-to-train-N> <label-N>
```

#### Structure of test file
```
<absolute-path-to-test-1>
<absolute-path-to-test-2>
.
.
.
<absolute-path-to-test-M>
```

#### output is as follows:
```
<test-label-1>
<test-label-2>
.
.
.
<test-label-M>
```


### dataset links

[dataset link](https://drive.google.com/open?id=141FxNzJM1aNa8RL9CWb6ayFNqNNIzALe)

[testdata link](https://drive.google.com/open?id=1KlGMqbVNaBUFIWHubFMmUkn9UvcX00rK)

### How to run the eval.py 

- The dataset folder must be in the same directory as the create_dataset1.py file.

- Run the create_dataset1.py file to generate random input.

- Run the eval.py to compute the accuracies.
