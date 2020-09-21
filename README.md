# Test tasks for the internship in NtechLab

## Tasks description
### Task 1. Programming

Goal of this task was to find a continuous subarray in the array containing at least one number, which has the biggest sum.

implemented algorithm that solves this problem, contains `find_Max_Sub_Array(A)` function, which accepts an array of integers A of nonzero length and returns a continuous subarray of array A of non-zero length, which has the biggest sum among all continuous subarrays of array A.

A few test cases for checking the correctness of the result were also implemented.

### Task 2. Machine learning (deep learning)

Goal of this task was to train a neural network that can determine the gender of a person in the image from the input image of the face.

Train dataset contains 50 thousands female images and 50 thousands male images of different sizes.

## Structure of the repository
* `task1.ipynb` -- file contains code of the first programming task
* `task2_gender_classification` -- folder contains following files related to the task of gender classification:
  * `network_training.ipynb` -- jupyter notebook with the code of network model creation and training
  * `test_network.py` -- script for processing test data. Receive one argument -- the directory where the folder with test data is located
  * `gender_classification_net.pth` -- file with the pretrained neural network model 
  
## Instructions for running tasks
### Task 1. Programming

You should upload jupyter notebook with the code, and run cell with the `find_Max_Sub_Array(A)` function creation.
After that you can test `find_Max_Sub_Array(A)` on any appropriate array (expected type of the array: list) or run other cells and see, how it works on the given samples.

### Task 2. Machine learning (deep learning)

You can work with the given files in two ways:

#### Classify images using pretrained model

If you want to classify your own images using pretrained CNN model but do not want to run or make changes in model training, you need to upload two files from the `task2_gender_classification` folder: `gender_classification_net.pth` -- file with the pretrained neural network model and the script `test_network.py`.

Both files should be placed in the same folder. Script `test_network.py` receive one argument -- the directory where the folder with test data is located. The directory with the test data should be placed in the same folder, where the script and network model file are located. Also note that for the script to work correctly, the passed folder must contain a subdirectory with test data. This is due to the specifics of the pytorch DataLoader.

#### Train the model

If you want to train model or make some changes to the neural network model, you should upload `network_training.ipynb` jupyter notebook and run the required cells. Сomments in the code will help you navigate. Note that the train data should be placed in the folder 'internship_data' if you don't want to change the folder path.
  
## Gender Classification task description

### Data preprocessing

Since initially the images in the training set had different sizes, they were resized to the same size 64x64. 

After resizing all images were converted to pytorch tensors and normalized.

### Neural network architecture selection

The choice was made among various convolutional neural network architectures, because they are resistant to shifts and rotations and have proven themselves well in solving a big amount of problems related to image analysis, including binary classification.

The architecture of my neural network is inspired by the LeNet-5 architecture. The main differences from the classical version are the use of ReLU non-linearity instead of sigmoid and tanh activation functions (as in many other similar examples, here the use of ReLU allowed to improve the quality of classification) and the use of a 5-by-5 filter in the first convolutional layer instead of a 3-by-3 filter. This also led to improved results.

### Parameters learning

The weights of the neural network were trained using the error back propagation method. 

The Cross Entropy function was used as the loss function. Cross Entropy, which minimizes the distance between two probability distributions -- predicted and actual, it is well suited for classification tasks.

Stochastic gradient descent was used as a method for optimizing (minimizing) the loss function. Advantage of using stochastic gradient descent is that on massive datasets, it can converges faster because it performs updates more frequently. In addition, it proved to be better than other optimization methods when compared in practice. Learning rate and momentum for SGD were also found empirically.

### Achieved results

Training set, which contains 100'009 images was splitted into training and validation sets. Training set has size of 95'008 samples and the validation set has size of 5'001 sample. The accuracy of classification of validation samples is 95%.
