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

If you want to train model or make some changes to the neural network model, you should upload `network_training.ipynb` jupyter notebook and run the required cells. Сomments in the code will help you navigate
  
## Gender Classification task description
