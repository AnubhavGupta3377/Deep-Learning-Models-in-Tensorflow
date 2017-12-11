** This is my implementation of recurrent neural network languate model. I have coded RNNLM from scratch in Tensorflow.**

# Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
  
## RNNLM Implementation
  
 - Given words ![](https://latex.codecogs.com/gif.latex?x_1%2C%5Ccdots%20%2C%20x_t), a language model predicts the following
    word  ![](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D) by modeling:
	
	![](https://latex.codecogs.com/gif.latex?P%28x_%7Bt&plus;1%7D%20%3D%20v_j%20%7C%20x_t%2C%5Ccdots%20%2C%20x_1%29)

	where ![](https://latex.codecogs.com/gif.latex?v_j) is a word in the vocabulary.
  
  - Formally, the model is, for t = 1, . . . , n-1:
  
	![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cbegin%7Bsplit%7D%20e%5E%7B%28t%29%7D%20%26%3D%20x%5E%7B%28t%29%7DL%5C%5C%20h%5E%7B%28t%29%7D%20%26%3D%20%5Ctext%7Bsigmoid%7D%28h%5E%7B%28t-1%29%7DH%20&plus;%20e%5E%7B%28t%29%7DI%20&plus;%20b_1%29%5C%5C%20%5Chat%7By%7D%5E%7B%28t%29%7D%20%26%3D%20%5Ctext%7Bsoftmax%7D%28h%5E%7B%28t%29%7DU%20&plus;%20b_2%29%5C%5C%20%5Cbar%7BP%7D%28x_%7Bt&plus;1%7D%20%3D%20v_j%20%7C%20x_t%2C%5Ccdots%2Cx_1%29%20%26%3D%20%5Chat%7By%7D%5E%7B%28t%29%7D%20%5Cend%7Bsplit%7D%20%5Cend%7Balign*%7D)

where ![](https://latex.codecogs.com/gif.latex?h%5E%7B%280%29%7D%20%3D%20h_0%20%5Cin%20%5Cmathcal%7BR%7D%5E%7BD_h%7D) is some initialization vector for the hidden layer and ![](https://latex.codecogs.com/gif.latex?x%5E%7B%28t%29%7DL) is the product of L with
the one-hot row-vector ![](https://latex.codecogs.com/gif.latex?x%5E%7B%28t%29%7D) representing index of the current word. The parameters are:

![](https://latex.codecogs.com/gif.latex?%5C%5CL%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%7CV%7C%20%5Ctimes%20d%7D%20%5Chspace%7B12pt%7D%20H%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BD_h%20%5Ctimes%20D_h%7D%20%5Chspace%7B12pt%7D%20I%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20D_h%7D%20%5Chspace%7B12pt%7D%20b_1%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BD_h%7D%20%5Chspace%7B12pt%7D%20U%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BD_h%20%5Ctimes%20%7CV%7C%7D%20%5Chspace%7B12pt%7D%20b_2%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%7CV%7C%7D)

where ![](https://latex.codecogs.com/gif.latex?L) is the embedding matrix, ![](https://latex.codecogs.com/gif.latex?I) the input word representation matrix, ![](https://latex.codecogs.com/gif.latex?H) the hidden transformation
matrix, and ![](https://latex.codecogs.com/gif.latex?U) is the output word representation matrix.  ![](https://latex.codecogs.com/gif.latex?b_1) and  ![](https://latex.codecogs.com/gif.latex?b_2) are biases. ![](https://latex.codecogs.com/gif.latex?d) is the embedding dimension,
https://latex.codecogs.com/gif.latex?%7CV%7C is the vocabulary size, and ![](https://latex.codecogs.com/gif.latex?D_h) is the hidden layer dimension.
