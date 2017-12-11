# Author
* ** Anubhav Gupta

  - This is my implementation of recurrent neural network languate model.
    I have coded RNNLM from scratch in Tensorflow.

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
  
## RNNLM Implementation
  
  - Given words $x_1,\cdots , x_t$, a language model predicts the following
    word $x_{t+1}$ by modeling:
	
	![](https://latex.codecogs.com/gif.latex?P%28x_%7Bt&plus;1%7D%20%3D%20v_j%20%7C%20x_t%2C%5Ccdots%20%2C%20x_1%29)

	where vj is a word in the vocabulary.
  
  - Formally, the model is, for t = 1, . . . , n-1:
  
	
