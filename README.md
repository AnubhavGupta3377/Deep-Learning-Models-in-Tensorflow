********************************************
*         Author: Anubhav Gupta            *
********************************************

  - This is my implementation of recurrent neural network languate model.
    I have coded RNNLM from scratch in Tensorflow.

*****************************************
*         RNNLM Implementation          *
*****************************************

  - Python 3.5.4
  - Tensorflow 1.2.1
  
  - Given words $x_1,\cdots , x_t, a language model predicts the following
    word x_{t+1} by modeling:
	
	P(x_t+1 = v_j | x_t, . . . , x_1)

	where vj is a word in the vocabulary.
  
  - Formally, the model is, for t = 1, . . . , n-1:
  
	
