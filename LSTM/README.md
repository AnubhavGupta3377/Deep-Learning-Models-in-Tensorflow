# Long Short Term Memory (LSTM) for Language Modeling
- Here, I have implemented LSTM from scratch in Tensorflow for language modeling task.
  
## LSTM Details
  
 - Long-Short-Term-Memories are another type of complex activation unit that differ a little from GRUs. The motivation for using these is similar to those for GRUs however the architecture of such units does differ. Let us first take a look at the mathematical formulation of LSTM units before diving into the intuition behind this design:
 
 - The model is, for t = 1, ..., n-1:
 
      ![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20i_t%20%26%3D%20%5Csigma%28W%5E%7B%28i%29%7Dx_t%20&plus;%20U%5E%7B%28i%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2.15cm%7D%5Ctextbf%7B%28Input%20gate%29%7D%5C%5C%20f_t%20%26%3D%20%5Csigma%28W%5E%7B%28f%29%7Dx_t%20&plus;%20U%5E%7B%28f%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2cm%7D%5Ctextbf%7B%28Forget%20gate%29%7D%5C%5C%20o_t%20%26%3D%20%5Csigma%28W%5E%7B%28o%29%7Dx_t%20&plus;%20U%5E%7B%28o%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2.1cm%7D%5Ctextbf%7B%28Output%20gate%29%7D%5C%5C%20%5Ctilde%7Bc%7D_t%20%26%3D%20%5Ctanh%28W%5E%7B%28c%29%7Dx_t%20&plus;%20U%5E%7B%28c%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B1.5cm%7D%5Ctextbf%7B%28New%20memory%29%7D%5C%5C%20c_t%20%26%3D%20f_t%5Codot%20c_%7Bt-1%7D%20&plus;%20i_t%20%5Codot%20%5Ctilde%7Bc%7D_t%20%5Chspace%7B2.7cm%7D%5Ctextbf%7B%28Final%20memory%29%7D%5C%5C%20h_t%20%26%3D%20o_t%20%5Codot%20%5Ctanh%28c_t%29%20%5Chspace%7B3.5cm%7D%5Ctextbf%7B%28Final%20hidden%20state%29%7D%5C%5C%20%5Chat%7By%7D_t%20%26%3D%20%5Ctext%7Bsoftmax%7D%28h%5E%7B%28t%29%7DU%20&plus;%20b%29%5C%5C%20%5Cbar%7BP%7D%28x_%7Bt&plus;1%7D%20%3D%20v_j%20%7C%20x_t%2C%5Ccdot%2Cx_1%29%20%26%3D%20%5Chat%7By%7D_t%20%5Cend%7Balign%7D)
      
 - The above equations can be thought of a LSTM's fundamental operational stages and they have intuitive interpretations that make this model much more intellectually satisfying. Stages of LSTM can be thought of as performing the following operations:
 
     - **New memory**: This stage is analogous to the new memory generation stage we saw in GRUs. We essentially use the input word ![](https://latex.codecogs.com/gif.latex?x_t) and the past hidden state ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) to generate a new memory ![](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bc%7D_t) which includes aspects of the new word ![](https://latex.codecogs.com/gif.latex?x_t).
  
     - **Input gate:** We see that the new memory generation stage doesn't check if the new word is even important before generating the new memory - this is exactly the input gate's function. The input gate uses the input word and the past hidden state to determine whether or not the input is worth preserving and thus is used to gate the new memory. It thus produces ![](https://latex.codecogs.com/gif.latex?i_t) as an indicator of this information.
  
     - **Forget gate**: This gate is similar to the input gate except that it does not make a determination of usefulness of the input word – instead it makes an assessment on whether the past memory cell is useful for the computation of the current memory cell. Thus, the forget gate looks at the input word and the past hidden state and produces ![](https://latex.codecogs.com/gif.latex?f_t).

     - **Final memory generation:** This stage first takes the advice of the forget gate ![](https://latex.codecogs.com/gif.latex?f_t) and accordingly forgets the past memory ![](https://latex.codecogs.com/gif.latex?c_%7Bt-1%7D). Similarly, it takes the advice of the input gate ![](https://latex.codecogs.com/gif.latex?i_t) and accordingly gates the new memory ![](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bc%7D_t). It then sums these two results to produce the final memory ![](https://latex.codecogs.com/gif.latex?c_t).
     
     - **Output/Exposure Gate:** This is a gate that does not explicitly exist in GRUs. It's purpose is to separate the final memory from the hidden state. The final memory ![](https://latex.codecogs.com/gif.latex?c_t) contains a lot of information that is not necessarily required to be saved in the hidden state. Hidden states are used in every single gate of an LSTM and thus, this gate makes the assessment regarding what parts of the memory ![](https://latex.codecogs.com/gif.latex?c_t) needs to be exposed/present in the hidden state ![](https://latex.codecogs.com/gif.latex?h_t). The signal it produces to indicate this is ![](https://latex.codecogs.com/gif.latex?o_t) and this is used to gate the point-wise tanh of the memory.

 - The output vector ![](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%5E%7B%28t%29%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%7CV%7C%7D) is a probability distribution over the vocabulary, and we optimize the (unregularized) cross-entropy loss:

	![](https://latex.codecogs.com/gif.latex?J%5E%7B%28t%29%7D%28%5Ctheta%29%20%3D%20%5Ctext%7BCE%7D%28y%5E%7B%28t%29%7D%2C%20%5Chat%7By%7D%5E%7B%28t%29%7D%29%20%3D%20-%5Csum_%7Bi%3D1%7D%5E%7B%7CV%7C%7Dy%5E%7B%28t%29%7D_i%20%5Ctext%7Blog%7D%7E%5Chat%7By%7D%5E%7B%28t%29%7D_i)
	
	where ![](https://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D) is the one-hot vector corresponding to the target word (which here is equal to ![](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D).

## Results
  - Some Example sentences generated using the GRU models trained for 16 epochs:
  	1. how to prestigious end it revised for $ N <eos>
	2. in palo alto prepare at the first two years of the <unk> 's foreign issues had an after-tax N N gain with another acquisition of commercial banks <eos>
	3. india is about $ N chip <unk> and in two years and mercury their usual and block air reports are made $ N is followed that mr. bergsma and the approval on agreement president died the members of bay growth <eos>
	4. is english into the firm in the nation 's dance <eos>
  
## Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
