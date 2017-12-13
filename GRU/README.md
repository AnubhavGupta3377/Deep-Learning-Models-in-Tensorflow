# Gated Recurrent Unit (GRU) for Language Modeling
- Here, I have implemented GRU from scratch in Tensorflow for language modeling task.
  
## GRU Details
  
 - Let's first discuss the motivation behind GRUs. RNNs can theoretically capture long-term dependencies, however, they are very hard to actually train to do this. Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies. Let us see mathematically how a GRU uses ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) and ![](https://latex.codecogs.com/gif.latex?x_t) to generate the next hidden state ![](https://latex.codecogs.com/gif.latex?h_t).
 
 - The model is, for t = 1, ..., n-1:
 
      ![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20z_t%20%26%3D%20%5Csigma%28W%5E%7B%28z%29%7Dx_t%20&plus;%20U%5E%7B%28z%29%7Dh_%7Bt-1%7D%20&plus;%20b_1%29%20%5Chspace%7B2cm%7D%5Ctextbf%7B%28Update%20gate%29%7D%5C%5C%20r_t%20%26%3D%20%5Csigma%28W%5E%7B%28r%29%7Dx_t%20&plus;%20U%5E%7B%28r%29%7Dh_%7Bt-1%7D%20&plus;%20b_2%29%20%5Chspace%7B2cm%7D%5Ctextbf%7B%28Reset%20gate%29%7D%5C%5C%20%5Ctilde%7Bh%7D_t%20%26%3D%20%5Ctanh%28W%20x_t%20&plus;%20r_t%20%5Codot%20U%20h_%7Bt-1%7D%20&plus;%20b_3%29%20%5Chspace%7B1.4cm%7D%5Ctextbf%7B%28New%20memory%29%7D%5C%5C%20h_t%20%26%3D%20%281-z_t%29%5Codot%20%5Ctilde%7Bh%7D_t%20&plus;%20z_t%20%5Codot%20h_t-1%20%5Chspace%7B1.9cm%7D%5Ctextbf%7B%28Hideen%20state%29%7D%5C%5C%20%5Chat%7By%7D_t%20%26%3D%20%5Ctext%7Bsoftmax%7D%28h_tU%20&plus;%20b_4%29%5C%5C%20%5Cbar%7BP%7D%28x_%7Bt&plus;1%7D%20%3D%20v_j%20%7C%20x_t%2C%5Ccdot%2Cx_1%29%20%26%3D%20%5Chat%7By%7D_t%20%5Cend%7Balign%7D)
      
 - The above equations can be thought of a GRU's four fundamental operational stages and they have intuitive interpretations that make this model much more intellectually satisfying. The four stages can be thought of as performing the following operations:
 
     - **New memory:** A new memory ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t) is the consolidation of a new input word ![](https://latex.codecogs.com/gif.latex?x_t) with the past hidden state ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D). This stage is the one who knows the recipe of combining a newly observed word with the past hidden state ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) to summarize this new word in light of the contextual past as the vector ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t).
  
     - **Reset gate:** The reset signal ![](https://latex.codecogs.com/gif.latex?r_t) is responsible for determining how important ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) is to the summarization ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t). The reset gate has the ability to completely diminish past hidden state if it finds that ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) is irrelevant to the computation of the new memory.
  
     - **Update gate:** The update signal ![](https://latex.codecogs.com/gif.latex?z_t) is responsible for determining how much of ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) should be carried forward to the next state. For instance, if ![](https://latex.codecogs.com/gif.latex?z_t%20%5Capprox%201), then ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) is almost entirely copied out to ![](https://latex.codecogs.com/gif.latex?h_t). Conversely, if ![](https://latex.codecogs.com/gif.latex?z_t%20%5Capprox%200), then mostly the new memory ![](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bh%7D_t) is forwarded to the next hidden state.
     - **Hidden state:** The hidden state ![](https://latex.codecogs.com/gif.latex?h_t) is finally generated using the past hidden input ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) and the new memory generated ![](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bh%7D_%7Bt%7D) with the advice of the update gate.

- The output vector ![](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%5E%7B%28t%29%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%7CV%7C%7D) is a probability distribution over the vocabulary, and we optimize the (unregularized) cross-entropy loss:

	![](https://latex.codecogs.com/gif.latex?J%5E%7B%28t%29%7D%28%5Ctheta%29%20%3D%20%5Ctext%7BCE%7D%28y%5E%7B%28t%29%7D%2C%20%5Chat%7By%7D%5E%7B%28t%29%7D%29%20%3D%20-%5Csum_%7Bi%3D1%7D%5E%7B%7CV%7C%7Dy%5E%7B%28t%29%7D_i%20%5Ctext%7Blog%7D%7E%5Chat%7By%7D%5E%7B%28t%29%7D_i)
	
	where ![](https://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D) is the one-hot vector corresponding to the target word (which here is equal to ![](https://latex.codecogs.com/gif.latex?x_%7Bt&plus;1%7D).

- Results
  - Some Example sentences generated using the GRU models trained for 16 epochs:
  	1. how to invest N N to N such the government <unk> <eos>
	2. in palo alto eyes came as more <eos>
	3. is english compensation service inc alone taken me for an <unk> to seek big ownership they can stop the <unk> which issued a resulting deal with the ages in oat soft which had a damage for this life to leading africa and eventually to increase issues more than had about a month earlier floor <eos>
	4. india is exceptionally long seriously convenience green investment only tourists on lead to chevron he pay nov. N N <eos>
  
## Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
