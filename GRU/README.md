# Gated Recurrent Unit (GRU) for Language Modeling
- Here, I have implemented GRU from scratch in Tensorflow for language modeling task.
  
## GRU Implementation Details
  
 - Let's first discuss the motivation behind GRUs. RNNs can theoretically capture long-term dependencies, however, they are very hard to actually train to do this. Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies. Let us see mathematically how a GRU uses ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) and ![](https://latex.codecogs.com/gif.latex?x_t) to generate the next hidden state ![](https://latex.codecogs.com/gif.latex?h_t).
 
 - The model is, for t = 1, ..., n-1:
 
      ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Balign*%7D%20z_t%20%26%3D%20%5Csigma%28W%5E%7B%28z%29%7Dx_t%20&plus;%20U%5E%7B%28z%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2cm%7D%5Ctextbf%7B%28Update%20gate%29%7D%5C%5C%20r_t%20%26%3D%20%5Csigma%28W%5E%7B%28r%29%7Dx_t%20&plus;%20U%5E%7B%28r%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2cm%7D%5Ctextbf%7B%28Reset%20gate%29%7D%5C%5C%20%5Ctilde%7Bh%7D_t%20%26%3D%20%5Ctanh%28W%20x_t%20&plus;%20r_t%20%5Codot%20U%20h_%7Bt-1%7D%29%20%5Chspace%7B1.4cm%7D%5Ctextbf%7B%28New%20memory%29%7D%5C%5C%20h_t%20%26%3D%20%281-z_t%29%5Codot%20%5Ctilde%7Bh%7D_t%20&plus;%20z_t%20%5Codot%20h_%7Bt-1%7D%20%5Chspace%7B1.43cm%7D%5Ctextbf%7B%28Hidden%20state%29%7D%5C%5C%20%5Cend%7Balign%7D)
      
 - The above equations can be thought of a GRU's four fundamental operational stages and they have intuitive interpretations that make this model much more intellectually satisfying:
 -- **New memory generation:** A new memory ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t) is the consolidation of a new input word ![](https://latex.codecogs.com/gif.latex?x_t) with the past hidden state ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D). This stage is the one who knows the recipe of combining a newly observed word with the past hidden state ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) to summarize this new word in light of the contextual past as the vector ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t).
 
 -- **Reset Gate:** The reset signal ![](https://latex.codecogs.com/gif.latex?r_t) is responsible for determining how important ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) is to the summarization ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Ctilde%7Bh%7D_t). The reset gate has the ability to completely diminish past hidden state if it finds that ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) is irrelevant to the computation of the new memory.
 
3. Update Gate: The update signal zt
is responsible for determining
how much of ht−1 should be carried forward to the next state. For
instance, if zt ≈ 1, then ht−1 is almost entirely copied out to ht
.
Conversely, if zt ≈ 0, then mostly the new memory ˜ht
is forwarded
to the next hidden state.
4. Hidden state: The hidden state ht
is finally generated using the
past hidden input ht−1 and the new memory generated ˜ht with the
advice of the update gate.

## Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
