# Gated Recurrent Unit (GRU) for Language Modeling
- Here, I have implemented GRU from scratch in Tensorflow for language modeling task.
  
## GRU Implementation Details
  
 - Let's first discuss the motivation behind GRUs. RNNs can theoretically capture long-term dependencies, however, they are very hard to actually train to do this. Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies. Let us see mathematically how a GRU uses ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) and ![](https://latex.codecogs.com/gif.latex?x_t) to generate the next hidden state ![](https://latex.codecogs.com/gif.latex?h_t).
 
 - The model is, for t = 1, ..., n-1:
 
      ![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Clarge%20%5Cbegin%7Balign*%7D%20z_t%20%26%3D%20%5Csigma%28W%5E%7B%28z%29%7Dx_t%20&plus;%20U%5E%7B%28z%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2.15cm%7D%5Ctextbf%7B%28Update%20gate%29%7D%5C%5C%20r_t%20%26%3D%20%5Csigma%28W%5E%7B%28r%29%7Dx_t%20&plus;%20U%5E%7B%28r%29%7Dh_%7Bt-1%7D%29%20%5Chspace%7B2.15cm%7D%5Ctextbf%7B%28Reset%20gate%29%7D%5C%5C%20%5Ctilde%7Bh%7D_t%20%26%3D%20%5Ctanh%28W%20x_t%20&plus;%20r_t%20%5Codot%20U%20h_%7Bt-1%7D%29%20%5Chspace%7B1.4cm%7D%5Ctextbf%7B%28New%20memory%29%7D%5C%5C%20h_t%20%26%3D%20%281-z_t%29%5Codot%20%5Ctilde%7Bh%7D_t%20&plus;%20z_t%20%5Codot%20h_%7Bt-1%7D%20%5Chspace%7B1.43cm%7D%5Ctextbf%7B%28Hidden%20state%29%7D%5C%5C%20%5Cend%7Balign%7D)
        
## Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
