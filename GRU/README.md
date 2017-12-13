# Project Title
- This is my implementation of Gated Recurrent Units. I have coded GRU from scratch in Tensorflow. Details of the implementation are provided below.
  
## GRU Implementation Details
  
- Let's first discuss the motivation behind GRUs. RNNs can theoretically capture long-term dependencies, however, they are very hard to actually train to do this. Gated recurrent units are designed in a manner to have more persistent memory thereby making it easier for RNNs to capture long-term dependencies. Let us see mathematically how a GRU uses ![](https://latex.codecogs.com/gif.latex?h_%7Bt-1%7D) and ![](https://latex.codecogs.com/gif.latex?x_t) to generate the next hidden state ![](https://latex.codecogs.com/gif.latex?h_t)

## Author
* **Anubhav Gupta**

## Prerequisites
- Python 3.5.4
- Tensorflow 1.2.1
