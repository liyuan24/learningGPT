# Learning GPT
This is a step by step implementation of GPT model. Mostly I followed Andrej Karpathy's [Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY). The hardware I use to train the model is one GTX 3090 and each training could be finished in less than 10mins.

## Obtain the Jupyter Notebook
Please run `make notebook` to get the notebook from the `gpt_dev.py`. We use python file for better code change history tracking. 

Under the hood, we use [jupytext](https://github.com/mwouts/jupytext) to convert between `py` and `ipynb`. I learned this from the [annotated-transformer](https://github.com/harvardnlp/annotated-transformer) repo.

## Training Data
[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. Every trainig example is a random sample of `block_size` length of text from the training dataset. For each step, we group `batch_size` examples.

The number of tokens(or characters in this implementation) is about 1 million.

## Tokenization
For simplicity, in this implementation, we use character level tokenization. It is very simple. Just obtain the unique characters from the training dataset which constitue the vocabulary. The vocabulary size is 65.

GPT2 uses Byte Pair Encoding(BPE) tokenization whose vocabulary size is about 50,000. For the implementation, please check the [minBPE repo](https://github.com/liyuan24/minbpe).

## Step by step improvement of the model

I breakdown the improvements and hopefully it will make the code change easier to udnerstand. 

## Bigram Model
We started with a very simple model, Bigram model. It only has two layers, an embedding layer and a final linear layer to project to the probability for each token in the vocabulary set. And it only uses the information of the immediate preceeding token to predict the next token.

## Bag of Words
In Bigram, the information of previous token is used to predict the next token. In Bag of word, the average of the previous tokens in the `block_size` history is used to predict the next token.

## Self-attention
For self-attention, the weighted average is used to predict the next token. The weight is calculated by the dot product

## Multi-head attention
Each head is a standalone attention transformation. This could further improve the model performance with lower training and validation loss.

The naive implementation is just use a `ModuleList` to chain all heads together and use a `for` loop to go through each head. But since we use for loop, the processing of each head cannot be parallelized. In the next section, we will use batch matrix multiplication to speed up the process. 

### Batch Matrix Multiplication Multi-head attention
We could add another `head` dimension to the batch matrix and implement the multi-head attention with matrix multiplication tricks. From my experiment, with `8` heads, this could improve the training speed by about `4x`.

### Add a Feedforward layer after the Multi-head attention

The attention can be seen as a communication mechanism so that each token could understand the connections across different tokens. Adding a feedforward layer can be seen as learning to digest those information. This will further improve the model performance.

### Multiple blocks of multi-head attention and feedforward

Each block is multi-head attention + feedforward. Mutiple blocks stacking together could further improve the model performance.

### Residual connection and Layer Norm
Deeper neural network makes it harder to train. It could suffer the vanishing or exploding gradients. To tackle those problems, we introduce [residual connection](https://arxiv.org/abs/1512.03385) and [layer normalization](https://arxiv.org/abs/1607.06450).

### Adding Dropout
Finally to mitigate overfitting for the deep neural network, we introduce [dropout](https://arxiv.org/abs/1207.0580) as a regularization.

Dropout is added to 3 places:
1. After the feedforward layer
2. After the multi-head attention layer
3. To the attention weights

## A more advanced implementation

A more advance implementation can be found in [nanoGPT repo](https://github.com/liyuan24/nanoGPT/blob/master/model.py) and I added my comments to `model.py`.

A few changes to highlight here:
1. weight initialization: 
    - for all weight except the bias for the linear layers, it uses normal distribution with mean 0 and std 0.02. For bias, it uses 0.
    - for the final linear layer of feedforward and multi-head attention, it will scale the standard deviation by `1/sqrt(2*n_blocks)`. Why? Because this projetion will be added by the input, if you assume that the input of the linear layer has standard deviation 1, after all blocks, with this scaling, the output will have standard deviation 1. This is a good property. 
2. `device` can be obtained from the input tensor, so no need to explicitly specify as a parameter
3. `dropout` is also added after embedding layer
4. AdamW optimizer: 
    - it uses weight decay for parameters with more than 1 dimension. And no weight decay for 1 dimension parameters.
    - it uses `fused` kernel if available.
5. Feedforward layer:
    - it uses `gelu` activation function.
6. generate function
    - It includes `top_k` sampling.
    - It uses `temperature` to further control the randomness of the sampling.
7. weight tying:
    - it uses `tie_weights` to tie the weights of the embedding layer and the final linear layer
    - you can uanderstand it in this way. For embedding layer, similar token have similar embedding. And in final linear layer to get the logits, let's say we have 1 token representation in `embed_dim` length. To get the logit of first output token, we will use the first row of the weight matrix of the final linear layer. And to get the logit of the second output token, we will use the second row of the weight matrix of the final linear layer. In the same way, here the two rows are two transformations. To get the simialr probability for two similar output tokens, the transformation should be similar. That is why we could share the weight between the embedding layer and the final linear layer.
8. [Flash Attention](https://arxiv.org/abs/2205.14135): it makes the attention much faster.
