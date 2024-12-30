# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: transformer2
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# %% [markdown]
# # Get the training data
#
# We follow Andrej Karpathy's [video on GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) to use shakespeare dataset.

# %%
# Get shakespeare dataset

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# %%
# read the training data

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# %%
print(f"length of training dataset: {len(text)}")
print(f"First 100 characters: \n {text[:100]}")

# %% [markdown]
# # Character level tokenization
#
# For simplicity, we use character level tokenizer. The vocabular size is really small and the tokenization output is long

# %%
uniq_chars = sorted(list(set(text)))
print(f"Unique characters: {''.join(uniq_chars)}")

# %%
# create the mapping between character and integer and vice versa

ctoi = {c:i for i, c in enumerate(uniq_chars)}
itoc = {i:c for i, c in enumerate(uniq_chars)}


# %%
def encode(s: str):
    # convert a string to a list of integers
    return [ctoi[c] for c in s]

def decode(tokens: list[int]):
    # convert a list of integer tokens into a string
    return ''.join([itoc[t] for t in tokens])


# %%
encode('i love you')

# %%
decode(encode('i love you'))

# %% [markdown]
# # Generate training and validation datasets
#
# Split the whole Shakespeare dataset into training and validation dataset

# %%
text_data = torch.tensor(encode(text), dtype=torch.long)
training_size = int(0.9 * len(text))
train_data = text_data[:training_size]
val_data = text_data[training_size:]

# %%
train_data[:100]

# %%
val_data[:100]

# %% [markdown]
# # context length dimension(block size dimension)
#
# During the training, we cannot feed the whole training data text into transformer because that would be prohibitively expensive. We need to sample a `block_size` or `context_length` tokens for each training example

# %%
block_size = 8
# sample the first block
train_data[:block_size+1]

# %% [markdown]
# when sample a `context_length` data, we actually have multiple pairs of inputs and outputs. The input is of different size and output is of size 1

# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]

for i in range(block_size):
    print(f"when the input is {x[:i+1]}, the output is {y[i]}")


# %% [markdown]
# # Batch dimension
#
# During trainig, to maximize the efficiency of GPUs, multiple blocks of data will be processed in parallel. Since it takes almost the same amount of time to process a batch of blocks than a single block if the GPU memory allows.

# %%
# define a function to get a batch of data

def get_batch(split: str, block_size: int, batch_size: int, device=torch.device('cpu')):
    data = train_data if split == 'train' else val_data
    # sample a ramdom offset, between 0 and len(data)-block_size-1(inclusive)
    offsets = torch.randint(len(data)-block_size, (batch_size, ))
    xb = torch.stack([data[i:i+block_size] for i in offsets])
    yb = torch.stack([data[i+1:i+1+block_size] for i in offsets])
    return xb.to(device), yb.to(device)


# %%
torch.manual_seed(1337)

# %%
xb, yb = get_batch('train', block_size=8, batch_size=4)
print(f"input data of shape: {xb.shape}")
print(xb)
print(f"output data of shape: {yb.shape}")
print(yb)

# %%
batch_size = 2
block_size = 8
for b in range(batch_size):
    for i in range(block_size):
        print(f"when the input is {xb[b, :i+1].tolist()}, the output is {yb[b, i]}")


# %% [markdown]
# # Bigram Neutral Network

# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''Given a [B, T] idx, predict the next tokens for max_new_tokens times'''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last one to predict the next token
            logits, _ = self(idx[:, -1].view(-1, 1)) # [B, C] and idx[:, -1] is of shape [B,] not [B, 1]
            probs = F.softmax(logits, dim=-1)
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs.view(B, -1), num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx
            
        

# %%
vocab_size = len(uniq_chars)
embed_size = 128
bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, torch.device('cuda')).to(torch.device('cuda'))

# %%
bigram_model(xb, yb)

# %%
idx = torch.zeros((1, 1), dtype=torch.long, device=torch.device('cuda'))
res = bigram_model.generate(idx, 100)

# %%
res.shape

# %%
# decode
print(decode(res.squeeze().tolist()))

# %% [markdown]
# # Train

# %%
batch_size = 32
block_size = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10


# %%
# tell Pytorch that everying happens in this function will not call backward()
# so no intermediate resutls will be memorized during forward pass which will speed up
# the calculations
@torch.no_grad()
def evaluate(model: nn.Module, eval_iters: int, device=torch.device('cpu')):
    # set to evaluation mode to not have dropout, batch norm and etc
    # that are only used in training
    model.eval()
    out = {}
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for step in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[step] = loss
        out[split] = losses.mean()
    model.train()
    return out    
            


# %%
bigram_model = BigramLanguageModel(vocab_size, embed_size)
bigram_model = bigram_model.to(device)


# %%
def train(model, epochs, block_size, batch_size, device, lr):
    start_time = time.time()
    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=lr)
    for epoch in range(epochs):
        # sample data
        xb, yb = get_batch('train', block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # backpropagation to calculate partial derivatives
        loss.backward()
        # update weights
        optimizer.step()
        
        if (epoch + 1) % eval_interval == 0:
            eval_res = evaluate(bigram_model, eval_iters, device)
            print(f"Epoch: {epoch+1}, the average training loss: {eval_res['train']}, val loss: {eval_res['eval']}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")


# %%
train(bigram_model, epochs, block_size, batch_size, device, lr)


# %%
def generate(model, device, max_len):
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    res = model.generate(idx, max_len)
    return decode(res.squeeze().tolist())


# %%
print(generate(bigram_model, device, 100))

# %%
bigram_model = BigramLanguageModel(vocab_size, embed_size)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)

# %%
print(generate(bigram_model, device, 1000))

# %% [markdown]
# # The math tricks of self-attention
#
# In Bigram Model above, we only use the immediate previous token to predict the next token. Obviously we don't consider the history of the sequence. In Transformer, all elements in the context length history are considered when predicting the next token.
#
# Let's take a step back. What is the easiest way to consider the information in the context length history? Probably just average the embeddings of them.

# %%
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)


# %%
# bag of words means using the average of history
def get_bow(x):
    B, T, C = x.shape
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            prev_x = x[b, :t+1] # t+1, C
            xbow[b, t] = prev_x.mean(dim=0)
    return xbow


# %%
# a more efficient way to get rid of for loop
def get_bow_optim(x):
    T = x.shape[1]
    x = torch.cumsum(x, dim=1) # get cumulative sum along dim 1 which is T dim
    divisors = torch.arange(1, T+1).view(1, T, 1) # [1, T, 1]
    return x / divisors


# %%
B, T, C = 2, 3, 2 # batch, time, channels
x = torch.randn(B, T, C)

# %%
x

# %%
get_bow(x)

# %%
get_bow_optim(x)


# %%
# another way of thinking about how to do this bow is matrix multiplication
# we can use the lower triangle matrix where all elements are 1 to multiply the x matrix to get the cumulative sum
# then we use divisors to get the average

def get_bow_mm(x):
    T = x.shape[1]
    m = torch.tril(torch.ones(T, T))
    m = m / torch.sum(m, dim=1, keepdim=True)
    # x = m.unsqueeze(0) @ x, you don't even need to do unsqueeze
    return m @ x


# %%
get_bow_mm(x)


# %%
# we can also use mask fill and softmax to construct the weights of x
# softmax is more flexible because the weights can be any number instead of just one
def get_bow_mask_fill_and_softmax(x):
    T = x.shape[1]
    # upper triangle not including diagonal 
    mask = torch.triu(torch.ones(T, T), diagonal=1)
    wei = torch.ones((T, T)) # for attention, this could be any weight matrix instead of just one
    wei.masked_fill_(mask == 1, float('-inf'))
    wei = F.softmax(wei, dim=1)
    return wei @ x 


# %%
get_bow_mask_fill_and_softmax(x)


# %% [markdown]
# ## Self-attention
#
# First a single head case

# %%
def self_attention(x, head_dim):
    # x is of shape [B, T, C]
    C = x.shape[2]
    generator = nn.Linear(C, head_dim)
    k = generator(x) # [B, T, head_dim]
    q = generator(x) # [B, T, head_dim]
    v = generator(x) # [B, T, head_dim]
    wei = q @ k.transpose(1, 2) # [B, T, T]
    # upper triangle not including diagonal 
    mask = torch.triu(torch.ones(T, T), diagonal=1)
    wei.masked_fill_(mask == 1, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    print(f"attention weights:\n{wei}")
    return wei @ v


# %%
self_attention(x, 3)


# %% [markdown]
# In the Transformer paper, they use scaled dot product attention. They divide the weight matrix by $\sqrt{d\_model}$. This is to make the variance of the weight not impacted by the $\sqrt{d\_model}$

# %%
def scaled_self_attention(x, head_dim):
    # x is of shape [B, T, C]
    C = x.shape[2]
    key = nn.Linear(C, head_dim, bias=False)
    query = nn.Linear(C, head_dim, bias=False)
    value = nn.Linear(C, head_dim, bias=False)
    k = key(x) # [B, T, head_dim]
    q = query(x) # [B, T, head_dim]
    v = value(x) # [B, T, head_dim]
    # for each dot prodct, it is the sum of head_dim values
    # so divide sqrt(head_dim) to make sure the variance is independent of head_dim
    # this further make sure softmax is not dominated by the max value since that would be one-hot encoding
    wei = q @ k.transpose(1, 2) / math.sqrt(head_dim) # [B, T, T]
    # upper triangle not including diagonal 
    mask = torch.triu(torch.ones(T, T), diagonal=1)
    wei.masked_fill_(mask == 1, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    return wei, wei @ v


# %%
scaled_self_attention(x, 3)

# %% [markdown]
# why this normalization matters? It is to constrain the variance. Why we should constrain the variance? Because if the variance is very large, the largest value in the weight matrix will dominate the distribution after **Softmax**

# %%
C = x.shape[2]
head_dim = 3
key = nn.Linear(C, head_dim, bias=False)
query = nn.Linear(C, head_dim, bias=False)
value = nn.Linear(C, head_dim, bias=False)
k = key(x) # [B, T, head_dim]
q = query(x) # [B, T, head_dim]
v = value(x) # [B, T, head_dim]
wei = q @ k.transpose(1, 2) # [B, T, T]

# %%
k.var().item(), v.var().item(), q.var().item(), wei.var().item()

# %%
C = x.shape[2]
head_dim = 30
key = nn.Linear(C, head_dim, bias=False)
query = nn.Linear(C, head_dim, bias=False)
value = nn.Linear(C, head_dim, bias=False)
k = key(x) # [B, T, head_dim]
q = query(x) # [B, T, head_dim]
v = value(x) # [B, T, head_dim]
wei = q @ k.transpose(1, 2) # [B, T, T]

# %%
k.var().item(), v.var().item(), q.var().item(), wei.var().item()

# %% [markdown]
# you can see that the variance of weight is significantly larger for larger head_dim

# %%
C = x.shape[2]
head_dim = 3
key = nn.Linear(C, head_dim, bias=False)
query = nn.Linear(C, head_dim, bias=False)
value = nn.Linear(C, head_dim, bias=False)
k = key(x) # [B, T, head_dim]
q = query(x) # [B, T, head_dim]
v = value(x) # [B, T, head_dim]
wei = q @ k.transpose(1, 2) / math.sqrt(head_dim) # [B, T, T]
k.var().item(), v.var().item(), q.var().item(), wei.var().item()

# %%
C = x.shape[2]
head_dim = 30
key = nn.Linear(C, head_dim, bias=False)
query = nn.Linear(C, head_dim, bias=False)
value = nn.Linear(C, head_dim, bias=False)
k = key(x) # [B, T, head_dim]
q = query(x) # [B, T, head_dim]
v = value(x) # [B, T, head_dim]
wei = q @ k.transpose(1, 2) / math.sqrt(head_dim) # [B, T, T]
k.var().item(), v.var().item(), q.var().item(), wei.var().item()

# %% [markdown]
# To illustrate the behavior of softmax, we can use the following examples

# %%
torch.softmax(torch.tensor([1, 2, 3, 4]), dim=-1, dtype=torch.float32)

# %%
torch.softmax(torch.tensor([1, 2, 3, 4])*10, dim=-1, dtype=torch.float32)


# %%
class Attention(nn.Module):
    def __init__(self, embed_size, head_size, block_size, device=torch.device('cpu')):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size)
        self.key = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        self.head_size = head_size
        self.register_buffer('tril', torch.triu(torch.ones(block_size, block_size, device=device), diagonal=1))
    
    def forward(self, x):
        # x is of shape [B, T, C]
        T = x.shape[1]
        q = self.query(x) # B, T, head_dim
        k = self.key(x) # B, T, head_dim
        v = self.value(x) # B, T, head_dim
        
        wei = q @ k.transpose(1, 2) / math.sqrt(self.head_size) # B, T, T
        mask = self.tril[:T, :T] # T, T
        wei.masked_fill_(mask == 1, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei, wei @ v


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, head_size, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(head_size, vocab_size)
        self.attention = Attention(embed_size, head_size, block_size, device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        _, x = self.attention(x) # [B, T, head_size]
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
batch_size = 32
block_size = 8
head_size = 64
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

# %%
bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, head_size, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)

# %%
print(generate(bigram_model, device, 100))


# %% [markdown]
# ## Multi-head attention

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        assert embed_size % nheads == 0
        self.heads = nn.ModuleList([Attention(embed_size, embed_size // nheads, block_size, device) for _ in range(nheads)])
    
    def forward(self, x):
        outputs = [head(x)[1] for head in self.heads]
        return torch.concat(outputs, dim=-1)


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.mha = MultiHeadAttention(nheads, embed_size, block_size, device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        x = self.mha(x) # [B, T, C]
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

# %%
bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)


# %% [markdown]
# Multi-head attention achieves smaller training and val loss than single head attention

# %% [markdown]
# # Multi-head Attention with Matrix Multiplication
#
# The multi-head attention using `nn.ModuleList` is an easy implementation. But it uses a for loop to do the attention for each head which is not processed in parallel. We could use matrix multiplication to process all heads in parallel. An example reference could be found in [Harvard's annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

# %%
def attention(q, k, v, mask):
    d_k = q.shape[-1]
    wei = q @ k.transpose(-1, -2) / math.sqrt(d_k) # B, nheads, T, T
    wei.masked_fill_(mask==1, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = wei @ v
    return wei, v


# %%
class MultiHeadAttentionMatrixMultiplication(nn.Module):
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        assert embed_size % nheads == 0
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.nheads = nheads
        self.head_dim = embed_size // nheads
        self.register_buffer('tril', torch.triu(torch.ones(block_size, block_size, device=device), diagonal=1))
        self.attn = None
    
    def forward(self, x):
        # x is B, T, C
        B, T = x.shape[0], x.shape[1]
        q = self.query(x) # B, T, C
        k = self.key(x) # B, T, C
        v = self.value(x) # B, T, C
        q = q.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        k = k.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        v = v.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        q.transpose_(1, 2) # B, nheads, T, head_dim
        k.transpose_(1, 2) # B, nheads, T, head_dim
        v.transpose_(1, 2) # B, nheads, T, head_dim
        mask = self.tril[:T, :T]
        
        self.attn, v = attention(q, k, v, mask) 
        v = v.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.head_dim)
        
        return v


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.mha = MultiHeadAttentionMatrixMultiplication(nheads, embed_size, block_size, device)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        x = self.mha(x) # [B, T, C]
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)


# %% [markdown]
# ## Add a feed forward linear layer
#
# After attention layer, each token gets the connections or understanding among each other. A  feed forward layer is like digesting the aggregation information.

# %%
class Feedforward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.mha = MultiHeadAttentionMatrixMultiplication(nheads, embed_size, block_size, device)
        self.ffw = Feedforward(embed_size)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        x = self.mha(x) # [B, T, C]
        x = self.ffw(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 100))


# %% [markdown]
# # Make a block of Multi-head Attention followed by Feedforward

# %%
class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        self.mha = MultiHeadAttentionMatrixMultiplication(nheads, embed_size, block_size, device)
        self.ffw = Feedforward(embed_size)
    
    def forward(self, x):
        x = self.mha(x)
        x = self.ffw(x)
        return x


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, device) for _ in range(num_blocks)])
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        for block in self.blocks:
            x = block(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
num_blocks = 3
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 100))


# %% [markdown]
# We can see that with multiple blocks, the training and val loss contiue to go down. This is great!

# %% [markdown]
# Since we make the neutral network deeper by having multiple blocks, I also want to introduce two techniques to improve the performance, residual connection and layer normalization

# %% [markdown]
# First following Attention is All You Need, add a linear layer for the output

# %%
class MultiHeadAttentionMatrixMultiplication(nn.Module):
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        assert embed_size % nheads == 0
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)
        self.nheads = nheads
        self.head_dim = embed_size // nheads
        self.register_buffer('tril', torch.triu(torch.ones(block_size, block_size, device=device), diagonal=1))
        self.attn = None
    
    def forward(self, x):
        # x is B, T, C
        B, T = x.shape[0], x.shape[1]
        q = self.query(x) # B, T, C
        k = self.key(x) # B, T, C
        v = self.value(x) # B, T, C
        q = q.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        k = k.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        v = v.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        q.transpose_(1, 2) # B, nheads, T, head_dim
        k.transpose_(1, 2) # B, nheads, T, head_dim
        v.transpose_(1, 2) # B, nheads, T, head_dim
        mask = self.tril[:T, :T]
        
        self.attn, v = attention(q, k, v, mask) 
        v = v.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.head_dim)
        return self.out(v)


# %% [markdown]
# Second, still following the paper to add another linear layer for the feedforward

# %%
class Feedforward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
    
    def forward(self, x):
        return self.net(x)


# %% [markdown]
# Third add residual connection to both attention layer and feedforward layer

# %%
class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        self.mha = MultiHeadAttentionMatrixMultiplication(nheads, embed_size, block_size, device)
        self.ffw = Feedforward(embed_size)
    get_batch
    def forward(self, x):
        x = self.mha(x) + x
        x = self.ffw(x) + x
        return x


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, device) for _ in range(num_blocks)])
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        for block in self.blocks:
            x = block(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
num_blocks = 3
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 500))


# %% [markdown]
# Next, we add layer norm BEFORE the input is feed into attention block and feedforward block

# %%
class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, nheads, embed_size, block_size, device=torch.device('cpu')):
        super().__init__()
        self.mha = MultiHeadAttentionMatrixMultiplication(nheads, embed_size, block_size, device)
        self.ffw = Feedforward(embed_size)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)       
    def forward(self, x):
        x = self.mha(self.layernorm1(x)) + x
        x = self.ffw(self.layernorm2(x)) + x
        return x


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, device) for _ in range(num_blocks)])
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        for block in self.blocks:
            x = block(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
num_blocks = 3
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 500))


# %% [markdown]
# Next add another Layer Norm for the output of the transformer blocks before the final linear layer to projecting to vocab_size

# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, device) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(embed_size)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
num_blocks = 3
batch_size = 32
block_size = 8
nheads = 8
lr = 1e-3
epochs = 50000
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 1000
eval_iters = 10

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 500))


# %% [markdown]
# Next we add dropout to improve the validation loss. We will add dropout in 3 places
# 1. final output of the multi-head attention layer
# 2. final output of feedforward layer
# 3. also dropout to the attention weights

# %%
class FeedforwardWithDropout(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


# %%
def attention(q, k, v, mask, dropout=None):
    d_k = q.shape[-1]
    wei = q @ k.transpose(-1, -2) / math.sqrt(d_k) # B, nheads, T, T
    wei.masked_fill_(mask==1, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    if dropout is not None:
        wei = dropout(wei)
    v = wei @ v
    return wei, v


# %%
class MultiHeadAttentionMatrixMultiplicationWithDropout(nn.Module):
    def __init__(self, nheads, embed_size, block_size, dropout, device=torch.device('cpu')):
        super().__init__()
        assert embed_size % nheads == 0
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)
        self.nheads = nheads
        self.head_dim = embed_size // nheads
        self.register_buffer('tril', torch.triu(torch.ones(block_size, block_size, device=device), diagonal=1))
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x is B, T, C
        B, T = x.shape[0], x.shape[1]
        q = self.query(x) # B, T, C
        k = self.key(x) # B, T, C
        v = self.value(x) # B, T, C
        q = q.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        k = k.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        v = v.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        q.transpose_(1, 2) # B, nheads, T, head_dim
        k.transpose_(1, 2) # B, nheads, T, head_dim
        v.transpose_(1, 2) # B, nheads, T, head_dim
        mask = self.tril[:T, :T]
        
        self.attn, v = attention(q, k, v, mask, self.attn_dropout) 
        v = v.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.head_dim)
        v = self.out(v)
        return self.dropout(v)


# %%
class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, nheads, embed_size, block_size, dropout, device=torch.device('cpu')):
        super().__init__()
        self.mha = MultiHeadAttentionMatrixMultiplicationWithDropout(nheads, embed_size, block_size, dropout, device)
        self.ffw = FeedforwardWithDropout(embed_size, dropout)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)       
    def forward(self, x):
        x = self.mha(self.layernorm1(x)) + x
        x = self.ffw(self.layernorm2(x)) + x
        return x


# %%
class BigramLanguageModel(nn.Module):
    '''
    Bigram LM is used to predict the next token only considering the previous token
    '''
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device=torch.device('cpu')):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, dropout, device) for _ in range(num_blocks)])
        self.layernorm = nn.LayerNorm(embed_size)
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.block_size = block_size
        
    def forward(self, inputs, targets=None):
        # inputs is of shape [B, T] and targets is of shape [B, T]
        # first we use embedding lookup table to transform input from [B, T] to [B, T, embed_size]
        T = inputs.shape[1]
        embeddings = self.lut(inputs) # [B, T, C]
        position_embeddings = self.positional_embedding(torch.arange(T, device=self.device)) # [T, C]
        x = embeddings + position_embeddings # [B, T, C]
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)
        logits = self.generator(x)
        loss = None
        if targets is not None:
            # check pytorch cross entropy loss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            # the x should be of shape [B, C] or [B, C, k1, k2, k3]
            # so we need to make logits from [B, T, embed_size] to [B * T, embed_size]
            # and y should be of shape [B] or [B, k1, k2, k3]
            # so we need to make targets of shape [B * T]
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1) # equivalent to targets.view(B * T)
            loss = self.loss(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Given a [B, T] idx, predict the next tokens for max_new_tokens times
        And this time, we use history tokens up to block_size to predict the next token
        '''
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # only use the last block_size to predict the next token
            logits, _ = self(idx[:, -block_size:]) # [B, T, vocab_size]
            # focus only on the last timestamp
            logits = logits[:, -1] # [B, vocab_size]
            probs = F.softmax(logits, dim=-1) # [B, vocab_size]
            # sample token from probs for each one in the batch
            samples = torch.multinomial(probs, num_samples=1) # [B, 1]
            # concatenate samples into idx
            idx = torch.concat((idx, samples), dim=1) # [B, T+1]
        return idx


# %%
num_blocks = 6
batch_size = 64
block_size = 256
nheads = 8
lr = 3e-4
epochs = 5000
vocab_size = len(uniq_chars)
embed_size = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_interval = 200
eval_iters = 10
dropout = 0.2

bigram_model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device)
bigram_model = bigram_model.to(device)
for param in bigram_model.parameters():
    if param.dim() > 1:
        nn.init.xavier_normal_(param)
train(bigram_model, epochs, block_size, batch_size, device, lr)
print(generate(bigram_model, device, 500))

# %%
