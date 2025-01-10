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
import matplotlib.pyplot as plt
import os

# %%
# %matplotlib inline

# %% [markdown]
# # Get the training data
#
# We follow Andrej Karpathy's [video on GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) to use shakespeare dataset.

# %%
# Get shakespeare dataset
if not os.path.exists("input.txt"):
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# %%
# read the training data

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# %% [markdown]
# # Character level tokenization
#
# For simplicity, we use character level tokenizer. The vocabular size is really small and the tokenization output is long

# %%
uniq_chars = sorted(list(set(text)))
print(f"Unique characters: {''.join(uniq_chars)}")
# create the mapping between character and integer and vice versa

ctoi = {c:i for i, c in enumerate(uniq_chars)}
itoc = {i:c for i, c in enumerate(uniq_chars)}
def encode(s: str):
    # convert a string to a list of integers
    return [ctoi[c] for c in s]

def decode(tokens: list[int]):
    # convert a list of integer tokens into a string
    return ''.join([itoc[t] for t in tokens])


# %%
# sampling tokens
def generate(model, device, max_len, batch_size=1):
    idx = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
    res = model.generate(idx, max_len)
    return res


# %% [markdown]
# # Define the model

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
    if mask is not None:
        wei.masked_fill_(mask==1, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    if dropout is not None:
        wei = dropout(wei)
    v = wei @ v
    return wei, v


# %%
class MultiHeadAttentionMatrixMultiplicationWithDropout(nn.Module):
    def __init__(self, nheads, embed_size, block_size, dropout, device=torch.device('cpu'), use_kv_cache=False):
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
        self.use_kv_cache = use_kv_cache
        self.block_size = block_size
        self.cache_k = None
        self.cache_v = None
        
    def forward(self, x):
        if self.use_kv_cache:
            return self.forward_with_kv_cache(x)
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
        
        self.attn, output = attention(q, k, v, mask, self.attn_dropout) 
        output = output.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.head_dim)
        output = self.out(output)
        return self.dropout(output)
    
    def forward_with_kv_cache(self, x):
        # x is B, T, C
        B, T = x.shape[0], x.shape[1]
        assert T <= self.block_size, (
            "When using KV cache, the total length of the input "
            "should be smaller than the block size"
        )
        # when cache is empty, we should calculate qkv for all tokens
        if self.cache_k is None:
            self.cache_k = self.key(x) # B, T, C
            self.cache_v = self.value(x) # B, T, C
            q_cur = self.query(x) # B, T, C
        else:
            # only do linear projection on the latest token
            q_cur = self.query(x[:, [-1], :]) # B, 1, C
            k_cur = self.key(x[:, [-1], :]) # B, 1, C
            v_cur = self.value(x[:, [-1], :]) # B, 1, C
            self.cache_k = torch.concat((self.cache_k, k_cur), dim=1) # B, T, C
            self.cache_v = torch.concat((self.cache_v, v_cur), dim=1) # B, T, C
        q_cur = q_cur.view(B, -1, self.nheads, self.head_dim) # B, T(or 1), nheads, head_dim
        k = self.cache_k.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        v = self.cache_v.view(B, -1, self.nheads, self.head_dim) # B, T, nheads, head_dim
        q_cur.transpose_(1, 2) # B, nheads, 1, head_dim
        k.transpose_(1, 2) # B, nheads, T, head_dim
        v.transpose_(1, 2) # B, nheads, T, head_dim
        self.attn, out = attention(q_cur, k, v, mask=None)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.nheads * self.head_dim) # B, 1, C
        # since kv cache is only used for inference, we don't have the dropout here
        return self.out(out)
        
        

# %%
class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, nheads, embed_size, block_size, dropout, device=torch.device('cpu'), use_kv_cache=False):
        super().__init__()
        self.mha = MultiHeadAttentionMatrixMultiplicationWithDropout(nheads, embed_size, block_size, dropout, device, use_kv_cache)
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
    def __init__(self, vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device=torch.device('cpu'), use_kv_cache=False):
        super(BigramLanguageModel, self).__init__()
        # the embedding table would be of shape [vocab_size, embed_size]
        self.lut = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(nheads, embed_size, block_size, dropout, device, use_kv_cache) for _ in range(num_blocks)])
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

# %% [markdown]
# # [GTX 3090 Specs](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
#
# * FLOPS: 30T FLOPS(30 * 10^12 flops)
# * Memory bandwidth: 936 GB/s(936 * 10^9 bytes/s)
# * Memory size: 24GB(24 * 10^9 bytes)
#
#

# %% [markdown]
# # Benchmark
#
#

# %%
num_blocks = 4
batch_size = 512
block_size = 1024
nheads = 8
vocab_size = len(uniq_chars)
embed_size = 512
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dropout = 0.2

# %% [markdown]
# Roughtly speaking, for each transformer block
# * The k, v, q weights are of shape [embed_size, embed_size] which takes embed_size * embed_size * 4 bytes(assuming we use float32)
# * The KV cache is 2 * 4 * embed_size * seq_len * batch_size
#   * First 2 is because we have k and v
#   * 4 is because we have 4 bytes for each float32

# %%
embed_sizes = [128, 256, 512, 1024, 2048]


# %%
def benchmark_model(model):
    model = model.to(device)
    # set the model to evaluation mode to disable dropout and increase the speed
    model.eval()
    max_iters = 5
    max_tokens = 1000
    run_times = []
    for i in range(max_iters):
        start_time = time.time()
        output = generate(model, device, max_tokens)
        # let GPU finish the computation before we time the end time
        torch.cuda.synchronize()
        end_time = time.time()
        run_times.append(end_time - start_time)
        print(f"Iteration {i+1} took {end_time - start_time} seconds")
    del bigram_model_with_kv_cache
    return sum(run_times) / max_iters


# %%
def benchmark(embed_sizes, use_kv_cache):
    benchmark_run_times = []
    max_iters = 10
    max_tokens = 400
    for embed_size in embed_sizes:
        run_times = []
        for i in range(max_iters):
            model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device, use_kv_cache=True)
            model = model.to(device)
            model.eval()
            start_time = time.time()
            output = generate(model, device, max_tokens)
            # let GPU finish the computation before we time the end time
            torch.cuda.synchronize()
            end_time = time.time()
            run_times.append(end_time - start_time)
            # print(f"KV Cache: {use_kv_cache}, embed size: {embed_size}, iteration {i+1} took {end_time - start_time} seconds")
            del model
            del output
        print(f"KV Cache: {use_kv_cache}, embed size: {embed_size}, average run time: {sum(run_times) / max_iters}")
        benchmark_run_times.append(sum(run_times) / max_iters)
    return benchmark_run_times


# %%
kv_cache_run_times = benchmark(embed_sizes, use_kv_cache=True)

# %%
no_kv_cache_run_times = benchmark(embed_sizes, use_kv_cache=False)

# %%
plt.plot(embed_sizes, kv_cache_run_times, label='With KV Cache')
plt.plot(embed_sizes, no_kv_cache_run_times, label='Without KV Cache')
plt.xlabel('Embed size')
plt.ylabel('Run time')
plt.legend()
plt.show()

# %%
print(torch.cuda.memory_summary())

# %%
# this will clear the pytorch reserved memory
torch.cuda.empty_cache()


# %%
def benchmark_seq_len(seq_lens, use_kv_cache):
    benchmark_run_times = []
    max_iters = 10
    for seq_len in seq_lens:
        run_times = []
        for i in range(max_iters):
            model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device, use_kv_cache=True)
            model = model.to(device)
            model.eval()
            start_time = time.time()
            output = generate(model, device, seq_len)
            # let GPU finish the computation before we time the end time
            torch.cuda.synchronize()
            end_time = time.time()
            run_times.append(end_time - start_time)
            # print(f"KV Cache: {use_kv_cache}, embed size: {embed_size}, iteration {i+1} took {end_time - start_time} seconds")
            del model
            del output
        print(f"KV Cache: {use_kv_cache}, max_seq_len: {seq_len}, average run time: {sum(run_times) / max_iters}")
        benchmark_run_times.append(sum(run_times) / max_iters)
    return benchmark_run_times


# %%
seq_lens = [50, 100, 200, 400, 600, 800]

# %%
run_times_with_kv_cache = benchmark_seq_len(seq_lens, use_kv_cache=True)

# %%
run_times_without_kv_cache = benchmark_seq_len(seq_lens, use_kv_cache=False)

# %%
plt.plot(seq_lens, run_times_with_kv_cache, label='With KV Cache')
plt.plot(seq_lens, run_times_without_kv_cache, label='Without KV Cache')
plt.xlabel('Max sequence length')
plt.ylabel('Run time')
plt.legend()
plt.show()


# %%
def benchmark_batch_size(batch_sizes, use_kv_cache):
    benchmark_run_times = []
    max_iters = 10
    max_tokens = 200
    for batch_size in batch_sizes:
        run_times = []
        for i in range(max_iters):
            model = BigramLanguageModel(vocab_size, block_size, embed_size, nheads, num_blocks, dropout, device, use_kv_cache=True)
            model = model.to(device)
            model.eval()
            start_time = time.time()
            output = generate(model, device, max_tokens, batch_size)
            # let GPU finish the computation before we time the end time
            torch.cuda.synchronize()
            end_time = time.time()
            run_times.append(end_time - start_time)
            # print(f"KV Cache: {use_kv_cache}, embed size: {embed_size}, iteration {i+1} took {end_time - start_time} seconds")
            del model
            del output
        print(f"KV Cache: {use_kv_cache}, batch size: {batch_size}, average run time: {sum(run_times) / max_iters}")
        benchmark_run_times.append(sum(run_times) / max_iters)
    return benchmark_run_times


# %%
num_blocks = 6
block_size = 256
nheads = 8
vocab_size = len(uniq_chars)
embed_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dropout = 0.2

# %%
batch_sizes = [1, 2, 4, 8, 16, 32, 64]

# %%

run_times_with_kv_cache = benchmark_batch_size(batch_sizes, use_kv_cache=True)


# %%
run_times_without_kv_cache = benchmark_batch_size(batch_sizes, use_kv_cache=False)

# %%

plt.plot(batch_sizes, run_times_with_kv_cache, label='With KV Cache')
plt.plot(batch_sizes, run_times_without_kv_cache, label='Without KV Cache')
plt.xlabel('Batch size')
plt.ylabel('Run time')
plt.legend()
plt.show()
