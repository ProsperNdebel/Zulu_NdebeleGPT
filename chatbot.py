import argparse
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='This is a demonstration program')

#Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

#Now we can use the argument value in our program.
print(f'batch size: {args.batch_size}')

batch_size = 64
block_size = 128
max_iters = 10
learning_rate = 2e-5
eval_iters = 1
dropout = 0.2
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
import mmap
import random
import torch.nn as nn
from torch.nn import functional as F
import pickle


chars = ""
with open("vocab.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

#tokenizer
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

#estimate loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#the architecture described in the research paper
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #they run in parallel
        self.proj = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embed, 4*n_embed),
                nn.ReLU(),
                nn.Linear(4*n_embed, n_embed),
                nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, index, targets=None):
        
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        
        logits = self.lm_head(x)
        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
        
        
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            #gets the predictions
            index_cond = index[:,-block_size:]
            logits, loss = self.forward(index_cond)
    
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        
        return index
    
model =  GPTLanguageModel(vocab_size)

print('loading model parameters....')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded succesfully')
m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')

    




