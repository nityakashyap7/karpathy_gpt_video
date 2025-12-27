import torch
import torch.nn.functional as F
import torch.nn as nn


# ****************** begin hyperparameters ******************
train_proportion = 0.9  # proportion of the data that's reserved for training, rest is for validation
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000  # this many steps of gradient descent
eval_interval = 500  # eval every so often to plot loss as model trains
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu" # took 12 mins on complete transfomer
# device = "mps" if torch.backends.mps.is_available() else "cpu" # for mac gpu; 41 mins to get to 3500 steps 💀
eval_iters = 200  # when u do an eval, sample 200 batches so u can report avg loss across those (individual batches can be noisy)
n_embd = 384  # num dims of embedding vectors
# n_head = 6  # how big is the embedding vector coming out of transformer head
n_layer = 6  # number of attention + feedforward blocks in the transformer
dropout_prob = 0.2
# ****************** END hyperparameters ******************


# ****************** BEGIN other global variables ******************
dataset_filename = "input.txt"
torch.manual_seed(1337)  # allows reproducibility
# ****************** END other global variables ******************


# ****************** BEGIN data loading, tokenization, train-val split ******************

with open(dataset_filename, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): 
    return [stoi[c] for c in s]

def decode(lst): 
    return "".join([itos[i] for i in lst])

data = torch.tensor(encode(text), dtype=torch.long)


n = int(
    train_proportion * len(data)
)  # truncate to a whole number bc index can only be an int

train_data = data[0:n]  # first 90% of data will be used to train
val_data = data[n:]  # remaining 10% will be held out for validation

# ****************** END data loading, tokenization, train-val split ******************


# ****************** BEGIN batching and eval loss helper functions ******************


def get_batch(split: str):
    """generate a batch of data of inputs x and targets y. split param can either be 'train' or 'val'"""

    data = train_data if split == "train" else val_data
    idx = torch.randint(
        0, len(data) - block_size, (batch_size,)
    )  # generate batch_size number of random indices to pull snippets from
    x = torch.stack(
        [data[i : i + block_size] for i in idx]
    )  # torch.stack() takes in a list/tuple of tensors
    y = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in idx]
    )  # e.g. for x = snippet[0] y = snippet[1], hence +1
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """evaluating current model on training and validation sets. spits out avg loss for both in a dictionary w 2 keys"""
    metrics = {}
    model.eval()
    losses = torch.zeros(
        eval_iters
    )  # use this instead of list so u can call losses.mean() instead of sum(losses)/len(losses) for a list
    splits = ["train", "val"]
    for split in splits:
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        metrics[split] = losses.mean()

    model.train()  # reset to training mode
    return metrics


# ****************** END batching and eval loss helper functions ******************


# ****************** BEGIN define transformer model ******************
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # for every token in the vocab (characters in our case) there should be a vector representation
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.attention_head = MultiHeadAttn(4, n_embd // 4)
        #self.ff = FeedForward()

        # self.blocks = nn.Sequential(
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        # )
        # better: use n_layer. Sequential takes in variable # of args, create a list comprehension and unpack
        self.blocks = nn.Sequential(*[Block(n_embd, 4) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd) # normalize one last time
        self.linear = nn.Linear(
            n_embd, vocab_size
        )  # map n_head dims to vocab_size dims so that u can run that thru softmax and get a probability distribution

    """ 
    Why pass idx into forward instead of x? Bc embedding params r part of the learned model. But tokenization isnt, so it stays outside. 
    
    forward used in both training and inference (targets=None)
    """

    def forward(self, idx, targets=None):  # x is  idx but not really, y = targets. forward returns loss and LOGITS (not chosen next token)
        token_embeddings = self.token_embedding_table(idx)  # B, T, n_embd
        B, T = idx.shape
        positional_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # takes literal 0, 1, 2, 3, position and maps it to a vector of size n_embd. shape: T, n_embd (bc position doesnt depend on batch, stays same across all examples)
        X = token_embeddings + positional_embeddings  # B, T, n_embd

        # pass X thru attention head
        # Z = self.attention_head(X)

        #Z = self.ff(Z)

        Z = self.blocks(X)
        Z = self.layer_norm(Z)

        logits = self.linear(Z)  # linear layer after transformer; shape: B, T, vocab_size

        if targets is None:
            loss = None

        else:
            # targets has shape: B, T; for each dim you'll have a nested []. for one example itll be [idx, idx, idx], not [[idx], [idx], [idx]]
            # cross_entropy requires second dim to be the softmax dim and B and T to be collapsed into the first dim, so logits and targets need to be reshaped
            targets = targets.view(B * T)  # tensors are immutable
            logits = logits.view(B * T, -1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, num_new_tokens):
        # idx shape: B, T
        for i in range(num_new_tokens):
            context = idx[:, -block_size:]  # the last block_size tokens
            logits, loss = self(context)  # logits shape: B, T, vocab_size

            logits = logits[
                :, -1, :
            ]  # only take the last token's next word logits, leave all other dims alone
            probs = F.softmax(
                logits, dim=-1
            )  # last dim to softmax over vocabulary; shape: B, 1, vocab_size
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # add to time dim (2nd dim)

        return idx


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.W_q = nn.Linear(
            n_embd, head_size, bias=False
        )  # Q = (W_q * X^T)^T -> (B, T, d_attn)
        self.W_k = nn.Linear(
            n_embd, head_size, bias=False
        )  # K = (W_k * X^T)^T -> (B, T, d_attn)
        self.W_v = nn.Linear(
            n_embd, head_size, bias=False
        )  # V = (W_v * X^T)^T -> (B, T, d_attn)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        B, T, C = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scaling_factor = K.size(-1)**0.5

        dot_prods = Q @ K.transpose(-2, -1) / scaling_factor  # B, T, T

        
        dot_prods = dot_prods.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # type: ignore

        weights = F.softmax(dot_prods, dim=-1)  # B, T, T

        weights = self.dropout(weights)

        return weights @ V

class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(n_embd, n_embd) # this is the projection matrix back up to the word-space, will typically go from d < n_embd back up to n_embd
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.linear(out))
    

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # up-projection into fact-space (higher dimension space)
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # down-projection back into word-space (lower dimension space)
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, numheads) -> None:
        super().__init__()
        head_size = n_embd // numheads
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.multi_head_attn = MultiHeadAttn(numheads, head_size)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward()
        

    def forward(self, x):
        out = x + self.multi_head_attn(
                self.layer_norm1(x))
        
        return out + self.ff(
                self.layer_norm2(out))
    


# ****************** END define transformer model ******************


# ****************** BEGIN initialize model and optimizer ******************
print(f'using device: {device}')

model = Transformer()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# ****************** END initialize model and optimizer ******************


# ****************** BEGIN training loop w periodic evals to gauge progress ******************

for step in range(max_iters):
    if (
        step % eval_interval == 0 or step == max_iters - 1
    ):  # check if its time for the periodic eval
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    optimizer.zero_grad(set_to_none=True)  # clear gradients from previous training step

    x, y = get_batch("train")  # sample new batch

    logits, loss = m(x, y)  # forward pass
    loss.backward()  # calculate gradients
    optimizer.step()  # step in the direction of steepest descent

# ****************** END training loop w periodic evals to gauge progress ******************


# ****************** BEGIN generate from final model ******************

context = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # First dimension (1): Batch size - generating 1 sequence at a time. Second dimension (1): Sequence length - starting with just 1 token
print(decode(m.generate(context, num_new_tokens=500)[0].tolist()))

# ****************** END generate from final model ******************
