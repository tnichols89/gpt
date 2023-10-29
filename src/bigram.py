from typing import Callable

import torch
from torch import tensor
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparams
batch_size: int = 32
block_size: int = 8
max_iters: int = 3000
eval_interval: int = 300
learning_rate: float = 1e-2
eval_iters: int = 200

device: torch.device = torch.device('cpu')
if torch.backends.mps.is_available():
  device = torch.device('mps')
elif torch.cuda.is_available():
  device = torch.device('cuda')
# ---

torch.manual_seed(1337)

text: str = None
with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars: list[str] = sorted(list(set(text)))
vocab_size: int = len(chars)

stoi: dict[str, int] = { ch: i for i, ch in enumerate(chars) }
itos: dict[int, str] = { i: ch for i, ch in enumerate(chars) }
encode: Callable[[str], list[int]] = lambda s: [stoi[c] for c in s]
decode: Callable[[int], str] = lambda l: ''.join([itos[i] for i in l])

data: tensor = tensor(encode(text), dtype=torch.long)
n: int = int(0.9 * len(data))
train_data: tensor = data[:n]
val_data: tensor = data[n:]

def get_batch(split: str) -> tuple[tensor, tensor]:
  data: tensor = train_data if split == 'train' else val_data

  # Pick `batch_size` number of random offsets into the dataset
  offsets: tensor = torch.randint(len(data) - block_size, (batch_size,))
  x: tensor = torch.stack([data[i: i + block_size] for i in offsets])
  y: tensor = torch.stack([data[i + 1: i + block_size + 1] for i in offsets])

  return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict[str, tensor]:
  # Averages loss over multiple batches to reduce noise in reported
  # loss for validation and training sets

  out: dict[str, tensor] = {}
  model.eval()

  for split in ['train', 'val']:
    losses: tensor = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    
    out[split] = losses.mean()

  model.train()
  return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size: int) -> None:
    super().__init__()
    
    # Each token directly reads off the logits for the next token
    # from a lookup table.
    #
    # For example, given input index 25, we will retrieve row
    # 25 from this table, and that row will contain `vocab_size`
    # scores, one score corresponding to each possible subsequent
    # token.
    #
    # From there we can basically just softmax that entire row
    # to get a probability distribution over the likelihood of
    # seeing any given token appear next.
    self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx: tensor, targets: tensor = None) -> tuple[tensor]:
    # Let:
    # - B: batch size [4]
    # - T: maximum sequence/block length [8]
    # - C: embedding size/dimensionality [65]

    # idx and targets both have shape (B, T)
    logits: tensor = self.token_embeddings(idx) # (B, T) => (B, T, C)

    loss: tensor = None
    if targets != None:
      # PyTorch cross entropy wants (B, C) shape, so we reshape our
      # tensors to accommodate
      #
      # TODO
      # Try just logits.transpose(1, 2): (B, T, C) => (B, C, T) with
      # targets: (B, T) where each cell in `targets` contains the index
      # of the correct class.
      B, T, C = logits.size()
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx: tensor, max_new_tokens: int) -> tensor:
    # idx: (B, T) tensor of indices in the current context
    for _ in range(max_new_tokens):
      # get predictions
      logits, loss = self(idx)

      # focus only on the last token in the sequence since this is
      # for a bigram model - given a single token, we're trying to
      # predict the next token, thereby generating a bigram.
      #
      # note that this is computationally sort of absurd since this
      # generate function consumes the entire history of generated
      # tokens so far but only ever uses the last generated token.
      #
      # this total history of generated tokens will become useful
      # when we implement a transformer that can attend to all
      # prior generated tokens so we will keep it this way, but
      # just note that it's silly for this particular bigram model.
      logits = logits[:, -1, :] # (B, T, C) => (B, C)

      # convert to probability distribution
      probs = F.softmax(logits, dim=-1) # (B, C) => (B, C)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T) => (B, T+1)

    return idx

model: BigramLanguageModel = BigramLanguageModel(vocab_size)
model = model.to(device)

# LR is often 3e-4 for bigger networks but smaller networks like this one
# are fine with much bigger gradient updates like 1e-3
optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in tqdm(range(max_iters), desc='training'):
  if iter % eval_interval == 0:
    losses: tensor = estimate_loss(model)
    print(f'step {iter}; train loss {losses["train"] :.4f}, val loss {losses["val"] :.4f}')

  xb, yb = get_batch('train')

  optimizer.zero_grad(set_to_none=True)
  logits, loss = model(xb, yb)
  loss.backward()
  optimizer.step()

context: tensor = torch.zeros([1, 1], dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))