# improving efficiency (tok/s)
    # using TF32 instead of FP32
    # using mixed precision i.e. mix of BF16 and TF32
    # torch.compile -> kernel, ops fusion + CUDA graph capture + memory planning
    # fusing kernels/optimizing where torch.compile misses e.g. flash attn
    # using "nice" numbers -> large powers of 2 e.g. changing vocab_size from 50257 to 50304
    # kernel fusion in AdamW optimizer


from dataclasses import dataclass
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3) # key query and value projections for all heads, initialized in a batch
        self.c_proj = nn.Linear(self.n_embd, self.n_embd) # output projection after self attn
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # crude way to scale weights of residual layers at init by 1/root(n) where n is num of residual layers (continued below)
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))          # following gpt2 name convention but "bias" is actually a mask 
                                        #.view(1, 1, config.block_size, config.block_size)) # (1, 1, T, T)  # --> replaced with flash attn
                                    

    def forward(self, x):
        B, T, C = x.size() # batch size, seq length, n_embd
        qkv = self.c_attn(x) # (B, T, C * 3)
        q, k, v = qkv.split(self.n_embd, dim=2) # split into query, key and value each has a shape of (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        
        # ensure that queries and keys cannot communicate in future time steps
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # -inf becomes 0 after softmax
        #att = F.softmax(att, dim=-1)
        #y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # kernel fusion of above 4 lines of code (which torch.compile doesn't pick up)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
                                                            # note: .view() only works on contiguous tensors 
                                                            # and transpose makes elements in memory chunk non-contiguous
        y = self.c_proj(y) # output projection

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # the tanh approximate was faster previously and hence implemented in gpt2 
                                                # but now the difference compute time is negligible when using the exact value
                                                # but approx is still used to model gpt2 as closely as possible
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layernorm used before attn block and MLP in gpt2
                                                # unlike model struct in "attention is all you need" paper 
        self.attn = CausalSelfAttention(config) # masked self attn
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # note: gpt2 deviates from "attention is all you need" paper model struct as LN is not in residual pathway
        
        x = x + self.attn(self.ln_1(x)) # communication operation btwn tokens -> it is a aggregation i.e. reduce operation
        x = x + self.mlp(self.ln_2(x)) # indiv token by token operation -> mapping operation
        # the two operations together kind of just makes a transformer a glorified map-reduce

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # vocab size = 50000 BPE merged tokens + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension results in head_size = 768//12 = 64

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict( # index using string
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # using list since indexed using int instead of str in gpt2 model
            ln_f = nn.LayerNorm(config.n_embd), # additional layernorm block (deviating from "attention is all you need" paper)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying scheme -> improves performance 
        self.transformer.wte.weight = self.lm_head.weight   # copies data pointer of wte weight to lm_head weight 
                                                            # this means that std for this tensor will be init twice as 0.02 in _init_weights
                                                            # but bopes

        # init params according to gpt2 code -> norm with 0.02 std and 0.0 as bias
        # note: although typically 1/root(fan_in) is better, 0.02 is surprisingly more or less consistent with that
        # but hardcoding still introduces scaling issues
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 2 x n_layer since both mlp block and attn block contribute to residual pathway
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean= 0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length of size {T}, block size is only {self.config.block_size}"
        # forward token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # make sure positions are in same device as idx
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd) after broadcasting of pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # cross entropy func likes 2d tensors hence tensors have to be flattened
        
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position since loss not being calculated
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type): # loads pretrained gpt2 weights from huggingface
        assert model_type in {'gpt2', 'gpt2-medium','gpt2-large', 'gpt2-xl'}, f"Model must be one of {'gpt2', 'gpt2-medium','gpt2-large', 'gpt2-xl'}"
        from transformers import GPT2LMHeadModel
        print(f"Loading model from: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard all the masks/buffers in self attn

        # initialize a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the params are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # discard all the masks/buffers in self attn not params
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard all the masks/buffers in self attn not params
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad(): # don't track gradients here
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad(): # don't track gradients here
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type): # for weight decay as shown in GPT 3 paper + kernel fusion in optimizer
        # start with all the candidate params which require grad
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups -> weight decaying only 2d params i.e tensors in matmuls and embeddings
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2] # e.g. LN, scales, biases
        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # create AdamW optimizer (use fused ver if avail)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # checks if this ver has fuse is avail in AdamW
        use_fused = fused_available and device_type == 'cuda' # fused implements kernel fusion in AdamW instead of simply iterating over every param
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) # betas and eps set according to GPT 3 paper
        return optimizer

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ------------------------------------------------------------------------------------------------------------------------------------
import tiktoken
import os
import time
import numpy as np
from hellaswag import render_example, iterate_examples

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite: 
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank 
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get shard filenames
        data_root = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, shards) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split: {split}"
        if master_process:
            print(f"Found {len(shards)} shards for {split} split")
        self.reset()      

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # gpu 0 starts at 0, gpu 1 starts at 1(B*T), gpu 2 starts at 2(B*T)...

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] # splice tokens to get batch
        x = buf[:-1].view(B, T) # everything but the last token -> inputs
        y = buf[1:].view(B, T) # everything but the first token -> targets

        self.current_position += B * T * self.num_processes # advance the current position to the next batch

        if self.current_position + (B * T + 1) > len(self.tokens): # advance to next shard if next batch out of bounds 
            self.current_shard = (self.current_shard + 1) % len(self.shards) # % enables us to support training over multiple epochs
            self.tokens = load_tokens(self.shards[self.current_shard]) 
            self.current_position = B * T * self.process_rank

        return x, y

# run training loop 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up distributed data parallel (ddp) --> using multiple gpus in parallel
# requires file to be run by torchrun instead of the normal python "filename"
# torchrun sets up local params RANK, LOCAL_RANK WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) # check if this is a ddp run
ddp =False
if ddp:
    # use of ddp demands CUDA
    assert torch.cuda.is_available(), "CUDA required for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # each gpu will have a rank i.e. gpu 0 will have rank 0 etc -> ensure coordination so that they do not run on the same data
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # only used in multi-node setting -> if you have multiple gpus on each "node" with multiple nodes
                                                    # hence on single node local_rank is the same as rank
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of processes running (equal to number of gpus)
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    # non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

if device.startswith("cuda"):
    device_type = "cuda" 
elif device.startswith("mps"):
    device_type = "mps"
else:
    device_type="cpu"

rng_seed = 1337
torch.manual_seed(rng_seed)
torch.mps.manual_seed(rng_seed)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2^19 -> large power of 2, ~0.5M batch tokens as shown in GPT 3 paper
B = 2 # "micro batch" size
T = 1024 # seq length
assert total_batch_size % (B * T) == 0, "Make sure total_batch_size divisble by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: # ensure that it is only printed once instead of n_gpu times
    print(f"Total desired batch_size: {total_batch_size}")
    print(f"=> calculated grad accum steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')  # 'high' enables TF32 and 'medium' enables BF16 instead of FP32 i.e. 'highest' (default) 
                                            # BF16 (like TF32) reduces mantissa (precision) even further (than TF32) 
                                            # while keeping same exponent (range) as FP32 and TF32
                                            # but do not set to medium, use BF16 using the method below instead

# get our own model
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)    # takes compile time but reduces time taken per step (almost always better to use if possible) 
                                    # optimizes number of "trips" taken to memory through kernel/operator fusion
                                    # -> does not support mps & breaks out sampling loop, hence disabled
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # allows for gradients to be averaged and distributed across the gpus for backward pass
raw_model = model.module if ddp else model

# init lr params for scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 19073   # 10e9 // 2**19 --> tokens in 1 epoch // tokens processed in one step (i.e. totalbatchsize)
warmup_steps = 715  # 375e6// 2**19 since GPT3 paper says that warmup is done over 375e6 tokens, 
                    # divide by no. tokens processed in one step
                    # note: this setup is quite mild, something more aggressive like warmup_steps=100 would be good enough

def get_lr(step): # lr scheduler according to GPT 3 paper
    
    # (1) linear increase in lr until warmup is done
    if step < warmup_steps: 
        return max_lr * ((step + 1)/max_steps) 

    # (2) if step > lr_decay_iters ,return min learning rate
    if step > max_steps:
        return min_lr
    
    # (3) in between cosine decay down to min lr
    decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# creates a "log" directory and "log.txt"
log_dir = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

# optimization + eval + generation (slightly messy)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate= 6e-4, device_type=device_type)
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    # once in a while eval validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad:
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                #with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # mixed precision -> not supported for mps
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"Step: {step} Val_loss: {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # write model checkpoints to load and prevent lost progress during training due to an abrupt stop 
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model' : raw_model.state_dict(),
                        'config' : raw_model.config,
                        'step' : step,
                        'val_loss' : val_loss_accum.item(),
                        'state_dict' : optimizer.state_dict(),
                        'rng_seed' : rng_seed
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank i.e. no two gpus process the same example
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
        
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps): # grad accumulation
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1 # only sync all gpus on the last microstep

        #with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # mixed precision -> not supported for mps
            # note: pytorch documentation says to only wrap this around forward pass and loss calc
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # this is to ensure that loss is normalized (over the number of grad_accum_steps) -> loss = 1/n(a**2 + b**2 + ...)
                                        # as it would be if we called model on the larger batch size directly.
                                        # the problem happens because accum of gradient is a sum in the losses of the minibatches 
                                        # but lacks the normalizing component -> loss = (a**2 + b**2 + ...)
        loss_accum += loss.detach() # loss_accum to print loss later
                                    # .detach() detaches the tensor from the graph 
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # take average and distribute loss on all ranks for printing

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clips mean squared grad of the param length to 1.0 
                                                                    # to reduce effect of something like
                                                                    # super unlucky batch -> high loss -> high gradients -> shocking the model
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if device == "mps": # --> change based on device
        torch.mps.synchronize() 
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) // dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()


