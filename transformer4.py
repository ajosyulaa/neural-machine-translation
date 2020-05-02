import torch, pickle, os, sys, random, time
import torchtext
from torchtext.data import Field, Iterator, TabularDataset
import spacy
import math
from torch import nn, optim
from torch.autograd import Variable
import copy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import csv
with open("train.de", "r") as ins:
    train_de = []
    for line in ins:
        train_de.append(line.strip('\n'))

with open("train.en", "r") as ins:
    train_en = []
    for line in ins:
        train_en.append(line.strip('\n'))

with open("dev.de", "r") as ins:
    dev_de = []
    for line in ins:
        dev_de.append(line.strip('\n'))

with open("dev.en", "r") as ins:
    dev_en = []
    for line in ins:
        dev_en.append(line.strip('\n'))

import pandas as pd
train_data = {'German' : [line for line in train_de], 'English': [line for line in train_en]}
df_train = pd.DataFrame(train_data, columns=["German", "English"])

dev_data = {'German' : [line for line in dev_de], 'English': [line for line in dev_en]}
df_dev = pd.DataFrame(dev_data, columns=["German", "English"])

df_train.to_csv("train.csv", index=False)
df_dev.to_csv("val.csv", index=False)

class PositionalEncoding(nn.Module):
  def __init__(self,params):
    super(PositionalEncoding,self).__init__()
    self.d_model = params['d_model']
    self.seq_len = params['max_len']
    self.dropout = nn.Dropout(p = params['dropout'])
    pe = torch.zeros(self.seq_len,self.d_model)
    for pos in range(self.seq_len):
      for i in range(0,self.d_model,2):
        dr = 10000 ** (i/self.d_model)
        pe[pos, i] = math.sin(pos/dr)
        pe[pos, i + 1] = math.cos(pos/dr)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe',pe)
  
  def forward(self, inp):
    inp = inp * math.sqrt(self.d_model)
    inp = inp + Variable(self.pe[:, :inp.size(1),:], requires_grad = False).cuda()
    return self.dropout(inp)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim = -1, keepdim = True)
        x_std = x.std(dim = -1, keepdim = True)
        x_norm = self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias
        return x_norm

class AttentionSubLayer(nn.Module):
  def __init__(self,params):
    super(AttentionSubLayer, self).__init__()
    self.d_model = params['d_model']
    self.nheads = params['nheads']
    self.d_h = self.d_model // self.nheads
    self.query = nn.Linear(self.d_model, self.d_model)
    self.key = nn.Linear(self.d_model, self.d_model)
    self.value = nn.Linear(self.d_model, self.d_model)
    self.W = nn.Linear(self.d_model, self.d_model)
    self.dropout = nn.Dropout(params['dropout'])

  def forward(self,q,k,v,mask):
    '''
    q : bsz x seq_len2 x d_model
    k : bsz x seq_len1 x d_model
    v : bsz x seq_len1 x d_model
    '''
    bsz,seq_len2,seq_len1 = q.size()[0],q.size()[1],k.size()[1]
    queries = self.query(q).view(bsz,seq_len2,self.nheads,self.d_h).transpose(1,2) # bsz x 8 x seq_len2 x 64
    keys = self.key(k).view(bsz,seq_len1,self.nheads,self.d_h).transpose(1,2) # bsz x 8 x seq_len1 x 64
    values = self.value(v).view(bsz,seq_len1,self.nheads,self.d_h).transpose(1,2) # bsz x 8 x seq_len1 x 64
    concat_hidden = self.dot_product_attention(queries,keys,values,mask).transpose(1,2).contiguous().view(bsz,seq_len2,self.d_model) # bsz x seq_len2 x d_model
    attention_out = self.W(concat_hidden) # bsz x seq_len2 x d_hid
    return self.dropout(attention_out)
  
  def dot_product_attention(self,q,k,v,mask):
    '''
    Dimensions :
    q : bsz x n_heads x seq_len2 x d_h
    k : bsz x n_heads x seq_len1 x d_h
    v : bsz x n_heads x seq_len1 x d_h
    mask : bsz x seq_len
    '''
    # d = torch.tensor(q.size()[3],dtype = torch.float)
    d = q.size(-1)
    scores = torch.matmul(q,k.transpose(2,3))/math.sqrt(d) # bsz x n_heads x seq_len2 x seq_len1
#     mask = torch.ones(scores.size(),dtype = torch.uint8).tril()
    scores = scores.masked_fill(mask == 0, -1e9)
    # scores[~mask] = float('-inf')
    scores = nn.functional.softmax(scores, dim = -1) # bsz x n_heads x seq_len2 x seq_len1
#     print(scores[0,0,:,:])
    return torch.matmul(scores, v) # bsz x n_heads x seq_len2 x d_h

class FeedForwardSubLayer(nn.Module):
  def __init__(self,params):
    super(FeedForwardSubLayer, self).__init__()
    self.d_model = params['d_model']
    self.d_ff = params['d_ff']
    self.ff1 = nn.Linear(self.d_model,self.d_ff)
    self.ff2 = nn.Linear(self.d_ff,self.d_model)
    self.relu = nn.ReLU()
    self.dropout1 = nn.Dropout(params['dropout'])
    self.dropout2 = nn.Dropout(params['dropout'])
    
  def forward(self,inp):
    return self.dropout2(self.ff2(self.dropout1(self.relu(self.ff1(inp)))))

class EncoderLayer(nn.Module):
  def __init__(self, params):
    super(EncoderLayer, self).__init__()
    self.d_model = params['d_model']
    self.attention = AttentionSubLayer(params)
    self.feedforward = FeedForwardSubLayer(params)
    self.layernorm_attention = LayerNorm(self.d_model)
    self.layernorm_ff = LayerNorm(self.d_model)
  
  def forward(self, inp, mask):
    layernorm_attention_out = self.layernorm_attention(inp)
    attention_out = inp + self.attention(layernorm_attention_out, layernorm_attention_out, layernorm_attention_out, mask)
    layer_out = attention_out + self.feedforward(self.layernorm_ff(attention_out))
    return layer_out

class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.nlayers = params['nlayers']
    self.layers = [copy.deepcopy(EncoderLayer(params)) for l in range(self.nlayers)]
    self.layers = nn.ModuleList(self.layers)
    self.norm = LayerNorm(params['d_model'])
  
  def forward(self, inp, mask):
    temp = inp
    for layer in self.layers:
      temp = layer(temp, mask)
    return self.norm(temp)

class DecoderLayer(nn.Module):
  def __init__(self, params):
    super(DecoderLayer, self).__init__()
    self.d_model = params['d_model']
    self.masked_attention = AttentionSubLayer(params)
    self.attention = AttentionSubLayer(params)
    self.feedforward = FeedForwardSubLayer(params)
    self.layernorm_masked_attention = LayerNorm(self.d_model)
    self.layernorm_attention = LayerNorm(self.d_model)
    self.layernorm_ff = LayerNorm(self.d_model)
    
  def forward(self, inp, encoder_out, mask1, mask2):
    layernorm_masked_attention_out = self.layernorm_masked_attention(inp)
    masked_attention_out = inp + self.attention(layernorm_masked_attention_out, layernorm_masked_attention_out, layernorm_masked_attention_out, mask1)
    layernorm_attention_out = self.layernorm_attention(masked_attention_out)
    attention_out = masked_attention_out + self.attention(layernorm_attention_out, encoder_out, encoder_out, mask2)
    layer_out = attention_out + self.feedforward(self.layernorm_ff(attention_out))
    return layer_out

class Decoder(nn.Module):
  def __init__(self, params):
    super(Decoder, self).__init__()
    self.nlayers = params['nlayers']
    self.layers = [copy.deepcopy(DecoderLayer(params)) for l in range(self.nlayers)]
    self.layers = nn.ModuleList(self.layers)
    self.norm = LayerNorm(params['d_model'])
  
  def forward(self, inp, encoder_out, mask1, mask2):
    temp = inp
    for layer in self.layers:
      temp = layer(temp, encoder_out, mask1, mask2)
    return self.norm(temp)

class Transformer(nn.Module):
  def __init__(self,params):
    super(Transformer, self).__init__()
    self.d_model = params['d_model']
    self.nheads = params['nheads']
    self.tgt_vocab_size = params['tgt_vocab_size']
    self.src_vocab_size = params['src_vocab_size']
    self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
    self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model)
    self.positional_encoder = PositionalEncoding(params)
    self.encoder = Encoder(params)
    self.decoder = Decoder(params)
    self.output = nn.Linear(self.d_model, self.tgt_vocab_size)
    
  def forward(self, src_inp, tgt_inp, pad_idx):
    '''
    src_inp : bsz x seq_len1
    tgt_inp : bsz x seq_len2
    '''
    bsz, seq_len1, seq_len2 = src_inp.size()[0],src_inp.size()[1], tgt_inp.size()[1]
    src_mask, tgt_mask = self.create_masks(src_inp, tgt_inp, pad_idx)
    src_mask, tgt_mask = src_mask.unsqueeze(1), tgt_mask.unsqueeze(1)
#     src_mask_2 = (src_inp != pad_idx).unsqueeze(-2).unsqueeze(-2) # bsz x 1 x 1 x seq_len1
#     tgt_pad_mask = (tgt_inp != pad_idx).unsqueeze(-2) # bsz x 1 x seq_len2
#     tgt_future_mask = torch.ones((seq_len2,seq_len2),dtype = torch.uint8).tril().cuda().unsqueeze(0) # 1 x seq_len2 x seq_len2
#     tgt_mask_2 = (tgt_pad_mask & tgt_future_mask).unsqueeze(1) # bsz x nheads x seq_len2 x seq_len2
    src_embedding = self.src_embedding(src_inp)
    src_input = self.positional_encoder(src_embedding)
    encoder_out = self.encoder(src_input, src_mask)
    tgt_embedding = self.tgt_embedding(tgt_inp)
    tgt_input = self.positional_encoder(tgt_embedding)
    decoder_out = self.decoder(tgt_input, encoder_out, tgt_mask, src_mask)
    transformer_output = self.output(decoder_out)
    return transformer_output
  
  def create_masks(self, src, trg, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(-2) # bsz x 1 x seq_len1
    trg_mask = (trg != pad_idx).unsqueeze(-2) # bsz x 1 x seq_len2
    size = trg.size(1) # get seq_len for matrix
    np_mask = self.nopeak_mask(size, pad_idx).cuda() # 1 x seq_len2 x seq_len2
    trg_mask = trg_mask & np_mask # bsz x seq_len2 x seq_len2
    return src_mask, trg_mask
  
  def nopeak_mask(self, size, pad_idx):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    return np_mask

 en = spacy.load('en')
de = spacy.load('de')

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

def tokenize_de(sentence):
    return [tok.text for tok in de.tokenizer(sentence)]

DE_TEXT = Field(tokenize=tokenize_de)
EN_TEXT = Field(tokenize=tokenize_en, init_token = "<bos>", eos_token = "<eos>")

# associate the text in the 'English' column with the EN_TEXT field, # and 'German' with DE_TEXT
data_fields = [('German', DE_TEXT),('English', EN_TEXT)]
train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)

MIN_FREQ = 1
DE_TEXT.build_vocab(train, min_freq = MIN_FREQ)
EN_TEXT.build_vocab(train, min_freq = MIN_FREQ)
pad_idx = EN_TEXT.vocab.stoi['<pad>']

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.German))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.English) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in torchtext.data.batch(d, self.batch_size * 100):
                    p_batch = torchtext.data.batch(sorted(p, key=self.sort_key),self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

BATCH_SIZE = 4096
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device = device,
                        repeat=False, sort_key= lambda x:
                        (len(x.German), len(x.English)),
                        batch_size_fn=batch_size_fn, train=True,
                        shuffle=True)
val_iter2 = MyIterator(val, batch_size=BATCH_SIZE, device = device,
                        repeat=False, sort_key= lambda x:
                        (len(x.German), len(x.English)),
                        batch_size_fn=batch_size_fn, train=True,
                        shuffle=True)
val_iter = MyIterator(val, batch_size=BATCH_SIZE, device = device,
                        repeat=False, sort_key=lambda x: (len(x.German), len(x.English)),
                        batch_size_fn=batch_size_fn, train=False, shuffle = False, sort = False)

def calculate_loss(data_iter,net):
    calc_loss = 0.0
    num_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
    for i, batch in enumerate(data_iter):
      preds = net(batch.German.transpose(0,1), batch.English.transpose(0,1),pad_idx)
      preds = preds[:, :-1, :].contiguous().view(-1, net.tgt_vocab_size)
      targets = batch.English.transpose(0,1)[:, 1:].contiguous().view(-1)
      num_tokens += (targets != pad_idx).data.sum()
      loss = criterion(preds, targets)
      calc_loss += loss.detach()
    return calc_loss/num_tokens,calc_loss

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup**(-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def train_NMT(net, params, train_iter, val_iter):
    best_loss = 1e9
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
    opt = get_std_opt(net)
    # optimizer = optim.Adam(net.parameters(), lr = params['learning_rate'],betas=(0.9, 0.98), eps=1e-9)
    net.train()
    for epoch in range(params['epochs']):
        ep_loss = 0.
        start_time = time.time()
        ntokens = 0
        # for each batch, calculate loss and optimize model parameters
        for i, batch in enumerate(train_iter):
            if i % 1000 == 0:
              print(i)
            src = batch.German.transpose(0,1)
            trg = batch.English.transpose(0,1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            preds = net(src, trg_input, pad_idx).contiguous().view(-1, net.tgt_vocab_size)
            ntokens += (targets != pad_idx).data.sum()
            loss = criterion(preds, targets)
            
            loss.backward()
            opt.step()
            opt.optimizer.zero_grad()
            ep_loss += loss
        if ep_loss < best_loss:
          best_loss = ep_loss
#           torch.save(net, "best_transformer_" + str(epoch) + ".ckpt")
        print('epoch: %d, loss: %0.2f, per token loss : %0.2f, time: %0.2f sec' %(epoch, ep_loss, ep_loss/ntokens, time.time()-start_time))
        dev_loss_per_token, dev_loss = calculate_loss(val_iter,net)
        print('epoch: %d, dev_loss_token: %0.2f, dev_loss: %0.2f' %(epoch, dev_loss_per_token,dev_loss))

params = {}
params['src_vocab_size'] = len(DE_TEXT.vocab.stoi)
params['tgt_vocab_size'] = len(EN_TEXT.vocab.stoi)
params['d_model'] = 512
params['d_ff'] = 2048
params['nheads'] = 8
params['epochs'] = 30
params['nlayers'] = 6
params['dropout'] = 0.1
params['max_len'] = 5000

transformer_obj = Transformer(params)
for p in transformer_obj.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p) 
transformer_obj.cuda()
train_NMT(transformer_obj, params, train_iter, val_iter)

def calculate_bleu(net,data_iter):
  print()
  f = open("sentences5.txt","w")
  net.eval()
  num_batches = 0
  for it, batch in enumerate(data_iter):
#     print(batch.German.transpose(0,1).size(),batch.English.transpose(0,1).size())
    bsz = batch.German.transpose(0,1).size(0)
    for i in range(bsz):
      de_sent = batch.German.transpose(0,1)[i,:].unsqueeze(0)
      en_sent = [EN_TEXT.vocab.stoi['<bos>']]
      en_sent = torch.tensor(en_sent).unsqueeze(0).cuda()
      token = -1
      sentence = []
      while token != EN_TEXT.vocab.stoi['<eos>']:
        preds = net(de_sent, en_sent, pad_idx) # 1 x seq_len2 x tgt_vocab_size
        preds = nn.functional.softmax(preds, dim = -1) # 1 x seq_len2 x tgt_vocab_size
        __, indices = torch.max(preds, -1)
        token = int(indices[:,-1])
        if token != EN_TEXT.vocab.stoi['<eos>']:
          en_sent = torch.cat((en_sent,indices[:,-1].unsqueeze(1)),1)
          sentence.append(EN_TEXT.vocab.itos[int(token)])
      sentence_joined = " ".join(sentence)
      sentence_joined += '\n'
#       print(sentence_joined)
      if i != 0 or it != 0:
        f.write(sentence_joined)
  f.close()

calculate_bleu(transformer_obj,val_iter)