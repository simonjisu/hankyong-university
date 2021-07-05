import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import BertModel, BertConfig, BertTokenizer

class QADataset(Dataset):
    def __init__(self, data_path, fmt="\t", end_tkn="[E]"):
        # read files
        data = pd.read_csv(data_path, sep=fmt, encoding="utf-8", header=None).rename(columns=dict(enumerate(["Q", "A"])))
        self.Q = data["Q"].tolist()
        self.A = (data["A"] + end_tkn).tolist()
        
    def __getitem__(self, index):
        # return the matched index data
        return self.Q[index], self.A[index]
        
    def __len__(self):
        # length of dataset
        return len(self.Q)
    
# -------------------------------Seq2Seq--------------------------------------
class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, bidirec=False):
        super().__init__()        
        self.hidden_size = hidden_size
        self.n_direc = 2 if bidirec else 1
        
        # Embedding Layer: Embedding(30002, 768, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data = embedding_weight
        # Encoder RNN
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          bidirectional=bidirec, 
                          batch_first=True)

    def forward(self, inputs):
        """
        Inputs:
        - inputs: (B, T_e)
        Outputs:
        - hiddens: (1, B, H_e)
        ==========================================
        B: Mini Batch size
        T_e: Encoder Max Length of Tokens
        E_e: Encoder Embedding Size
        H_e: Encoder Hidden Size
        """
        # (B, T_e) > (B, T_e, E_e)
        inputs = self.embedding(inputs)
        
        # gru 
        # outputs: (B, T_e, n_direc*H_e)
        # hiddens: (n_direc*n_layers, B, H_e)
        _, hiddens = self.gru(inputs)
        
        # Take the last hidden vector
        # last_hidden: (n_direc*n_layers, B, H_e) > (B, H_e)
        # summation the last hidden
        last_hidden = hiddens[-self.n_direc:].sum(0)
        
        # unsqueeze last_hidden: (1, B, H_e)
        return last_hidden.unsqueeze(0)
    
class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, start_idx, end_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        # Embedding Layer: Embedding(30002, 768, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data = embedding_weight
        # Decoder RNN
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            bidirectional=False, 
            batch_first=True
        )
        # Final Linear Layer
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.linear.weight.data = embedding_weight

    def init_sos(self, batch_size, device):
        # Create Tensor with start token: [S]
        return torch.LongTensor([self.start_idx]*batch_size).unsqueeze(1).to(device)
    
    def forward(self, enc_hiddens, gold=None, max_len=None):
        """
        Inputs:
        - enc_hiddens: (1, B, H_d)
        - max_len: T_d, if it is None means at training phase
        - gold: answer token if it is not None means using Teacher Force Model
        Outputs:
        - scores: results of all predictions = (B, T_d, vocab_size)
        ==========================================
        B: Mini Batch size
        T_d: Decoder Max Length of Tokens
        E_d: Decoder Embedding Size
        H_d: Decoder Hidden Size
        V_d: Vocab Length
        """
        max_len = gold.size(1) if max_len is None else max_len
        batch_size = enc_hiddens.size(1)
        # initialize input tokens with start token [S]: (B, 1)
        inputs = self.init_sos(batch_size, device=enc_hiddens.device)
        
        # (B, 1) > (B, 1, E_d)
        inputs = self.embedding(inputs)
        
        scores = []
        hiddens = enc_hiddens
        for i in range(0, max_len):
            # hiddens = (1, B, H_d)
            _, hiddens = self.gru(inputs, hiddens)
            
            # score = (1, B, H_d) > (B, H_d) > (B, V_d)
            score = self.linear(hiddens.squeeze(0))
            scores.append(score)
            
            if gold is not None:
                # Training
                pred = gold[:, i]  # (B,)
            else:
                # Testing
                # predict next token score
                pred = score.softmax(-1).argmax(-1)  # (B,)
                # stop when the token is end_idx
                if (pred == self.end_idx).sum() == batch_size:  # all stop
                    break
                
            inputs = self.embedding(pred.unsqueeze(1))

        # (T_d, B, vocab_size) > (B, T_d, vocab_size)
        scores = torch.stack(scores).transpose(0, 1)
        return scores
    
class EncoderDecoder(nn.Module):
    """Encoder - Decoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, start_idx, end_idx, bidirec=False):
        """
        Class for linking encoder and decoder
        """
        super().__init__()
        self.encoder = Encoder(
            embedding_weight=embedding_weight,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            bidirec=bidirec
        )
        self.decoder = Decoder(
            embedding_weight=embedding_weight,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
    def forward(self, input_qs, input_as, max_len=None):
        """
        scores 크기: (B*(T_d), vocab_size)
        """
        enc_hiddens = self.encoder(input_qs)
        scores = self.decoder(enc_hiddens, gold=input_as, max_len=max_len)
        return scores
    
    def load_embeddings(self, state_dict):
        self.encoder.embedding.weight.data = state_dict['encoder.embedding.weight']
        self.decoder.embedding.weight.data = state_dict['decoder.embedding.weight']
        
# -------------------------------Seq2Seq Attn--------------------------------------
class EncoderAttn(nn.Module):
    """Encoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, bidirec=False):
        super().__init__()        
        self.hidden_size = hidden_size
        self.n_direc = 2 if bidirec else 1
        
        # Embedding Layer: Embedding(30002, 768, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data = embedding_weight
        # Encoder RNN
        self.gru = nn.GRU(hidden_size, 
                          hidden_size, 
                          bidirectional=bidirec, 
                          batch_first=True)

    def forward(self, inputs):
        """
        Inputs:
        - inputs: (B, T_e)
        Outputs:
        - outputs: (B, T, H_e)
        - hiddens: (1, B, H_e)
        ==========================================
        B: Mini Batch size
        T_e: Encoder Max Length of Tokens
        E_e: Encoder Embedding Size
        H_e: Encoder Hidden Size
        """
        # (B, T_e) > (B, T_e, E_e)
        inputs = self.embedding(inputs)
        
        # gru 
        # outputs: (B, T_e, n_direc*H_e)
        # hiddens: (n_direc*n_layers, B, H_e)
        outputs, hiddens = self.gru(inputs)
        
        # Take the last hidden vector
        # last_hidden: (B, H)
        # summation the last hidden
        outputs = torch.stack(torch.chunk(outputs, chunks=2, dim=2)).sum(0)
        last_hidden = hiddens[-self.n_direc:].sum(0)
        
        # unsqueeze last_hidden: (1, B, H)
        return outputs, last_hidden.unsqueeze(0)
    
class DecoderAttn(nn.Module):
    """Decoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, start_idx, end_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        # Embedding Layer: Embedding(30002, 768, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data = embedding_weight
        # Decoder RNN
        self.gru = nn.GRU(
            2*hidden_size, 
            hidden_size, 
            bidirectional=False, 
            batch_first=True
        )
        # Final Linear Layer
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.linear.weight.data = embedding_weight

    def init_sos(self, batch_size, device):
        # Create Tensor with start token: [S]
        return torch.LongTensor([self.start_idx]*batch_size).unsqueeze(1).to(device)
    
    def forward(self, enc_outputs, enc_hiddens, gold=None, max_len=None, rt_attn=False):
        """
        Inputs:
        - enc_outputs: (B, T_e, H_d)
        - enc_hiddens: (1, B, H_d)
        - max_len: T_d, if it is None means at training phase
        - gold: answer token if it is not None means using Teacher Force Model
        Outputs:
        - scores: results of all predictions = (B, T_d, vocab_size)
        ==========================================
        B: Mini Batch size
        T_d: Decoder Max Length of Tokens
        E_d: Decoder Embedding Size
        H_d: Decoder Hidden Size
        V_d: Vocab Length
        """
        max_len = gold.size(1) if max_len is None else max_len
        batch_size = enc_hiddens.size(1)
        # initialize input tokens with start token [S]: (B, 1)
        inputs = self.init_sos(batch_size, device=enc_hiddens.device)
        
        # (B, 1) > (B, 1, E_d)
        inputs = self.embedding(inputs)
        
        scores = []
        attn_weights = []
        
        hiddens = enc_hiddens
        for i in range(0, max_len):
            # Create context
            # 1. attention scores: (B, 1, H_d) x (B, H_d, T_e) = (B, 1, T_e)
            attn_score = torch.bmm(
                hiddens.transpose(0, 1), enc_outputs.transpose(1, 2)
            )
            # 2. attention distribution: (B, 1, T_e)
            attn_dist = attn_score.softmax(-1)
            
            # 3. context matrix: (B, 1, T_e) x (B, T_e, H_d) = (B, 1, H_d)
            context = torch.bmm(attn_dist, enc_outputs)
            
            # 4. context concat: (B, 1, E_d + H_d)
            inputs = torch.cat([context, inputs], dim=2)
            
            # hiddens = (1, B, H_d)
            _, hiddens = self.gru(inputs, hiddens)
            
            # score = (1, B, H_d) > (B, H_d) > (B, V_d)
            score = self.linear(hiddens.squeeze(0))
            scores.append(score)
            attn_weights.append(attn_dist.data)
            
            if gold is not None:
                # Training
                pred = gold[:, i]  # (B,)
            else:
                # Testing
                # predict next token score
                pred = score.softmax(-1).argmax(-1)  # (B,)
                # stop when the token is end_idx
                if (pred == self.end_idx).sum() == batch_size:  # all stop
                    break
                
            inputs = self.embedding(pred.unsqueeze(1))

        # (T_d, B, vocab_size) > (B, T_d, vocab_size)
        scores = torch.stack(scores).transpose(0, 1)
        if rt_attn:
            attn_weights = torch.cat(attn_weights, 1)
            return scores, attn_weights
        return scores, None

class EncoderDecoderAttn(nn.Module):
    """Encoder - Decoder"""
    def __init__(self, embedding_weight, vocab_size, hidden_size, start_idx, end_idx, bidirec=False):
        """
        Class for linking encoder and decoder
        """
        super().__init__()
        self.encoder = EncoderAttn(
            embedding_weight=embedding_weight,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            bidirec=bidirec
        )
        self.decoder = DecoderAttn(
            embedding_weight=embedding_weight,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
    def forward(self, input_qs, input_as, max_len=None, rt_attn=False):
        """
        scores 크기: (B*(T_d), vocab_size)
        """
        enc_outputs, enc_hiddens = self.encoder(input_qs)
        scores, attn = self.decoder(enc_outputs, enc_hiddens, gold=input_as, max_len=max_len, rt_attn=rt_attn)
        return scores, attn
    
    def load_embeddings(self, state_dict):
        self.encoder.embedding.weight.data = state_dict['encoder.embedding.weight']
        self.decoder.embedding.weight.data = state_dict['decoder.embedding.weight']
# utils
def train(model, tokenizer, data_loader, loss_function, optimizer, print_step, device):
    # Training
    total_loss = 0
    vocab_size = len(tokenizer)
    n_train = len(data_loader.dataset)
    model.train()
    
    for batch_idx, (q, a) in enumerate(data_loader):
        # Tokenize
        input_qs = tokenizer(q, return_tensors="pt", add_special_tokens=False, padding=True, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
        input_as = tokenizer(a, return_tensors="pt", add_special_tokens=False, padding=True, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
        input_qs, input_as = input_qs.to(device), input_as.to(device)
        
        optimizer.zero_grad()
        scores = model(input_qs, input_as, max_len=None)
        loss = loss_function(scores.contiguous().view(-1, vocab_size), input_as.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % print_step == 0:
            percentage = (batch_idx*data_loader.batch_size / n_train) * 100
            print(f" - [{percentage:.2f}%] train loss: {loss:.4f}")
    return total_loss

def train_attn(model, tokenizer, data_loader, loss_function, optimizer, print_step, device):
    # Training
    total_loss = 0
    vocab_size = len(tokenizer)
    n_train = len(data_loader.dataset)
    model.train()
    
    for batch_idx, (q, a) in enumerate(data_loader):
        # Tokenize
        input_qs = tokenizer(q, return_tensors="pt", add_special_tokens=False, padding=True, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
        input_as = tokenizer(a, return_tensors="pt", add_special_tokens=False, padding=True, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
        input_qs, input_as = input_qs.to(device), input_as.to(device)
        
        optimizer.zero_grad()
        scores, _ = model(input_qs, input_as, max_len=None)
        loss = loss_function(scores.contiguous().view(-1, vocab_size), input_as.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % print_step == 0:
            percentage = (batch_idx*data_loader.batch_size / n_train) * 100
            print(f" - [{percentage:.2f}%] train loss: {loss:.4f}")
    return total_loss