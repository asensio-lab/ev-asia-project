#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import torch
import os 
import numpy as np
import random

# internal dependencies
from dataset import EVDataset, collate_fn
from model import BertClassifier

# external dependencies
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from torch.utils.data import DataLoader, RandomSampler
from functools import partial
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score


# In[12]:


def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[13]:


set_seed(seed=90)


# In[14]:


#Default Parameters
# BATCH_SIZE = 16
# bert_model_name = 'bert-base-cased'
# path = "./data/"
MAXLEN=512
# EPOCHS = 10
random.seed(90)
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")


# In[15]:


bert_model_name = 'bert-base-cased'
path = "./data/"
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
tokenizer = BertTokenizer.from_pretrained(bert_model_name, max_length=MAXLEN)
assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"
train_df = pd.read_csv(os.path.join(path, 'train_final.csv'))
val_df = pd.read_csv(os.path.join(path, 'valid_final.csv'))
test_df = pd.read_csv(os.path.join(path, 'test_final.csv'))
y_labels = list(train_df.columns[2:])
x_label = "review"


# In[16]:


BATCH_SIZE = 8

train_dataset = EVDataset(tokenizer, train_df, x_label, y_labels)
dev_dataset = EVDataset(tokenizer, val_df, x_label, y_labels)
collate_fn = partial(collate_fn, device=device)
train_sampler = RandomSampler(train_dataset)
dev_sampler = RandomSampler(dev_dataset)
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE, sampler=dev_sampler, collate_fn=collate_fn)


# In[17]:


model = BertClassifier(BertModel.from_pretrained(bert_model_name, output_attentions=True), len(y_labels)).to(device)


# In[18]:


def train(model, iterator, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        mask = (x != 0).float()
        loss, outputs, attn = model(x, attention_mask=mask, labels=y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
#         scheduler.step()
    print(f"Train loss {total_loss / len(iterator)}")

def evaluate(model, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            loss, outputs, attn = model(x, attention_mask=mask, labels=y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    print(classification_report(true, pred > 0.5, target_names = y_labels))
    print(f"Evaluate loss {total_loss / len(iterator)}")
    return f1_score(true, pred > 0.5, average="macro")


# In[19]:


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
EPOCH_NUM = 15
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)


# In[20]:


max_f1 = float('-inf')
for i in range(EPOCH_NUM):
    print('=' * 50, f"EPOCH {i}", '=' * 50)
    train(model, train_iterator, optimizer)
    curr_f1 = evaluate(model, dev_iterator)
    if curr_f1 > max_f1:
        max_f1 = curr_f1
        torch.save(model.state_dict(), './saved_checkpoints/best_model.pt')
        print(f"New best model with F-1: {max_f1}")


# In[21]:


#Gerating preds part 1
test_dataset = EVDataset(tokenizer, test_df, x_label, [])
collate_fn = partial(collate_fn, device=device)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

     
        


# In[22]:


#Generating preds part 2
model.eval()
pred = []
with torch.no_grad():
    total_loss = 0
    for x, y in tqdm(test_iterator):
        mask = (x != 0).float()
        loss, outputs, attn = model(x, attention_mask=mask)
        total_loss += loss
        outputs = outputs > 0.5
        pred += outputs.cpu().numpy().tolist()        
        


# In[23]:


# Saving preds
pd.DataFrame(pred, columns=y_labels).to_csv('preds_patterns_asiapaper' + str(date) + '_seed90'+'.csv')


# In[26]:


#Mapping Data
test_asean = pd.read_csv(os.path.join(path, 'asia_test_final.csv'))

BATCH_SIZE = 8
test_dataset = EVDataset(tokenizer, test_asean, x_label, [])
collate_fn = partial(collate_fn, device=device)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)



# In[27]:


model.eval()
pred = []
with torch.no_grad():
    total_loss = 0
    for x, y in tqdm(test_iterator):
        mask = (x != 0).float()
        loss, outputs, attn = model(x, attention_mask=mask)
        total_loss += loss
        outputs = outputs > 0.5
        pred += outputs.cpu().numpy().tolist()        


# In[28]:


pd.DataFrame(pred, columns=y_labels).to_csv("asia_test_predictions.csv")


# In[ ]:



