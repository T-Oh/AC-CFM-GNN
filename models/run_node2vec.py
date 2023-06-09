# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:38:58 2023

@author: tobia
"""
import torch
import os
from torch_geometric.nn import Node2Vec


from utils.get_optimizers import get_optimizer


def run_node2vec(cfg, trainloader, device, params, trial):
    edge_index = next(iter(trainloader)).edge_index

    #edge_index = edge_index[:,:int(len(edge_index[0])/trainloader.batch_size)]
    
    model = Node2Vec(edge_index,
                     embedding_dim = int(params['embedding_dim']),
                     walk_length = int(params['walk_length']),
                     context_size = int(params['context_size']),
                     walks_per_node = int(params['walks_per_node']),
                     num_negative_samples = int(params['num_negative_samples']),
                     p=params['p'],
                     q=params['q']        )
    
    model.to(device)
    loader = model.loader()
    #Get optimizer        
    optimizer = get_optimizer(cfg, model)
    for i in range(cfg['epochs']):
        print(f'Epoch: {i}')
        loss = train_epoch(model, loader, optimizer, device)
        #acc = test(model, labels)
        #print(f'Loss: {loss} \nAcc: {acc}')
        print(f'Loss: {loss}')
               
    embedding = model()

            
    if not os.path.exists('node2vec/'):
        os.makedirs('node2vec/')
    torch.save(embedding, f'node2vec/embedding_{trial}.pt')
    return embedding
    
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss=0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def test(model, labels, x):
    model.eval()
    embedding = model()
    acc = model.test(embedding, labels,embedding, labels,
                         max_iter=150)
    return acc