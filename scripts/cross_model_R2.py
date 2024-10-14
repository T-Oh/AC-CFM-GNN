# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:11:50 2023

@author: tobia

Script used to calculate the R2 scores between the predictions of the models.
"""
import torch
from torchmetrics import Recall, FBetaScore, R2Score
import numpy as np
from scipy.stats import f_oneway, tukey_hsd, ttest_rel
#CROSS MODEL R2s
PATH_TAG    = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/TAG/all_storms/crossval2/'
PATH_GINE   = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GINE/ajusted_std/crossval2/'
PATH_GAT    = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GAT/all_storms/crossval2/'
PATH_MLP    = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/MLP/crossval2/'
PATH_NODE2VEC   = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/Node2Vec/crossval2/'
PATH_NODE2VEC_EMBEDDING = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/Node2Vec/crossval/embedding_crossval.pt'
PATH_RIDGE  = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/Ridge/crossval2/'
PATH_NODEMEAN = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/NodeMean/'


output_gine = torch.load(PATH_GINE+'full_crossval_output_gine.pt')
labels = torch.load(PATH_GINE+'full_crossval_labels.pt')
output_tag = torch.load(PATH_TAG+'full_crossval_output_tag.pt')
output_gat = torch.load(PATH_GAT+'full_crossval_output_gat.pt')

output_mlp = torch.load(PATH_MLP+'full_crossval_output_mlp.pt')
output_N2V = torch.load(PATH_NODE2VEC+'full_crossval_output_N2V.pt')
output_nodemean = torch.load(PATH_NODEMEAN+'full_crossval_output_nodemean.pt')
output_ridge = torch.load(PATH_RIDGE+'full_crossval_output_ridge.pt')
for i in range(7):
    output_nodemean[i] = output_nodemean[i].repeat(int(len(output_N2V[i])/2000))

R2Score=R2Score()

#TAG
tag_gine_R2     = []
tag_gat_R2      = []
tag_mlp_R2      = []
tag_ridge_R2    = []
tag_N2V_R2      = []
tag_nodemean_R2  = []

for i in range(7):

    tag_gine_R2.append(R2Score(output_tag[i].reshape(-1), output_gine[i].reshape(-1)))
    tag_gat_R2.append(R2Score(output_tag[i].reshape(-1), output_gat[i].reshape(-1)))
    tag_mlp_R2.append(R2Score(output_tag[i].reshape(-1), output_mlp[i].reshape(-1)))
    tag_ridge_R2.append( R2Score(output_tag[i].reshape(-1), output_ridge[i].reshape(-1)))
    tag_N2V_R2.append(R2Score(output_tag[i].reshape(-1), output_N2V[i].reshape(-1)))
    tag_nodemean_R2.append(R2Score(output_tag[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(tag_gine_R2).mean()}$\pm${np.array(tag_gine_R2).std()}')
print(f'{np.array(tag_mlp_R2).mean()}$\pm${np.array(tag_mlp_R2).std()}')
print(f'{np.array(tag_ridge_R2).mean()}$\pm${np.array(tag_ridge_R2).std()}')
print(f'{np.array(tag_N2V_R2).mean()}$\pm${np.array(tag_N2V_R2).std()}')
print(f'{np.array(tag_nodemean_R2).mean()}$\pm${np.array(tag_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(tag_gat_R2).mean()}$\pm${np.array(tag_gat_R2).std()}')

#gine
gine_tag_R2     = []
gine_gat_R2      = []
gine_mlp_R2      = []
gine_ridge_R2    = []
gine_N2V_R2      = []
gine_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    gine_tag_R2.append(R2Score(output_gine[i].reshape(-1), output_tag[i].reshape(-1)))
    gine_gat_R2.append(R2Score(output_gine[i].reshape(-1), output_gat[i].reshape(-1)))
    gine_mlp_R2.append(R2Score(output_gine[i].reshape(-1), output_mlp[i].reshape(-1)))
    gine_ridge_R2.append( R2Score(output_gine[i].reshape(-1), output_ridge[i].reshape(-1)))
    gine_N2V_R2.append(R2Score(output_gine[i].reshape(-1), output_N2V[i].reshape(-1)))
    gine_nodemean_R2.append(R2Score(output_gine[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(gine_tag_R2).mean()}$\pm${np.array(gine_tag_R2).std()}')
print(f'{np.array(gine_mlp_R2).mean()}$\pm${np.array(gine_mlp_R2).std()}')
print(f'{np.array(gine_ridge_R2).mean()}$\pm${np.array(gine_ridge_R2).std()}')
print(f'{np.array(gine_N2V_R2).mean()}$\pm${np.array(gine_N2V_R2).std()}')
print(f'{np.array(gine_nodemean_R2).mean()}$\pm${np.array(gine_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(gine_gat_R2).mean()}$\pm${np.array(gine_gat_R2).std()}')

#TAG
tag_gine_R2     = []
tag_gat_R2      = []
tag_mlp_R2      = []
tag_ridge_R2    = []
tag_N2V_R2      = []
tag_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    tag_gine_R2.append(R2Score(output_tag[i].reshape(-1), output_gine[i].reshape(-1)))
    tag_gat_R2.append(R2Score(output_tag[i].reshape(-1), output_gat[i].reshape(-1)))
    tag_mlp_R2.append(R2Score(output_tag[i].reshape(-1), output_mlp[i].reshape(-1)))
    tag_ridge_R2.append( R2Score(output_tag[i].reshape(-1), output_ridge[i].reshape(-1)))
    tag_N2V_R2.append(R2Score(output_tag[i].reshape(-1), output_N2V[i].reshape(-1)))
    tag_nodemean_R2.append(R2Score(output_tag[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(tag_gine_R2).mean()}$\pm${np.array(tag_gine_R2).std()}')
print(f'{np.array(tag_mlp_R2).mean()}$\pm${np.array(tag_mlp_R2).std()}')
print(f'{np.array(tag_ridge_R2).mean()}$\pm${np.array(tag_ridge_R2).std()}')
print(f'{np.array(tag_N2V_R2).mean()}$\pm${np.array(tag_N2V_R2).std()}')
print(f'{np.array(tag_nodemean_R2).mean()}$\pm${np.array(tag_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(tag_gat_R2).mean()}$\pm${np.array(tag_gat_R2).std()}')

#mlp
mlp_gine_R2     = []
mlp_gat_R2      = []
mlp_tag_R2      = []
mlp_ridge_R2    = []
mlp_N2V_R2      = []
mlp_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    mlp_gine_R2.append(R2Score(output_mlp[i].reshape(-1), output_gine[i].reshape(-1)))
    mlp_gat_R2.append(R2Score(output_mlp[i].reshape(-1), output_gat[i].reshape(-1)))
    mlp_tag_R2.append(R2Score(output_mlp[i].reshape(-1), output_tag[i].reshape(-1)))
    mlp_ridge_R2.append( R2Score(output_mlp[i].reshape(-1), output_ridge[i].reshape(-1)))
    mlp_N2V_R2.append(R2Score(output_mlp[i].reshape(-1), output_N2V[i].reshape(-1)))
    mlp_nodemean_R2.append(R2Score(output_mlp[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(mlp_gine_R2).mean()}$\pm${np.array(mlp_gine_R2).std()}')
print(f'{np.array(mlp_tag_R2).mean()}$\pm${np.array(mlp_tag_R2).std()}')
print(f'{np.array(mlp_ridge_R2).mean()}$\pm${np.array(mlp_ridge_R2).std()}')
print(f'{np.array(mlp_N2V_R2).mean()}$\pm${np.array(mlp_N2V_R2).std()}')
print(f'{np.array(mlp_nodemean_R2).mean()}$\pm${np.array(mlp_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(mlp_gat_R2).mean()}$\pm${np.array(mlp_gat_R2).std()}')

#N2V
N2V_gine_R2     = []
N2V_gat_R2      = []
N2V_mlp_R2      = []
N2V_ridge_R2    = []
N2V_tag_R2      = []
N2V_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    N2V_gine_R2.append(R2Score(output_N2V[i].reshape(-1), output_gine[i].reshape(-1)))
    N2V_gat_R2.append(R2Score(output_N2V[i].reshape(-1), output_gat[i].reshape(-1)))
    N2V_mlp_R2.append(R2Score(output_N2V[i].reshape(-1), output_mlp[i].reshape(-1)))
    N2V_ridge_R2.append( R2Score(output_N2V[i].reshape(-1), output_ridge[i].reshape(-1)))
    N2V_tag_R2.append(R2Score(output_N2V[i].reshape(-1), output_tag[i].reshape(-1)))
    N2V_nodemean_R2.append(R2Score(output_N2V[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(N2V_gine_R2).mean()}$\pm${np.array(N2V_gine_R2).std()}')
print(f'{np.array(N2V_mlp_R2).mean()}$\pm${np.array(N2V_mlp_R2).std()}')
print(f'{np.array(N2V_ridge_R2).mean()}$\pm${np.array(N2V_ridge_R2).std()}')
print(f'{np.array(N2V_tag_R2).mean()}$\pm${np.array(N2V_tag_R2).std()}')
print(f'{np.array(N2V_nodemean_R2).mean()}$\pm${np.array(N2V_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(N2V_gat_R2).mean()}$\pm${np.array(N2V_gat_R2).std()}')

#nodemean
nodemean_gine_R2     = []
nodemean_gat_R2      = []
nodemean_mlp_R2      = []
nodemean_ridge_R2    = []
nodemean_N2V_R2      = []
nodemean_tag_R2  = []

for i in range(7):
    if i == 1: continue
    nodemean_gine_R2.append(R2Score(output_nodemean[i].reshape(-1), output_gine[i].reshape(-1)))
    nodemean_gat_R2.append(R2Score(output_nodemean[i].reshape(-1), output_gat[i].reshape(-1)))
    nodemean_mlp_R2.append(R2Score(output_nodemean[i].reshape(-1), output_mlp[i].reshape(-1)))
    nodemean_ridge_R2.append( R2Score(output_nodemean[i].reshape(-1), output_ridge[i].reshape(-1)))
    nodemean_N2V_R2.append(R2Score(output_nodemean[i].reshape(-1), output_N2V[i].reshape(-1)))
    nodemean_tag_R2.append(R2Score(output_nodemean[i].reshape(-1), output_tag[i]))

print(f'{np.array(nodemean_gine_R2).mean()}$\pm${np.array(nodemean_gine_R2).std()}')
print(f'{np.array(nodemean_mlp_R2).mean()}$\pm${np.array(nodemean_mlp_R2).std()}')
print(f'{np.array(nodemean_ridge_R2).mean()}$\pm${np.array(nodemean_ridge_R2).std()}')
print(f'{np.array(nodemean_N2V_R2).mean()}$\pm${np.array(nodemean_N2V_R2).std()}')
print(f'{np.array(nodemean_tag_R2).mean()}$\pm${np.array(nodemean_tag_R2).std()}')
print('GAT')
print(f'{np.array(nodemean_gat_R2).mean()}$\pm${np.array(nodemean_gat_R2).std()}')

#ridge
ridge_gine_R2     = []
ridge_gat_R2      = []
ridge_mlp_R2      = []
ridge_tag_R2    = []
ridge_N2V_R2      = []
ridge_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    ridge_gine_R2.append(R2Score(output_ridge[i].reshape(-1), output_gine[i].reshape(-1)))
    ridge_gat_R2.append(R2Score(output_ridge[i].reshape(-1), output_gat[i].reshape(-1)))
    ridge_mlp_R2.append(R2Score(output_ridge[i].reshape(-1), output_mlp[i].reshape(-1)))
    ridge_tag_R2.append( R2Score(output_ridge[i].reshape(-1), output_tag[i].reshape(-1)))
    ridge_N2V_R2.append(R2Score(output_ridge[i].reshape(-1), output_N2V[i].reshape(-1)))
    ridge_nodemean_R2.append(R2Score(output_ridge[i].reshape(-1), output_nodemean[i]))

print(f'{np.array(ridge_gine_R2).mean()}$\pm${np.array(ridge_gine_R2).std()}')
print(f'{np.array(ridge_mlp_R2).mean()}$\pm${np.array(ridge_mlp_R2).std()}')
print(f'{np.array(ridge_tag_R2).mean()}$\pm${np.array(ridge_tag_R2).std()}')
print(f'{np.array(ridge_N2V_R2).mean()}$\pm${np.array(ridge_N2V_R2).std()}')
print(f'{np.array(ridge_nodemean_R2).mean()}$\pm${np.array(ridge_nodemean_R2).std()}')
print('GAT')
print(f'{np.array(ridge_gat_R2).mean()}$\pm${np.array(ridge_gat_R2).std()}')

#gat
gat_gine_R2     = []
gat_ridge_R2      = []
gat_mlp_R2      = []
gat_tag_R2    = []
gat_N2V_R2      = []
gat_nodemean_R2  = []

for i in range(7):
    if i == 1: continue
    gat_gine_R2.append(R2Score(output_gat[i].reshape(-1), output_gine[i].reshape(-1)))
    gat_ridge_R2.append(R2Score(output_gat[i].reshape(-1), output_ridge[i].reshape(-1)))
    gat_mlp_R2.append(R2Score(output_gat[i].reshape(-1), output_mlp[i].reshape(-1)))
    gat_tag_R2.append( R2Score(output_gat[i].reshape(-1), output_tag[i].reshape(-1)))
    gat_N2V_R2.append(R2Score(output_gat[i].reshape(-1), output_N2V[i].reshape(-1)))
    gat_nodemean_R2.append(R2Score(output_gat[i].reshape(-1), output_nodemean[i]))
print('GAT')
print(f'{np.array(gat_gine_R2).mean()}$\pm${np.array(gat_gine_R2).std()}')
print(f'{np.array(gat_mlp_R2).mean()}$\pm${np.array(gat_mlp_R2).std()}')
print(f'{np.array(gat_tag_R2).mean()}$\pm${np.array(gat_tag_R2).std()}')
print(f'{np.array(gat_N2V_R2).mean()}$\pm${np.array(gat_N2V_R2).std()}')
print(f'{np.array(gat_nodemean_R2).mean()}$\pm${np.array(gat_nodemean_R2).std()}')
print(f'{np.array(gat_ridge_R2).mean()}$\pm${np.array(gat_ridge_R2).std()}')

#ANOVA
print(f_oneway(mlp_gine_R2, mlp_N2V_R2, mlp_tag_R2, gine_mlp_R2, N2V_mlp_R2, tag_mlp_R2))
print(tukey_hsd(mlp_gine_R2, mlp_N2V_R2, mlp_tag_R2, gine_mlp_R2, N2V_mlp_R2, tag_mlp_R2))
print(f_oneway(mlp_gine_R2, mlp_ridge_R2, mlp_tag_R2, gine_mlp_R2, ridge_mlp_R2, tag_mlp_R2, mlp_N2V_R2, N2V_mlp_R2))
print(ttest_rel(mlp_gine_R2, mlp_N2V_R2))



