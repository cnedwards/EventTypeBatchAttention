
import torch
from torch import nn
import torch.nn.functional as F


BCE = nn.BCELoss()
BCEL = nn.BCEWithLogitsLoss()
BCELR = nn.BCEWithLogitsLoss(reduction='none')
BCER = nn.BCELoss(reduction='none')
CE = nn.CrossEntropyLoss()

def type_contrastive_loss(values, onehot):
    logits = torch.matmul(values,torch.transpose(values, 0, 1))
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    return BCEL(logits, labels) + BCEL(torch.transpose(logits, 0, 1), labels)

def type_contrastive_sim_loss(sims, onehot):
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    return BCE(sims, labels) + BCE(torch.transpose(sims, 0, 1), labels)

#ignore unseen types except as negatives for seen types
def type_contrastive_sim_ZS_halfmask_loss(sims, onehot, seen):
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    loss = BCER(sims, labels)
    return (loss * seen).mean()


#ignore unseen types except as negatives for seen types
def type_contrastive_sim_ZS_mask_loss(sims, onehot, seen):
    labels = torch.logical_or(torch.matmul(onehot,torch.transpose(onehot, 0, 1)), torch.eye(seen.shape[0], device=sims.device)).float()
    mask = torch.logical_or(~(torch.transpose(torch.tile(~seen, (seen.shape[0], 1)),0,1) * (~seen)), torch.eye(seen.shape[0], device=sims.device))
    loss = BCER(sims, labels)    
    return (loss * mask).mean()
    

#ignore unseen types except as negatives for seen types
def type_contrastive_sim_ZS_mask_margin_loss(sims, onehot, seen, margin = 0.5):
    labels = torch.logical_or(torch.matmul(onehot,torch.transpose(onehot, 0, 1)), torch.eye(seen.shape[0], device=sims.device)).float()
    mask = torch.logical_or(~(torch.transpose(torch.tile(~seen, (seen.shape[0], 1)),0,1) * (~seen)), torch.eye(seen.shape[0], device=sims.device))
    loss = BCER(sims, labels)

    mask_margin = labels == 0
    loss[torch.logical_and(sims < margin, mask_margin)] = 0
    
    return (loss * mask).mean()

#ignore unseen types except as negatives for seen types
def type_contrastive_sim_ZS_mask_loss_entropy(sims, onehot, seen):

    labels = torch.logical_or(torch.matmul(onehot,torch.transpose(onehot, 0, 1)), torch.eye(seen.shape[0], device=sims.device)).float()
    mask = torch.logical_or(~(torch.transpose(torch.tile(~seen, (seen.shape[0], 1)),0,1) * (~seen)), torch.eye(seen.shape[0], device=sims.device))
    antimask = ~mask
    loss = BCER(sims, labels)
    
    entropy = - (torch.log(sims[antimask]) * sims[antimask]).mean() #.sum()
    entropy = torch.nan_to_num(entropy) #empty array...

    return loss[mask].mean(), entropy

#ignore negatives where logits value is less than threshold to mimic softmax
def type_contrastive_ignore_loss(keys, onehot, threshold = 0.1):
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    logits = torch.matmul(keys,torch.transpose(keys, 0, 1))
    logits[logits < threshold] = -20
    return BCEL(logits, labels) + BCEL(torch.transpose(logits, 0, 1), labels)

#weight negatives much less
def type_contrastive_weight_loss(keys, onehot, weight = 0.25):
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    logits = torch.matmul(keys,torch.transpose(keys, 0, 1))
    
    losses = BCELR(logits, labels)
    return weight * losses[~labels.bool()].mean() + (1 - weight) * losses[labels.bool()].mean()
    
def type_contrastive_sim_diagonly_loss(keys, device):
    logits = torch.matmul(keys,torch.transpose(keys, 0, 1))
    labels = torch.arange(logits.shape[0]).to(device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

MSE = nn.MSELoss()

def type_representation_MSE_loss(pred, target):
    return MSE(pred, target)

COS = nn.CosineSimilarity()

def type_representation_cos_loss(pred, target):
    return 1 - COS(pred, target).mean()

#ignore unseen types
def type_representation_cos_ZS_loss(pred, target, seen):
    return ((1 - COS(pred, target))*seen).mean()


def type_representation_contrast_loss(v1, v2, onehot):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.matmul(onehot,torch.transpose(onehot, 0, 1))
    return BCEL(logits, labels) + BCEL(torch.transpose(logits, 0, 1), labels)

def role_reconstruction_loss():
    pass

KLD = torch.nn.KLDivLoss(reduction='none')

def role_KL_ZS_loss(pred, target, seen):
    pred = torch.log(F.softmax(pred, dim=1))
    target = F.softmax(target, dim=1)
    loss = KLD(pred, target)
    loss = loss.mean(axis=1)
    return (loss*seen).mean()

def role_KL_loss(pred, target):
    pred = torch.log(F.softmax(pred, dim=1))
    target = F.softmax(target, dim=1)
    loss = KLD(pred, target)
    loss = loss.mean(axis=1)
    return (loss).mean()

    
def role_BCE_ZS_loss(pred, target, seen):
    loss = BCELR(pred, target)
    loss = loss.mean(axis=1)
    return (loss*seen).mean()
    
def role_BCE_loss(pred, target):
    return BCEL(pred, target)


def BT_contrastive_loss(x, bt_x, device):
  logits = torch.matmul(x, bt_x.T)
  labels = torch.arange(logits.shape[0]).to(device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def CosineReconstruction(x, x_hat):
    return 1 - COS(x, x_hat).mean()


MSER = nn.MSELoss(reduction='none')

def MSE_Margin(x, x_hat, margin):
    loss = MSER(x, x_hat)
    mask = loss < margin
    lmean, lstd = loss.mean(), loss.std()
    loss[mask] = 0
    return loss.mean(), lmean, lstd
    
def COS_Margin(x, x_hat, margin):
    loss = 1 - COS(x, x_hat)
    mask = loss < margin
    lmean, lstd = loss.mean(), loss.std()
    loss[mask] = 0
    return loss.mean(), lmean, lstd
    