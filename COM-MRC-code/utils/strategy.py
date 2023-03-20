import math
import torch
import torch.nn.functional as F

def logits2aspect(gs,gt,M,K,gamma,w,gamma2,token2word_idx,is_test):
    R,U,O=[],[],[]
    R2,U2,O2=[],[],[]
    M = min(M,gs[:-1].size(0))
    gs_topk_values,gs_topk_index = gs[:-1].topk(M,dim=0)
    gt_topk_values,gt_topk_index = gt[:-1].topk(M,dim=0)

    for i in gs_topk_index:
        for j in gt_topk_index:
            if i<=j and gs[i].item()+gt[j].item()>gamma:
                ul = gs[i].item()+gt[j].item()- w* (token2word_idx[j]-token2word_idx[i]+1)
                rl = (i.item(),j.item())
                U.append(ul)
                R.append(rl)

    while len(R)!=0 and len(O)<K:
        idx = U.index(max(U))
        l,r = R[idx][0], R[idx][1]
        O.append(R[idx])
        R.pop(idx)
        U.pop(idx)
        R_,U_ = [],[]
        for k in range(len(R)):
            if  R[k][1]<l-1 or R[k][0]>r+1:
                R_.append(R[k])
                U_.append(U[k])
        R,U=R_,U_

    if len(O)==0:
        for i in gs_topk_index:
            for j in gt_topk_index:
                if i<=j and gs[i].item()+gt[j].item()>gamma2:
                    ul2 = gs[i].item()+gt[j].item()- w* (j-i+1)
                    rl2 = (i.item(),j.item())
                    U2.append(ul2)
                    R2.append(rl2)
        if len(R2)!=0:
            idx = U2.index(max(U2))
            O.append(R2[idx])

    return O

def logits2aspect_aspect(gs,gt,M,K,gamma,w,gamma2,token2word_idx,is_test,test_i):
    R,U,O=[],[],[]
    R2,U2,O2=[],[],[]

    M = min(M,gs[:-1].size(0))
    gs_topk_values,gs_topk_index = gs[:-1].topk(M,dim=0)
    gt_topk_values,gt_topk_index = gt[:-1].topk(M,dim=0)

    for i in gs_topk_index:
        for j in gt_topk_index:
            if i<=j and gs[i].item()+gt[j].item()>gamma:
                ul = (2.5-i/gs[:-1].size(0))*(gs[i].item()+gt[j].item()- w* (token2word_idx[j]-token2word_idx[i]+1))
                rl = (i.item(),j.item())
                U.append(ul)
                R.append(rl)

    while len(R)!=0 and len(O)<K:
        idx = U.index(max(U))
        l,r = R[idx][0], R[idx][1]
        O.append(R[idx])
        R.pop(idx)
        U.pop(idx)
        R_,U_ = [],[]
        for k in range(len(R)):
            if  R[k][1]<l-1 or R[k][0]>r+1:
                R_.append(R[k])
                U_.append(U[k])
        R,U=R_,U_

    if len(O)==0:
        for i in gs_topk_index:
            for j in gt_topk_index:
                if i<=j and gs[i].item()+gt[j].item()>gamma2:
                    ul2 = gs[i].item()+gt[j].item()- w* (j-i+1)
                    rl2 = (i.item(),j.item())
                    U2.append(ul2)
                    R2.append(rl2)
        if len(R2)!=0:
            idx = U2.index(max(U2))
            O.append(R2[idx])

    return O

