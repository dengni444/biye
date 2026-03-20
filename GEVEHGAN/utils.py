import dgl
import numpy as np
import torch
from torch import nn
import torch as th
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
from config import *
import torch.optim as optim
args=parse()

device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')


def adversarial_perturbation(features, epsilon=1e-3):
    """
    随机扰动节点特征。
    """
    perturb = torch.randn_like(features) * epsilon
    return features + perturb

def fgsm_perturbation_no_grad(model, g, neg_g, node_feat, edge_attr, etype,epsilon=1e-3,pos_edge_index_2=None
                              ,neg_edge_index=None,lambda_3=None,lambda_2=None,lambda_1=None):
    """
    FGSM 对抗扰动，使用 no_grad 来减少内存开销。
    """
    # 创建副本，避免直接修改原始特征
    node_feat_adv = node_feat.copy()

    # 创建副本，避免直接修改原始特征
    # node_feat_adv = {k: v.clone().detach().requires_grad_(True) for k, v in node_feat.items()}

    # # 检查 node_feat 是否是字典
    # if isinstance(node_feat_adv, dict):
    #     # 提取 user 和 poi 特征
    #     user_feat = node_feat_adv.get('user')
    #     poi_feat = node_feat_adv.get('poi')
    #
    #     if user_feat is not None and poi_feat is not None:
    #         # 这里假设拼接 `user` 和 `poi` 特征为一个输入张量，具体根据模型需求调整
    #         node_feat_adv = torch.cat([user_feat, poi_feat], dim=0)  # 在适当的维度上拼接
    #     else:
    #         node_feat_adv = user_feat if user_feat is not None else poi_feat
    # else:
    #     # 如果 node_feat_adv 不是字典，直接使用它作为输入
    #     node_feat_adv = node_feat_adv


    # 前向传播
    pos_score, neg_score, node_emb, contrastive_loss, vae_loss = model(g, neg_g, node_feat_adv, edge_attr,
                                                                       etype)
    user_emb = node_emb['user']

    # 计算损失
    # pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr, ('user', 'friend', 'user'))
    # user_emb = node_emb['user']

    link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)
    link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)
    loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)  # 二元交叉熵损失
    loss = margin_loss(pos_score,
                       neg_score) * lambda_3 + loss_cor * lambda_2 + contrastive_loss * lambda_1  + vae_loss  # 仅在对抗训练阶段加入 adv_loss

    # 反向传播计算梯度
    loss.backward()
    # 使用 torch.no_grad() 来避免计算梯度
    # 使用 torch.no_grad() 来避免计算梯度
    # 打印梯度检查哪些 tensor 的 grad 是 None
    for k, v in node_feat_adv.items():
        if v.grad is None:
            print(f"Warning: {k} has no gradient!")
        else:
            print(f"Gradient for {k} is calculated.")

    # 使用 torch.no_grad() 来避免计算梯度
    perturb = {k: epsilon * v.grad.sign() if v.grad is not None else torch.zeros_like(v)
               for k, v in node_feat_adv.items()}


    # with torch.no_grad():
    #     perturb = {k: epsilon * v.grad.sign() for k, v in node_feat_adv.items()}
    # 更新对抗特征
    node_feat_adv = {k: v + perturb[k] for k, v in node_feat_adv.items()}
    return node_feat_adv


def pgd_perturbation(model, g, negative_graph, node_feat, edge_attr, etype, loss_fn, epsilon=1e-3, alpha=1e-4, iters=3):
    """
    基于PGD的对抗扰动生成，并加入扰动范围裁剪。

    Parameters:
    - model: 目标模型
    - g: 正图
    - negative_graph: 负图
    - node_feat: 节点特征字典
    - edge_attr: 边特征字典
    - etype: 边的类型
    - loss_fn: 损失函数
    - epsilon: 最大扰动幅度
    - alpha: 每次扰动的步长
    - iters: 迭代次数

    Returns:
    - node_feat_adv: 添加了对抗扰动的节点特征字典
    """
    # 初始化对抗特征，确保可以计算梯度
    node_feat_adv = {k: v.clone().detach().requires_grad_(True) for k, v in node_feat.items()}

    # 记录原始特征的上下界
    original_min_value = {k: v.min().item() for k, v in node_feat.items()}
    original_max_value = {k: v.max().item() for k, v in node_feat.items()}

    for _ in range(iters):
        # 清除前一次的梯度
        for k, v in node_feat_adv.items():
            if v.grad is not None:
                v.grad.zero_()

        # 前向传播
        pos_pred, neg_pred, h, _,_ = model(g, negative_graph, node_feat_adv, edge_attr, etype)

        # 计算损失
        loss = loss_fn(pos_pred, neg_pred, h)

        # 反向传播计算梯度
        loss.backward(retain_graph=True)

        # 生成扰动（基于梯度的符号）
        perturb = {k: alpha * v.grad.sign() if v.grad is not None else torch.zeros_like(v) for k, v in node_feat_adv.items()}

        # 更新对抗特征并裁剪
        node_feat_adv = {k: (v + perturb[k]).clamp(min=original_min_value[k] - epsilon, max=original_max_value[k] + epsilon)
                         for k, v in node_feat_adv.items()}

    return node_feat_adv





def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).to(device)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        utype, _, vtype = etype
        if utype==vtype:
            src, dst = graph.edges(etype=etype)
            h2=h[utype]
            logits = cosine_similarity(h2[src], h2[dst])
            logits_2 = torch.relu(logits)
            return logits_2
        if utype!=vtype:
            src, dst = graph.edges(etype=etype)
            h2_u=h[utype]
            h2_v=h[vtype]
            logits = cosine_similarity(h2_u[src], h2_v[dst])
            logits_2 = torch.relu(logits)
            return logits_2

def contrastive_loss(user_emb,g):       #计算对比损失
    # adj_friend=g.adj(scipy_fmt='coo',etype='friend')
    adj_friend = g.adj_external(scipy_fmt='coo',etype='friend' )  # for dgl >1.0.x , use adj_external()
    adj_friend=adj_friend.todense()
    row,col=np.diag_indices_from(adj_friend)
    adj_friend[row,col]=1
    # a=torch.norm(user_emb[0],dim=-1,keepdim=True)
    user_emb_norm=torch.norm(user_emb,dim=-1,keepdim=True)


    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = torch.exp(dot_numerator / dot_denominator / 0.2)
    x=(torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
    matrix_mp2sc = sim/(torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
    adj_friend=torch.tensor(adj_friend).to(device)
    lori_mp = -torch.log(matrix_mp2sc.mul(adj_friend).sum(dim=-1)).mean()
    return lori_mp

def margin_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def neg_edge_in(graph,k,etype):
    # edgtypes= ('user', 'friend', 'user')
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src  = src.repeat_interleave(k)#.to(device)
    neg_dst  = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    neg_dst  = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    neg_edge_= torch.stack([neg_src,neg_dst],dim=0)
    return neg_edge_


def test(user_emb,g,friend_list_index_test):
    src, dst = g.edges(etype='friend')
    src=list(src.cpu().detach().numpy())
    dst=list(dst.cpu().detach().numpy())
    friend_ture={}
    for i in range(len(src)):
        if src[i] in friend_ture.keys():
            friend_ture[src[i]]=friend_ture[src[i]]+[dst[i]]
        else:
            friend_ture[src[i]]=[dst[i]]

    test_pos_src, test_pos_dst = friend_list_index_test[0], friend_list_index_test[1]     \
    # Negative pairs
    seed = 30100
    torch.manual_seed(30100)
    torch.cuda.manual_seed(30100)
    torch.cuda.manual_seed_all(30100)
    test_neg_src = test_pos_src
    test_neg_dst = torch.randint(0, g.num_nodes(ntype='user'), (g.num_edges(etype='friend'),))
    test_src = torch.cat([test_pos_src, test_neg_src])
    test_dst = torch.cat([test_pos_dst, test_neg_dst])
    test_labels = torch.cat(
        [torch.ones_like(test_pos_src), torch.zeros_like(test_neg_src)])
    test_preds = []
    for i in range(len(test_src)):
        test_preds.append((F.cosine_similarity(user_emb[test_src[i]], user_emb[test_dst[i]], dim=0)))
    auc = sklearn.metrics.roc_auc_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    ap = sklearn.metrics.average_precision_score(test_labels.detach().numpy(), torch.tensor(test_preds))
    # print('Link Prediction AUC:', auc)
    # print("average_precision AP:", ap)

    #Top-k
    user_emb_norm = torch.norm(user_emb, dim=-1, keepdim=True)

    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = (dot_numerator / dot_denominator )



    user_number=g.num_nodes(ntype='user')
    cos=[[-1]*user_number for i in range(user_number) ]
    for i in range(g.num_nodes(ntype='user')):
        sim[i][i]=-1
        if i in friend_ture.keys():
            x=friend_ture[i]
            for j in x:
                sim[i][j]=-1



    friend_test_true={}
    test_pos_src=list(test_pos_src.numpy())
    test_pos_dst=list(test_pos_dst.numpy())
    for i in range(len(test_pos_src)):
        if test_pos_src[i] in friend_test_true.keys():
            friend_test_true[test_pos_src[i]]=friend_test_true[test_pos_src[i]]+[test_pos_dst[i]]
        else:
            friend_test_true[test_pos_src[i]]=[test_pos_dst[i]]

    for i in range(len(test_pos_dst)):
        if test_pos_dst[i] in friend_test_true.keys():
            friend_test_true[test_pos_dst[i]]=friend_test_true[test_pos_dst[i]]+[test_pos_src[i]]
        else:
            friend_test_true[test_pos_dst[i]]=[test_pos_src[i]]


    y_true=[]
    y_score=[]
    for i in friend_test_true.keys():
        y_true.append( friend_test_true[i])
        y_score.append(sim[i])

    # start counting top-k
    k=[1,5,10,15,20] #top-k

    right_k=[0 for i in range(len(k))]
    for i in range(len(y_true)):
        sim_i = y_score[i].cpu().detach().numpy()
        for j in range(len(k)):
            s = sim_i.argsort()[-k[j]:][::-1]
            if set(list(s)) & set(y_true[i]):
                right_k[j]+=1
    top_k=[0 for i in range(len(k))]
    for j in range(len(k)):
        top_k[j]=right_k[j]/len(y_true)
        print("Top ",k[j],'accuracy score is:', right_k[j]/len(y_true))
    #
    # 二值化预测值，阈值为0.5
    test_preds = torch.tensor(test_preds)  # 将列表转换为张量

    # 二值化预测值，阈值为0.5
    test_preds_binary = (test_preds > 0.5).float()

    # 正类F1分数 (默认计算正类)
    f1_pos = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(), pos_label=1)

    # 负类F1分数 (通过设置 pos_label=0 计算负类F1分数)
    f1_neg = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(), pos_label=0)

    # 如果想获取宏观平均F1分数，也可以计算
    f1_macro = sklearn.metrics.f1_score(test_labels.detach().numpy(), test_preds_binary.detach().numpy(),
                                        average='macro')

    print('Link Prediction AUC:', auc)
    print("Average Precision (AP):", ap)
    print("F1 Score (Positive Class):", f1_pos)
    print("F1 Score (Negative Class):", f1_neg)
    print("F1 Score (Macro Average):", f1_macro)

        # Top-k 准确率部分省略, 保持不变...


    # k=10
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    #
    # k=1
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=5
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=15
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))
    #
    # k=20
    # right=0
    # for i in range(len(y_true)):
    #     sim_i=y_score[i].cpu().detach().numpy()
    #     s=sim_i.argsort()[-k:][::-1]
    #     if set(list(s))& set(y_true[i]):
    #         right+=1
    # print("Top ",k,'accuracy score is:', right/len(y_true))

    # return auc, ap,top_k
    return auc, ap,top_k, f1_pos, f1_neg, f1_macro




class GradientAdaptiveLR:
    def __init__(self, optimizer, init_lr, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.lr = init_lr
        self.max_grad_norm = max_grad_norm  # 最大梯度值

    def step(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > self.max_grad_norm:
            self.lr *= 1.1  # 梯度过大时，增大学习率
        else:
            self.lr *= 0.9  # 梯度较小时，减小学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
