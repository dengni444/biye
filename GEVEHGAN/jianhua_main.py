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
from jianhua_dataset import *
from model import *
from utils import *

import torch.optim as optim
from config import *

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

args=parse()

device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')
city=args.city
#Hyper-parameters
d_node=128
epoch=args.epochs
K=args.multihead
lambda_1=args.lambda_1
lambda_2=args.lambda_2
lambda_3=args.lambda_3
file='output_new/'+str(city)+'-ist-e-4-vae-no_duochidu-5000---ceshi------*-_multi_head_'+str(K)+'lambda_1_'+str(lambda_1)+'lambda_2_'+str(lambda_2)+'lambda_3_'+str(lambda_3)+'.txt'
print(file)


if __name__ == '__main__':

    g,friend_list_index_test=data(d_node,city)
    g = g.to(device)
    etype = g.etypes
    etype = ('user', 'friend', 'user')
    # rel_names = ['friend', 'visit', 'co_occurrence', 'live_with', 're_live_with', 'class_same', 're_visit']
    rel_names = ['friend', 'visit', 'class_same']
    model = Model(d_node, 256, 512, rel_names, K,latent_dim=64).to(device)

    user_feats = g.nodes['user'].data['u_fe'].to(device)
    poi_feats = g.nodes['poi'].data['p_fe'].to(device)
    node_features = {'user': user_feats, 'poi': poi_feats}
    friend_feats = g.edges['friend'].data['f_fe'].to(device)
    visit_feats = g.edges['visit'].data['v_fe'].to(device)
    # co_occurrence_feat = g.edges['co_occurrence'].data['c_fe'].to(device)
    # live_with_feats = g.edges['live_with'].data['l_fe'].to(device)
    # re_live_with_feats = g.edges['re_live_with'].data['rl_fe'].to(device)
    class_same_feats = g.edges['class_same'].data['cl_fe'].to(device)
    # re_visit_feats = g.edges['re_visit'].data['r_fe'].to(device)
    # edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'co_occurrence': co_occurrence_feat,'live_with': live_with_feats, 're_live_with': re_live_with_feats, 'class_same': class_same_feats,'re_visit': re_visit_feats}
    edge_attr = {'friend': friend_feats, 'visit': visit_feats, 'class_same': class_same_feats}

    pos_edge_index_2 = []
    pos_edge_index = g.edges(etype=('user', 'friend', 'user'))
    pos_edge_index_2.append(pos_edge_index[0].cpu().detach().numpy())
    pos_edge_index_2.append(pos_edge_index[1].cpu().detach().numpy())
    pos_edge_index_2 = torch.tensor(np.array(pos_edge_index_2)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    grad_lr_scheduler = GradientAdaptiveLR(opt, init_lr=args.lr)

    best_auc = 0
    best_ap = 0
    print(city)

    # 获取原始节点特征
    node_feat = node_features.copy()

    # 初始化对抗特征
    # node_feat_adv = {k: v.clone().detach().requires_grad_(True) for k, v in node_feat.items()}
    # 配置对抗训练参数
    use_adversarial_training = args.adversarial  # 'none', 'random', 'fgsm', 'pgd'
    print(use_adversarial_training)
    epsilon = args.epsilon
    alpha = args.alpha
    pgd_iters = args.pgd_iters


    # 定义损失函数（用于对抗扰动）

    for epoch in range(epoch):
        # 获取当前学习率
        current_lr = opt.param_groups[0]['lr']
        negative_graph = construct_negative_graph(g, 5, ('user', 'friend', 'user'))
        neg_edge_index = neg_edge_in(g, 5, ('user', 'friend', 'user'))

        # # 定义损失函数
        # def loss_fn(pos, neg, h):
        #     loss_cor = F.binary_cross_entropy_with_logits(
        #         model.predict(h, pos_edge_index_2, neg_edge_index),
        #         get_link_labels(pos_edge_index_2, neg_edge_index).to(device)
        #     )
        #     return margin_loss(pos, neg) * lambda_3 + loss_cor * lambda_2 + contrastive_loss(h['user'], g) * lambda_1
        #



        # 生成对抗扰动后的特征每5个epoch
        # 默认情况下，node_feat_adv 和 link_logits 使用标准训练的值
        node_feat_adv = node_feat


        # link_logits = None
        # # 对抗训练的设置
        # epsilon = 1e-3
        # alpha = 1e-4  # PGD的步长
        # iters = 3  # PGD的迭代次数
        # pos_edge_index_2 = pos_edge_index_2
        # # 每5个epoch执行对抗训练
        # if epoch % 5 == 0:
        #     # 生成对抗扰动
        #     # 选择使用的对抗训练方法（可以切换FGSM或PGD）
        #     # node_feat_adv = {k: fgsm_perturbation_no_grad(model, g, negative_graph, node_feat, edge_attr, etype,
        #     #                                           epsilon=epsilon, pos_edge_index_2=pos_edge_index_2,
        #     #                                         neg_edge_index=neg_edge_index,lambda_3=3,lambda_2=2,lambda_1=1)
        #     #                      for k, v in node_feat.items()}
        #
        #
        #     node_feat_adv = {k: adversarial_perturbation(v, epsilon=epsilon) for k, v in node_feat_adv.items()}
        #     #
        #     # print("====================================1111")
        #     # print(node_feat)
        #     # print("333333")
        #     # print(node_feat_adv)
        #
        #     # 对抗扰动特征进行前向传播
        #     _, _, node_emb_adv, _,_ = model(g, negative_graph, node_feat_adv, edge_attr, etype)
        #
        #     # print("====================================1211")
        #     user_emb_adv = node_emb_adv['user']
        #
        #     # 计算对抗扰动下的链接得分
        #     link_logits_adv = model.predict(user_emb_adv, pos_edge_index_2, neg_edge_index)
        #
        #     # 计算对抗损失：使用 KL 散度或交叉熵
        #     if link_logits is not None:  # 确保标准损失已经计算
        #         adv_loss = F.kl_div(
        #             F.log_softmax(link_logits_adv, dim=-1),
        #             F.softmax(link_logits.detach(), dim=-1),
        #             reduction='batchmean'
        #         )
        #     else:
        #         adv_loss = 0  # 如果 link_logits 未定义，跳过对抗损失
        # else:
        #     adv_loss = 0  # 非对抗训练阶段，不计算对抗损失


        pos_score, neg_score, node_emb, contrastive_loss ,vae_loss= model(g, negative_graph, node_feat_adv, edge_attr, etype)
        user_emb = node_emb['user']

        # pos_score, neg_score, node_emb, contrastive_loss = model(g, negative_graph, node_features, edge_attr, ('user', 'friend', 'user'))
        # user_emb = node_emb['user']

        link_labels = get_link_labels(pos_edge_index_2, neg_edge_index).to(device)
        link_logits = model.predict(user_emb, pos_edge_index_2, neg_edge_index)
        loss_cor = F.binary_cross_entropy_with_logits(link_logits, link_labels)  # 二元交叉熵损失
        # loss = margin_loss(pos_score, neg_score) * lambda_3 + loss_cor * lambda_2 + contrastive_loss * lambda_1+ adv_loss * (1 if epoch % 5 == 0 else 0) +vae_loss # 仅在对抗训练阶段加入 adv_loss

        loss = margin_loss(pos_score,
                           neg_score) * lambda_3 + loss_cor * lambda_2 + contrastive_loss * lambda_1 + vae_loss  # 仅在对抗训练阶段加入 adv_loss

        # 边距损失   对比损失
        opt.zero_grad()
        loss.backward()

        # for k, v in node_feat_adv.items():
        #     if v.grad is None:
        #         print(f"Warning: {k} has no gradient!")
        #     else:
        #         print(f"Gradient for {k} is calculated.")
        opt.step()

        # grad_lr_scheduler.step(model)



        if epoch % 10 == 0:
            print("epoch:", epoch)
            print("LOSS:", loss.item())
            # test_auc, ap,top_k = test(user_emb,g,friend_list_index_test)
            # Test the model and compute metrics
            test_auc, ap,top_k , f1_pos, f1_neg, f1_macro = test(user_emb, g, friend_list_index_test)
            if test_auc > best_auc:
                best_auc = test_auc
                print("best_auc:", best_auc)
                np.save("data_new/save_user_embedding/ist/ist_vae_e-4-noduochidu-5000/vae_e-4---" + str(best_auc) + ".npy", user_emb.cpu().detach().numpy())
            if ap > best_ap:
                best_ap = ap
                print("beat_ap:", ap)

            # if test_auc > best_auc:
            #     best_auc = test_auc
            #     print("best_auc:", best_auc)
            #     torch.save(model.state_dict(), f"models/best_model_{city}_auc_{best_auc}.pth")
            #     np.save(f"data_new/save_user_embedding/JK/best_auc_JK_cancha1_{best_auc}.npy",
            #             user_emb.cpu().detach().numpy())
            # if ap > best_ap:
            #     best_ap = ap
            #     print("best_ap:", best_ap)
            #     torch.save(model.state_dict(), f"models/best_model_{city}_ap_{best_ap}.pth")

            # # Print F1 scores
            # print("F1 Score (Positive Class):", f1_pos)
            # print("F1 Score (Negative Class):", f1_neg)
            # print("F1 Score (Macro Average):", f1_macro)

            # need_write = f"epoch {epoch} loss: {loss.item()} best_auc: {best_auc} best_ap: {best_ap} " \
            #                  f"f1_pos: {f1_pos} f1_neg: {f1_neg} f1_macro: {f1_macro}"
            need_write = f"epoch {epoch} loss: {loss.item()} best_auc: {best_auc} best_ap: {best_ap} " \
                        f"f1_pos: {f1_pos} f1_neg: {f1_neg} f1_macro: {f1_macro} learning_rate: {current_lr}"
            #need_write="epoch"+str(epoch)+" best_auc: "+str(best_auc)+" best_ap: "+str(best_ap)
            top='top_1+'+str(top_k[0])+' top_5+'+str(top_k[1])+' top_10+'+str(top_k[2])+' top_15+'+str(top_k[3])+' top_20+'+str(top_k[4])
            with open(file, 'a+') as f:
                f.write(need_write + '\n')  # 加\n换行显示
                f.write(top + '\n')
