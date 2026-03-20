import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import torch_sparse
import pickle
import config
from model_Euclidean import HHGNN_Euclidean
from model_Poincare import HHGNN_Poincare,HHGNN_Poincare_multi
from ablation_study_models import HHGNN_Poincare_without_hetero,HHGNN_Poincare_multi_without_hetero, HHGNN_Poincare_without_hyperbolic,HHGNN_Poincare_multi_without_hyperbolic
from ablation_study_models import HHGNN_Poincare_without_hyperbolic_and_hetero,HHGNN_Poincare_multi_without_hyperbolic_and_hetero
try:
    from model_Poincare_adaptive import HHGNN_Poincare_Adaptive
    ADAPTIVE_MODEL_AVAILABLE = True
except ImportError:
    ADAPTIVE_MODEL_AVAILABLE = False
    print("Warning: Adaptive model not available")

import random
from collections import defaultdict

from torch_scatter import scatter

args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ========== 新增：三种采样策略函数 ==========

def neighbor_aware_sampling(friend_edge_train, user_number, K, args):
    """
    邻居感知采样（最简单）
    避免采样到直接朋友，但保持随机性
    
    Args:
        friend_edge_train: 训练友谊边
        user_number: 用户总数
        K: 每个正样本对应的负样本数量
        args: 配置参数
    
    Returns:
        negative_samples: 负样本列表
    """
    print(f"开始邻居感知采样，用户数: {user_number}, 每个正样本负样本数: {K}")
    
    # 构建用户朋友关系字典
    user_friends = defaultdict(set)
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user1, user2 = edge_list[0], edge_list[1]
        user_friends[user1].add(user2)
        user_friends[user2].add(user1)
    
    negative_samples = []
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user_id = edge_list[0]
        
        # 跳过超出范围的用户ID
        if user_id >= user_number:
            continue
            
        friends = user_friends.get(user_id, set())
        
        # 邻居感知采样：避免直接朋友
        for j in range(K):
            while True:
                neg_user = random.randint(0, user_number-1)
                if neg_user != user_id and neg_user not in friends:
                    negative_samples.append([user_id, neg_user])
                    break
    
    print(f"邻居感知采样完成，生成负样本数: {len(negative_samples)}")
    return negative_samples

def degree_aware_sampling(friend_edge_train, user_number, K, args):
    """
    改进的度感知采样（平衡权重采样 + 朋友关系检查）
    根据用户度数进行平衡权重采样，避免朋友关系
    
    Args:
        friend_edge_train: 训练友谊边
        user_number: 用户总数
        K: 每个正样本对应的负样本数量
        args: 配置参数
    
    Returns:
        negative_samples: 负样本列表
    """
    print(f"开始改进度感知采样（平衡权重采样+朋友关系检查），用户数: {user_number}, 每个正样本负样本数: {K}")
    
    # 构建用户朋友关系字典和度数
    user_friends = defaultdict(set)
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user1, user2 = edge_list[0], edge_list[1]
        user_friends[user1].add(user2)
        user_friends[user2].add(user1)
    
    user_degrees = {}
    for user_id in range(user_number):
        user_degrees[user_id] = len(user_friends.get(user_id, set()))
    
    # 计算度数分布统计
    degree_values = list(user_degrees.values())
    min_degree = min(degree_values)
    max_degree = max(degree_values)
    avg_degree = sum(degree_values) / len(degree_values)
    
    print(f"度数统计: 最小={min_degree}, 最大={max_degree}, 平均={avg_degree:.2f}")
    
    negative_samples = []
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user_id = edge_list[0]
        
        # 跳过超出范围的用户ID
        if user_id >= user_number:
            continue
            
        friends = user_friends.get(user_id, set())
        user_degree = user_degrees[user_id]
        
        # 策略1：改进的权重采样 + 朋友关系检查
        for j in range(K):
            # 计算权重：平衡度数差异和相似性
            weights = []
            candidates = []
            
            for candidate_id in range(user_number):
                if candidate_id != user_id and candidate_id not in friends:
                    candidate_degree = user_degrees[candidate_id]
                    
                    # 改进的权重计算：结合度数差异和相似性
                    degree_diff = abs(candidate_degree - user_degree)
                    
                    # 策略1：适度度数差异 + 相似性奖励
                    if degree_diff <= 2:
                        # 度数相近的用户，给予中等权重
                        weight = 2.0
                    elif degree_diff <= 5:
                        # 中度差异，给予较高权重
                        weight = 3.0
                    else:
                        # 高度差异，给予最高权重但不过分
                        weight = 4.0
                    
                    # 添加随机性因子，避免过于确定性
                    random_factor = random.uniform(0.8, 1.2)
                    weight *= random_factor
                    
                    weights.append(weight)
                    candidates.append(candidate_id)
            
            if candidates and weights:
                # 使用权重进行采样
                total_weight = sum(weights)
                if total_weight > 0:
                    # 归一化权重
                    normalized_weights = [w / total_weight for w in weights]
                    # 权重采样
                    neg_user = np.random.choice(candidates, p=normalized_weights)
                    negative_samples.append([user_id, neg_user])
                else:
                    # 如果权重为0，随机选择
                    neg_user = random.choice(candidates)
                    negative_samples.append([user_id, neg_user])
            else:
                # 如果没有候选用户，随机选择非朋友用户
                while True:
                    neg_user = random.randint(0, user_number-1)
                    if neg_user != user_id and neg_user not in friends:
                        negative_samples.append([user_id, neg_user])
                        break
    
    print(f"改进度感知采样完成，生成负样本数: {len(negative_samples)}")
    return negative_samples

def hierarchical_sampling(friend_edge_train, user_number, K, args):
    """
    分层采样（中等）
    结合度数、邻居关系和随机性的分层采样策略
    
    Args:
        friend_edge_train: 训练友谊边
        user_number: 用户总数
        K: 每个正样本对应的负样本数量
        args: 配置参数
    
    Returns:
        negative_samples: 负样本列表
    """
    print(f"开始分层采样，用户数: {user_number}, 每个正样本负样本数: {K}")
    
    # 构建用户朋友关系字典和度数
    user_friends = defaultdict(set)
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user1, user2 = edge_list[0], edge_list[1]
        user_friends[user1].add(user2)
        user_friends[user2].add(user1)
    
    user_degrees = {}
    for user_id in range(user_number):
        user_degrees[user_id] = len(user_friends.get(user_id, set()))
    
    negative_samples = []
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        user_id = edge_list[0]
        
        # 跳过超出范围的用户ID
        if user_id >= user_number:
            continue
            
        friends = user_friends.get(user_id, set())
        user_degree = user_degrees[user_id]
        
        # 分层采样策略
        for j in range(K):
            if j < K // 3:
                # 策略1: 随机采样（33%）
                while True:
                    neg_user = random.randint(0, user_number-1)
                    if neg_user != user_id and neg_user not in friends:
                        negative_samples.append([user_id, neg_user])
                        break
            elif j < 2 * K // 3:
                # 策略2: 度数相似采样（33%）
                degree_candidates = [uid for uid in range(user_number) 
                                  if uid != user_id and uid not in friends 
                                  and abs(user_degrees[uid] - user_degree) <= 2]
                if degree_candidates:
                    neg_user = random.choice(degree_candidates)
                    negative_samples.append([user_id, neg_user])
                else:
                    # 如果度数相似用户不够，随机选择
                    while True:
                        neg_user = random.randint(0, user_number-1)
                        if neg_user != user_id and neg_user not in friends:
                            negative_samples.append([user_id, neg_user])
                            break
            else:
                # 策略3: 度数差异采样（34%）
                if user_degree <= 3:
                    # 低度数用户选择高度数用户
                    degree_candidates = [uid for uid in range(user_number) 
                                      if uid != user_id and uid not in friends 
                                      and user_degrees[uid] > user_degree + 2]
                else:
                    # 高度数用户选择低度数用户
                    degree_candidates = [uid for uid in range(user_number) 
                                      if uid != user_id and uid not in friends 
                                      and user_degrees[uid] < user_degree - 2]
                
                if degree_candidates:
                    neg_user = random.choice(degree_candidates)
                    negative_samples.append([user_id, neg_user])
                else:
                    # 如果度数差异用户不够，随机选择
                    while True:
                        neg_user = random.randint(0, user_number-1)
                        if neg_user != user_id and neg_user not in friends:
                            negative_samples.append([user_id, neg_user])
                            break
    
    print(f"分层采样完成，生成负样本数: {len(negative_samples)}")
    return negative_samples

def dynamic_negative_sampling(friend_edge_train, user_number, K, args):
    """
    改进的动态负采样策略
    结合多种负采样策略：随机采样 + 困难负采样 + 基于度的采样
    添加自适应权重机制和温和困难采样
    
    Args:
        friend_edge_train: 训练友谊边
        user_number: 用户总数
        K: 每个正样本对应的负样本数量
        args: 配置参数
    
    Returns:
        negative_samples: 负样本列表
    """
    print(f"开始改进的动态负采样，用户数: {user_number}, 每个正样本负样本数: {K}")
    
    # 自适应策略权重：根据数据集大小调整
    if user_number < 4000:
        # 小数据集：更保守的策略
        random_ratio = 0.8
        degree_ratio = 0.15
        hard_ratio = 0.05
    elif user_number < 6000:
        # 中等数据集：平衡策略
        random_ratio = 0.7
        degree_ratio = 0.25
        hard_ratio = 0.05
    else:
        # 大数据集：更多结构化采样
        random_ratio = 0.6
        degree_ratio = 0.3
        hard_ratio = 0.1
    
    print(f"自适应策略权重: 随机{random_ratio*100:.0f}% + 基于度{degree_ratio*100:.0f}% + 困难{hard_ratio*100:.0f}%")
    
    # 获取实际用户ID范围
    actual_user_ids = set()
    for edge in friend_edge_train.values():
        edge_list = list(edge)
        actual_user_ids.add(edge_list[0])
        actual_user_ids.add(edge_list[1])
    
    valid_user_list = sorted(list(actual_user_ids))
    # 限制用户ID范围不超过user_number
    valid_user_list = [uid for uid in valid_user_list if uid < user_number]
    actual_user_count = len(valid_user_list)
    print(f"实际用户ID范围: {min(valid_user_list)} - {max(valid_user_list)}, 数量: {actual_user_count}")
    print(f"有效用户ID范围: {min(valid_user_list)} - {max(valid_user_list)}, 数量: {actual_user_count}")
    
    # 构建用户朋友关系字典
    user_friends = defaultdict(set)
    for edge in friend_edge_train.values():
        # 处理边可能是集合或列表的情况
        edge_list = list(edge)
        user1, user2 = edge_list[0], edge_list[1]
        user_friends[user1].add(user2)
        user_friends[user2].add(user1)
    
    # 计算用户度数（朋友数量）
    user_degrees = {}
    for user_id in valid_user_list:
        user_degrees[user_id] = len(user_friends.get(user_id, set()))
    
    negative_samples = []
    
    for edge in friend_edge_train.values():
        # 处理边可能是集合或列表的情况
        edge_list = list(edge)
        user_id = edge_list[0]
        
        # 跳过超出范围的用户ID
        if user_id >= user_number:
            continue
            
        friends = user_friends.get(user_id, set())
        
        # ========== 改进版本：使用自适应策略比例 ==========
        # 策略1: 随机负采样 - 使用自适应权重
        random_count = int(K * random_ratio)
        random_candidates = [i for i in valid_user_list 
                           if i != user_id and i not in friends]
        if len(random_candidates) >= random_count:
            random_negatives = random.sample(random_candidates, random_count)
        else:
            random_negatives = random_candidates
        
        # 策略2: 基于度的负采样 - 使用自适应权重
        degree_count = int(K * degree_ratio)
        # 选择度数相近但非朋友的用户（放宽条件）
        user_degree = user_degrees.get(user_id, 0)
        degree_candidates = [i for i in valid_user_list 
                           if i != user_id and i not in friends 
                           and abs(user_degrees.get(i, 0) - user_degree) <= 3]  # 放宽到3
        if len(degree_candidates) >= degree_count:
            degree_negatives = random.sample(degree_candidates, degree_count)
        else:
            # 如果度数相近的用户不够，用随机采样补充
            remaining_candidates = [i for i in valid_user_list 
                                  if i != user_id and i not in friends 
                                  and i not in random_negatives]
            degree_negatives = random.sample(remaining_candidates, 
                                           min(degree_count, len(remaining_candidates)))
        
        # 策略3: 温和困难负采样 - 使用自适应权重
        hard_count = K - len(random_negatives) - len(degree_negatives)
        if hard_count > 0:
            # 选择度数略高但非朋友的用户（降低困难程度）
            hard_candidates = [i for i in valid_user_list 
                             if i != user_id and i not in friends 
                             and i not in random_negatives and i not in degree_negatives
                             and user_degrees.get(i, 0) > user_degree 
                             and user_degrees.get(i, 0) <= user_degree + 3]  # 限制困难程度
            if len(hard_candidates) >= hard_count:
                hard_negatives = random.sample(hard_candidates, hard_count)
            else:
                # 如果困难负样本不够，用随机采样补充
                remaining_candidates = [i for i in valid_user_list 
                                      if i != user_id and i not in friends 
                                      and i not in random_negatives and i not in degree_negatives]
                hard_negatives = random.sample(remaining_candidates, 
                                             min(hard_count, len(remaining_candidates)))
        else:
            hard_negatives = []
        
        # 合并所有负样本
        all_negatives = random_negatives + degree_negatives + hard_negatives
        
        # 确保每个正样本都有K个负样本
        if len(all_negatives) < K:
            # 如果负样本不够，随机补充
            remaining_candidates = [i for i in valid_user_list 
                                  if i != user_id and i not in friends 
                                  and i not in all_negatives]
            needed = K - len(all_negatives)
            if len(remaining_candidates) >= needed:
                additional_negatives = random.sample(remaining_candidates, needed)
                all_negatives.extend(additional_negatives)
        
        # 添加到负样本列表
        for neg_user in all_negatives[:K]:  # 确保不超过K个
            negative_samples.append([user_id, neg_user])
    
    print(f"动态负采样完成，生成负样本数: {len(negative_samples)}")
    return negative_samples

def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()

def fetch_data(args):
    city=args.city
    print('city name: ',city)
    read_friend=open(  "data/"+city+"/friend_list_index.pkl",'rb' )
    friend_edge=pickle.load(read_friend)
    friend_edge_num=len(friend_edge)
    args.friend_edge_num=friend_edge_num
    print("the number of friendship hyperedge in raw dataset is:", friend_edge_num)
    print("the number of friendship hyperedge used for training is:", round(friend_edge_num *args.split))

    visit_poi=open(  "data/"+city+"/visit_list_edge_tensor.pkl",'rb' )
    visit_edge=pickle.load(visit_poi)
    visit_edge_num=len(visit_edge)
    args.visit_edge_num=visit_edge_num
    print("the number of check-in hyperedge is:", visit_edge_num)

    tra=open( "data/"+city+"/trajectory_list_index.pkl",'rb')
    trajectory_edge=pickle.load(tra)
    trajectory_edge_num=len(trajectory_edge)
    args.trajectory_edge_num=trajectory_edge_num
    print("the number of trajectory hyperedge is:", trajectory_edge_num)

    user_number=args.user_number
    poi_number=args.poi_number
    poi_class_number=args.poi_class_number
    time_point_number=args.time_point_number
    user_node_attr = torch.tensor(np.random.randint(0,10,(user_number,args.input_dim)) , dtype=torch.float32)
    poi_node_attr=torch.tensor(np.random.randint(0,10,(poi_number, args.input_dim)) , dtype=torch.float32)

    poi_class_attr=torch.zeros(poi_class_number,poi_class_number)
    index=range(0,poi_class_number,1)
    index=torch.LongTensor( index).view(-1,1)
    poi_class_attr=poi_class_attr.scatter_(dim=1,index=index,value=1)

    time_point_attr=torch.zeros(time_point_number,time_point_number)
    index2=range(0,time_point_number,1)
    index2 = torch.LongTensor(index2).view(-1, 1)
    time_point_attr=time_point_attr.scatter_(dim=1,index=index2,value=1)

    node_attr={}
    node_attr['user']=user_node_attr
    node_attr['poi']=poi_node_attr
    node_attr['poi_class']=poi_class_attr
    node_attr['time_point'] = time_point_attr


    train_rate=args.split
    friend_edge_train_len  =round(friend_edge_num *train_rate)
    all_index = list(np.arange(0, friend_edge_num, 1))
    train_edge_index = sorted(random.sample(all_index, friend_edge_train_len))
    test_edge_index_true = sorted(list(set(all_index).difference(set(train_edge_index))))

    friend_edge_train={}
    for i in range(len(train_edge_index)):
        friend_edge_train[i]=friend_edge[train_edge_index[i]]


    friend_edge_test=[]
    for i in range(len(test_edge_index_true)):
        friend_edge_test.append(list(friend_edge[ test_edge_index_true[i]]))

    for i in range(len(test_edge_index_true)):
        friend_edge_test.append([list(friend_edge[ test_edge_index_true[i]])[0], random.randint(0, user_number-1) ])

    test_label=[]
    for i in range(2* len(test_edge_index_true)):
        if i <len(test_edge_index_true):
            test_label.append(1)
        else:
            test_label.append(0)
    test_label=np.array(test_label)
    friend_edge_test=(torch.tensor(friend_edge_test,dtype=torch.long)).t().contiguous()
    K=args.negative_K
    friend_edge_train_all=[]
    
    # ========== 原始版本：添加正样本 ==========
    for i in range(len(friend_edge_train)):
        friend_edge_train_all.append(list(friend_edge_train[i]))

    # ========== 使用随机采样策略 ==========
    print("使用随机采样策略...")
    # 随机负采样
    negative_samples = []
    for i in range(len(friend_edge_train)):
        for j in range(K):
            negative_samples.append([list(friend_edge_train[i])[0], random.randint(0, user_number-1)])
    
    friend_edge_train_all.extend(negative_samples)

    friend_edge_train_all =(torch.tensor(np.array(friend_edge_train_all),dtype=torch.long)).t().contiguous()

    friend_edge_train_all_label=[]
    # ========== 原始版本：正样本标签 ==========
    for i in range(len(friend_edge_train)):
        friend_edge_train_all_label.append(1)
    
    # ========== 使用随机采样的负样本标签 ==========
    for i in range(len(negative_samples)):
        friend_edge_train_all_label.append(0)
    
    
    friend_edge_train_all_label=torch.tensor(np.array(friend_edge_train_all_label),dtype=torch.long)

    G={}
    G['friend']=friend_edge_train
    G['check_in'] = visit_edge
    G['trajectory'] = trajectory_edge

    print("There are", user_number, "user nodes in this hypergraph.")
    print("There are", poi_number, "POI nodes in this hypergraph.")
    print("There are", poi_class_number, "POI category nodes in this hypergraph.")
    print("There are", time_point_number, "time-day nodes in this hypergraph.")


    return   G, node_attr,friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,len(friend_edge_train)

def initialise( G,node_attr , args, node_type,edge_type, unseen=None):

    G2={}
    z=0
    for i in edge_type:
        for j in range(len(G[i])):
            G2[z]=G[i][j]
            z+=1
    print("There are", len(G2), "original hyperedges in this hypergraph.")

    G=G2.copy()
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    node_number= args.user_number+ args.poi_number+ args.poi_class_number+ args.time_point_number
    if args.add_self_loop:
        Vs = set(range(node_number))
        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and list(nodes)[0] in Vs:
                Vs.remove(list(nodes)[0])
        for v in Vs:
            G[f'self-loop-{v}'] = [v]

    print("There are", len(G), "hyperedges in this hypergraph.")
    print("Among them,", len(G) - len(G2), "are added self-loop hyperedges.")


    args.self_loop_edge_number=len(G)-len(G2)
    edge_type.append('self-loop')
    N, M = node_number, len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()
    H_dense=H.todense()
    H_dense=H_dense.transpose()
    H_dense=torch.tensor(H_dense,dtype=torch.long)
    H_dense=H_dense.to_sparse_coo()
    args.H_dense=H_dense.to(device)
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()
    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col

    degE = scatter(degV[V], E, dim=0, reduce='sum')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    args.edge_num=max(E)+1
    args.node_number=node_number
    args.degV = degV.to(device)
    args.degE = degE.to(device)
    args.degE2 = degE2.pow(-1.).to(device)

    nhid = args.nhid
    nhead = args.nhead
    edge_input_length=[round(args.friend_edge_num*args.split),args.visit_edge_num,args.trajectory_edge_num,args.self_loop_edge_number]
    node_input_dim=[]
    for i in node_type:
        node_input_dim.append(node_attr[i].shape[1])
    args.node_input_dim=node_input_dim
    node_input_length = [args.user_number,args.poi_number,args.poi_class_number,args.time_point_number]
    V_raw_index_type=[0 for i in range(args.user_number)]+ [1 for i in range(args.poi_number)]+[2 for i in range(args.poi_class_number)]+[3 for i in range(args.time_point_number)]
    args.V_raw_index_type=torch.tensor(V_raw_index_type,dtype=torch.long)
    args.edge_type=edge_type
    args.node_type=node_type

    a=0
    edge_input_length_raw = []
    for i in range(len(edge_input_length)):
        edge_input_length_raw.append(a+edge_input_length[i])
        a=a + edge_input_length[i]

    b=0
    node_input_length_raw=[]
    for i in range(len(node_input_length)):
        node_input_length_raw.append(b+node_input_length[i])
        b= b+ node_input_length[i]

    V_class=[]
    V_class_index_0,V_class_index_1,V_class_index_2,V_class_index_3=[],[],[],[]
    for i in range(V.shape[0]):
        if V[i] <node_input_length_raw[0]:
            V_class.append(0) #user
            V_class_index_0.append(i)
        elif node_input_length_raw[0]<= V[i] <node_input_length_raw[1]:
            V_class.append(1)#POI
            V_class_index_1.append(i)
        elif node_input_length_raw[1]<= V[i] <node_input_length_raw[2]:
            V_class.append(2)#POItype
            V_class_index_2.append(i)
        elif node_input_length_raw[2]<= V[i] <node_input_length_raw[3]:
            V_class.append(3)#timepoint
            V_class_index_3.append(i)

    E_class=[]
    E_class_index_0,E_class_index_1,E_class_index_2,E_class_index_3=[],[],[],[]
    # E_class_index=[]
    for i in range(E.shape[0]):
        if E[i]<edge_input_length_raw[0]:
            E_class.append(0) #friend
            E_class_index_0.append(i)
        elif edge_input_length_raw[0]<=E[i]<edge_input_length_raw[1]:
            E_class.append(1) #check-in
            E_class_index_1.append(i)
        elif edge_input_length_raw[1]<=E[i]<edge_input_length_raw[2]:
            E_class.append(2)#Trajectory
            E_class_index_2.append(i)
        elif edge_input_length_raw[2]<=E[i]<edge_input_length_raw[3]:
            E_class.append(3) #self-loop
            E_class_index_3.append(i)

    edge_to_v = {}  # Records the node indices within each hyperedge
    print(E.shape[0])
    for i in range(E.shape[0]):
        # print(edge_to_v.keys())
        if E[i].item() in edge_to_v.keys():
            edge_to_v[E[i].item()].append(V[i].item())
        else:
            edge_to_v[E[i].item()] = [V[i].item()]

    # Convert to tensor
    for i in range(args.edge_num):
        edge_to_v[i] = torch.tensor(edge_to_v[i], dtype=torch.long).to(device)
    args.edge_to_v = edge_to_v

    node_to_edge = {}  # Records the number of hyperedges each node belongs to
    for i in range(V.shape[0]):
        if V[i].item() in node_to_edge.keys():
            node_to_edge[V[i].item()].append(E[i].item())
        else:
            node_to_edge[V[i].item()] = [E[i].item()]

    # Convert to tensor
    for i in range(node_number):
        node_to_edge[i] = torch.tensor(node_to_edge[i], dtype=torch.long).to(device)
    args.node_to_edge = node_to_edge

    args.V_class = torch.tensor(V_class, dtype=torch.long)
    args.E_class = torch.tensor(E_class, dtype=torch.long)

    # V_class_index_0 records the sequence index of each V node type corresponding to E
    args.V_class_index_0 = torch.tensor(V_class_index_0, dtype=torch.long)
    args.V_class_index_1 = torch.tensor(V_class_index_1, dtype=torch.long)
    args.V_class_index_2 = torch.tensor(V_class_index_2, dtype=torch.long)
    args.V_class_index_3 = torch.tensor(V_class_index_3, dtype=torch.long)

    # E_class_index_0 records the sequence index of each E type corresponding to V for message passing
    args.E_class_index_0 = torch.tensor(E_class_index_0, dtype=torch.long)
    args.E_class_index_1 = torch.tensor(E_class_index_1, dtype=torch.long)
    args.E_class_index_2 = torch.tensor(E_class_index_2, dtype=torch.long)
    args.E_class_index_3 = torch.tensor(E_class_index_3, dtype=torch.long)

    E_class_index=torch.unsqueeze( torch.cat((args.E_class_index_0,args.E_class_index_1,args.E_class_index_2,args.E_class_index_3) ,0),1)
    args.E_class_index  =E_class_index.repeat(1, nhead)
    V_class_index=torch.unsqueeze(torch.cat((args.V_class_index_0,args.V_class_index_1,args.V_class_index_2,args.V_class_index_3) ,0) ,1)
    args.V_class_index= V_class_index.repeat(1, nhead)

    args.V=V
    args.E=E
    args.edge_input_length=edge_input_length_raw
    args.node_input_length=node_input_length_raw


    args.dataset_dict={'hypergraph':G,'n':N,'features':torch.randn(N,args.input_dim)}

    if args.ablation_study == 0:  # 0 indicates no ablation study, proceed with normal training
        if args.manifold_name == 'euclidean':
            model = HHGNN_Euclidean(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
            model.to(device)
        elif args.manifold_name == 'poincare':
            # 检查是否使用自适应模型
            if args.use_adaptive == 1 and ADAPTIVE_MODEL_AVAILABLE:
                if args.multi_cuda == 0:
                    use_dynamic_base = (args.use_dynamic_base == 1)
                    model = HHGNN_Poincare_Adaptive(
                        args, args.input_dim, nhid, args.out_dim, nhead, 
                        V, E, node_input_dim, edge_type, node_type, 
                        device_type=device, use_dynamic_base=use_dynamic_base
                    )
                    model.to(device)
                    print(f"Using Adaptive Model with dynamic_base={use_dynamic_base}")
                else:
                    print("Warning: Adaptive model does not support multi-GPU yet, using original model")
                    if args.multi_cuda == 0:
                        model = HHGNN_Poincare(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                        model.to(device)
                    elif args.multi_cuda == 1:
                        model = HHGNN_Poincare_multi(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
            else:
                if args.multi_cuda == 0:
                    model = HHGNN_Poincare(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                    model.to(device)
                elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                    model = HHGNN_Poincare_multi(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
            # optimiser = RiemannianAMSGrad(args, model.parameters(), lr=args.lr)

    elif args.ablation_study == 1:  # 1 indicates ablation study training
        # Check the ablation state. By default, the ablation study is only conducted in the Poincaré disk model.
        if args.ablation_state == 0:  # 0 removes heterogeneity
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
        elif args.ablation_state == 1:  # 1 removes the hyperbolic space
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hyperbolic(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hyperbolic(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)
        elif args.ablation_state == 10:  # 10 removes both hyperbolic space and heterogeneity
            if args.multi_cuda == 0:
                model = HHGNN_Poincare_without_hyperbolic_and_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type, device_type=device)
                model.to(device)
            elif args.multi_cuda == 1:  # Train using dual GPUs. No need for model.to(device), training starts directly.
                model = HHGNN_Poincare_multi_without_hyperbolic_and_hetero(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim, edge_type, node_type)


    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return model, optimiser, G


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """
    d = np.array(M.sum(1))
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}
    return DI.dot(M)
