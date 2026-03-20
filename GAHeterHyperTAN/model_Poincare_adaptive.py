'''
改进版：自适应双曲空间模型
实现：
1. 节点类型特定曲率学习（突破固定曲率局限）
2. 动态基点选择（突破固定原点映射局限）
'''

from manifold.PoincareManifold import PoincareManifold
import torch.nn.init as init
import torch
import torch.nn as nn
from dgl.nn.pytorch import TypedLinear
import config
import math
from torch_scatter import scatter
from torch_geometric.utils import softmax
import torch.nn.functional as F

args = config.parse()

device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
if args.multi_cuda==0:
    device2=device
else:
    device2 = torch.device('cuda:' + str(eval(args.cuda) + 1) if torch.cuda.is_available() else 'cpu')

EPS= 1e-15
clip_value=0.9899

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class HypLinear_Adaptive(nn.Module):
    """
    自适应双曲线性层：支持节点类型特定曲率
    """
    def __init__(self, args, in_features, out_features, c_dict=None, node_type_name=None, use_bias=True):
        super(HypLinear_Adaptive, self).__init__()
        self.manifold = PoincareManifold(args)
        self.in_features = in_features
        self.out_features = out_features
        self.c_dict = c_dict  # 节点类型到曲率的映射
        self.node_type_name = node_type_name
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        if self.c_dict is not None and self.node_type_name is None:
            raise ValueError("node_type_name must be provided when c_dict is set.")

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        """
        Args:
            x: 输入特征 [N, in_features] (已经在双曲空间中)
        """
        if self.c_dict is None or self.node_type_name is None:
            c = 1.0
        else:
            c_raw = self.c_dict.get(self.node_type_name, None)
            if c_raw is None:
                raise KeyError(f"Curvature parameter for node type '{self.node_type_name}' not found.")
            c = F.softplus(c_raw) + 1e-3
            c = torch.clamp(c, max=8.0)
        
        # Möbius矩阵向量乘法
        mv = self.manifold.mobius_matvec(self.weight, x, c)
        res = self.manifold.proj(mv, c)
        
        if self.use_bias:
            # 将bias映射到双曲空间
            bias_tan = self.manifold.proj_tan0(self.bias.view(1, -1), c)
            hyp_bias = self.manifold.exp_map_zero(bias_tan, c)
            hyp_bias = self.manifold.proj(hyp_bias, c)
            # Möbius加法（使用matrix_matrix_mobius_addition，它不需要c参数）
            # 扩展bias以匹配batch size
            hyp_bias_expanded = hyp_bias.expand(res.shape[0], -1)
            res = self.manifold.matrix_matrix_mobius_addition(res, hyp_bias_expanded)
            res = self.manifold.proj(res, c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c_dict={}'.format(
            self.in_features, self.out_features, self.c_dict
        )


class hhgnnConv_eu_adaptive(nn.Module):
    """
    自适应异构超图卷积层：支持节点类型特定曲率
    """
    def __init__(self, args, in_channels, out_channels, c_dict, device, heads=8, dropout=0., 
                 negative_slope=0.2, skip_sum=False):
        super().__init__()
        self.device = device
        self.type_w = TypedLinear(in_channels, heads * out_channels, num_types=4).to(self.device)
        
        # 注意力参数（与原版相同）
        self.att_v_user = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.att_e_friend = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_visit = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_occurrence = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_self = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.edge_num = args.edge_num
        self.node_number = args.node_number
        self.edge_to_v = args.edge_to_v
        self.node_to_edge = args.node_to_edge
        self.layer_norm = args.layer_norm
        self.edge_type = args.edge_type
        self.node_type = args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length
        self.H_dense = args.H_dense
        
        # 曲率字典
        self.c_dict = c_dict
        
        # 索引（与原版相同）
        self.V_raw_index_type = (args.V_raw_index_type).to(self.device)
        self.V_class = (args.V_class).to(self.device)
        self.E_class = (args.E_class).to(self.device)
        self.V_class_index = (args.V_class_index).to(self.device)
        self.E_class_index = (args.E_class_index).to(self.device)
        self.V_class_index_0 = (args.V_class_index_0).to(self.device)
        self.V_class_index_1 = (args.V_class_index_1).to(self.device)
        self.V_class_index_2 = (args.V_class_index_2).to(self.device)
        self.V_class_index_3 = (args.V_class_index_3).to(self.device)
        self.E_class_index_0 = (args.E_class_index_0).to(self.device)
        self.E_class_index_1 = (args.E_class_index_1).to(self.device)
        self.E_class_index_2 = (args.E_class_index_2).to(self.device)
        self.E_class_index_3 = (args.E_class_index_3).to(self.device)
        
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_v_user)
        glorot(self.att_v_poi)
        glorot(self.att_v_class)
        glorot(self.att_v_time)
        glorot(self.att_e_friend)
        glorot(self.att_e_visit)
        glorot(self.att_e_occurrence)
        glorot(self.att_e_self)

    def forward(self, X, vertex, edges, save_attn: bool = False, attn_cache: dict | None = None, layer_tag: str | None = None):
        """前向传播（与原版相同，但支持自适应曲率；可选保存注意力）"""
        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.type_w(X, self.V_raw_index_type)
        X = X0.view(N, H, C)
        Xve = X[vertex]
        X = Xve

        # 边注意力计算
        X_e_0 = (torch.index_select(X, 0, self.E_class_index_0) * self.att_e_friend).sum(-1)
        X_e_1 = (torch.index_select(X, 0, self.E_class_index_1) * self.att_e_visit).sum(-1)
        X_e_2 = (torch.index_select(X, 0, self.E_class_index_2) * self.att_e_occurrence).sum(-1)
        X_e_3 = (torch.index_select(X, 0, self.E_class_index_3) * self.att_e_self).sum(-1)
        X_e = torch.cat((X_e_0, X_e_1, X_e_2, X_e_3), 0)
        beta_v = torch.gather(X_e, 0, self.E_class_index)
        beta = self.leaky_relu(beta_v)
        beta = softmax(beta, edges, num_nodes=self.edge_num)
        beta = beta.unsqueeze(-1)
        Xe = Xve * beta
        Xe = self.relu(scatter(Xe, edges, dim=0, reduce='sum', dim_size=self.edge_num))

        # 节点注意力计算
        Xe = Xe[edges]
        Xe_2 = Xe
        Xe_2_0 = (torch.index_select(Xe_2, 0, self.V_class_index_0) * self.att_v_user).sum(-1)
        Xe_2_1 = (torch.index_select(Xe_2, 0, self.V_class_index_1) * self.att_v_poi).sum(-1)
        Xe_2_2 = (torch.index_select(Xe_2, 0, self.V_class_index_2) * self.att_v_class).sum(-1)
        Xe_2_3 = (torch.index_select(Xe_2, 0, self.V_class_index_3) * self.att_v_time).sum(-1)
        Xe_2 = torch.cat((Xe_2_0, Xe_2_1, Xe_2_2, Xe_2_3), 0)
        alpha_e = torch.gather(Xe_2, 0, self.V_class_index)
        alpha = self.leaky_relu(alpha_e)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)
        Xv = Xv.view(N, H * C)

        # 可选：缓存注意力信息用于可解释性
        if save_attn and attn_cache is not None:
            tag = layer_tag or "hhgnnConv_eu_adaptive"
            attn_cache[tag] = {
                "beta": beta.squeeze(-1).detach().cpu(),    # [E_all, H]
                "alpha": alpha.squeeze(-1).detach().cpu(),  # [E_all, H]
                "vertex": vertex.detach().cpu(),            # [E_all]
                "edges": edges.detach().cpu(),              # [E_all]
            }
        if self.layer_norm:
            # 如果需要，可以添加层归一化
            pass
        
        return Xv


class HHGNN_Poincare_Adaptive(nn.Module):
    """
    自适应双曲超图神经网络
    创新点：
    1. 节点类型特定曲率学习
    2. 动态基点选择
    """
    
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim, 
                 edge_type, node_type, device_type=device, use_dynamic_base=True):
        super().__init__()
        self.device = device_type
        self.V = V.to(device)
        self.E = E.to(device)
        self.node_type = node_type
        self.use_dynamic_base = use_dynamic_base
        
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim = node_input_dim
        self.edge_type = edge_type
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.lin_out1 = nn.Linear(nhid * args.out_nhead, out_dim, bias=True)
        self.fc_list_node = nn.ModuleList([
            nn.Linear(feats_dim, nfeat, bias=True) 
            for feats_dim in node_input_dim
        ])
        
        self.user_number = args.user_number
        
        def _softplus_inv(x):
            """Inverse softplus for scalar x (>0)."""
            x = float(x)
            # 保证输入大于 eps，避免数值问题
            x = max(x, 1e-6)
            return math.log(math.expm1(x))

        # 合理默认初始曲率（实际曲率 ≈ c_target）
        default_c_target = {
            'user': 1.0,
            'poi': 1.0,
            'poi_class': 1.5,
            'time_point': 0.3,
        }

        # ========== 创新点1：节点类型特定曲率学习 ==========
        # 为每种节点类型学习不同的曲率原始参数（经过 softplus+eps 得到正曲率）
        self.c_user_raw = torch.nn.Parameter(torch.tensor(_softplus_inv(default_c_target['user'] - 1e-3), dtype=torch.float), requires_grad=True)
        self.c_poi_raw = torch.nn.Parameter(torch.tensor(_softplus_inv(default_c_target['poi'] - 1e-3), dtype=torch.float), requires_grad=True)
        self.c_class_raw = torch.nn.Parameter(torch.tensor(_softplus_inv(default_c_target['poi_class'] - 1e-3), dtype=torch.float), requires_grad=True)
        self.c_time_raw = torch.nn.Parameter(torch.tensor(_softplus_inv(default_c_target['time_point'] - 1e-3), dtype=torch.float), requires_grad=True)
        
        # 曲率字典，方便访问
        # 存放原始参数的字典；实际取用时通过 get_curvature 做 softplus 变换
        self.c_dict = {
            'user': self.c_user_raw,
            'poi': self.c_poi_raw,
            'poi_class': self.c_class_raw,
            'time_point': self.c_time_raw
        }
        
        # ========== 创新点2：动态基点选择 ==========
        if self.use_dynamic_base:
            # 为每种节点类型学习一个基点（在双曲空间中的位置）
            # 基点初始化为接近原点的小向量
            # 注意：我们需要两套基点：
            # 1) 输入空间基点：维度为各类型原始输入维度（用于第一阶段映射）
            # 2) 隐藏空间基点：维度为nfeat（用于卷积层前后的映射）
            base_init_scale = 0.1
            # 输入空间基点
            self.base_user_input = nn.Parameter(
                torch.randn(1, node_input_dim[0]) * base_init_scale, requires_grad=True
            )
            self.base_poi_input = nn.Parameter(
                torch.randn(1, node_input_dim[1]) * base_init_scale, requires_grad=True
            )
            self.base_class_input = nn.Parameter(
                torch.randn(1, node_input_dim[2]) * base_init_scale, requires_grad=True
            )
            self.base_time_input = nn.Parameter(
                torch.randn(1, node_input_dim[3]) * base_init_scale, requires_grad=True
            )
            # 隐藏空间基点（统一维度为 nfeat）
            self.base_user_hidden = nn.Parameter(
                torch.randn(1, nfeat) * base_init_scale, requires_grad=True
            )
            self.base_poi_hidden = nn.Parameter(
                torch.randn(1, nfeat) * base_init_scale, requires_grad=True
            )
            self.base_class_hidden = nn.Parameter(
                torch.randn(1, nfeat) * base_init_scale, requires_grad=True
            )
            self.base_time_hidden = nn.Parameter(
                torch.randn(1, nfeat) * base_init_scale, requires_grad=True
            )
            
            # 基点字典
            self.base_input_dict = {
                'user': self.base_user_input,
                'poi': self.base_poi_input,
                'poi_class': self.base_class_input,
                'time_point': self.base_time_input
            }
            self.base_hidden_dict = {
                'user': self.base_user_hidden,
                'poi': self.base_poi_hidden,
                'poi_class': self.base_class_hidden,
                'time_point': self.base_time_hidden
            }
            self.base_input_dims = {
                'user': node_input_dim[0],
                'poi': node_input_dim[1],
                'poi_class': node_input_dim[2],
                'time_point': node_input_dim[3]
            }
        else:
            self.base_input_dict = None
            self.base_hidden_dict = None
            self.base_input_dims = None
        
        # 卷积层（使用自适应版本）
        self.conv_out = hhgnnConv_eu_adaptive(
            args, nhid * nhead, nhid, c_dict=self.c_dict, 
            heads=args.out_nhead, device=device
        )
        self.conv_in = hhgnnConv_eu_adaptive(
            args, nfeat, nhid, c_dict=self.c_dict, 
            heads=nhead, device=device
        )
        
        self.manifold = PoincareManifold(args)
        self.linear_first = nn.ModuleList([
            HypLinear_Adaptive(args, feats_dim, nfeat, c_dict=self.c_dict, node_type_name=ntype)
            for feats_dim, ntype in zip(node_input_dim, self.node_type)
        ])

    def get_curvature(self, node_type_name):
        """获取指定节点类型的曲率"""
        c_raw = self.c_dict.get(node_type_name, self.c_user_raw)
        # 正值约束与数值稳定：softplus + eps，并设置上限夹紧避免过大弯曲
        c = F.softplus(c_raw) + 1e-3
        c = torch.clamp(c, max=8.0)
        return c
    
    def get_base_point(self, node_type_name, space='input'):
        """获取指定节点类型的基点
        space: 'input' 或 'hidden'，分别对应输入空间维度或隐藏空间维度
        """
        if self.base_input_dict is None:
            return None
        if space == 'input':
            base = self.base_input_dict.get(node_type_name, self.base_user_input)
        else:
            base = self.base_hidden_dict.get(node_type_name, self.base_user_hidden)
        # 确保基点是2D张量
        if base.dim() == 1:
            base = base.unsqueeze(0)
        # 将基点投影到双曲球内，确保合法（当作球内坐标）
        c = self.get_curvature(node_type_name)
        base = self.manifold.proj(base, c=c)
        return base
    
    def exp_map_adaptive(self, v, node_type_name):
        """
        自适应指数映射：根据节点类型选择基点
        """
        c = self.get_curvature(node_type_name)
        base = self.get_base_point(node_type_name)
        
        if base is None or not self.use_dynamic_base:
            # 使用原点作为基点（原版方法）
            return self.manifold.exp_map_zero(v, c)
        else:
            # 使用学习到的基点
            # 先将v映射到基点的切空间，再映射到双曲空间
            return self.manifold.exp_map_x(v, base.squeeze(0))
    
    def log_map_adaptive(self, y, node_type_name):
        """
        自适应对数映射：根据节点类型选择基点
        """
        base = self.get_base_point(node_type_name)
        
        if base is None or not self.use_dynamic_base:
            # 使用原点作为基点（原版方法）
            return self.manifold.log_map_zero(y)
        else:
            # 使用学习到的基点
            return self.manifold.log_map_x(y, base.squeeze(0))

    def forward(self, node_attr, save_attn: bool = False, attn_cache: dict | None = None):
        """
        前向传播：使用自适应曲率和动态基点
        简化版本：使用平均曲率进行全局映射，但在关键步骤使用类型特定曲率
        """
        node_feat = {}
        node_type_boundaries = []  # 记录每种节点类型的边界
        start_idx = 0
        
        # 第一步：将节点特征映射到双曲空间（使用输入空间基点）
        for i, ntype in enumerate(self.node_type):
            x_euclidean = node_attr[ntype]
            num_nodes = x_euclidean.shape[0]
            node_type_boundaries.append((start_idx, start_idx + num_nodes, ntype))
            
            # 投影到切空间
            c = self.get_curvature(ntype)
            x_tan = self.manifold.proj_tan0(x_euclidean, c)
            
            # 使用自适应基点映射到双曲空间
            if self.use_dynamic_base:
                base = self.get_base_point(ntype, space='input')
                x_hyp = self.manifold.exp_map_x(x_tan, base)
            else:
                x_hyp = self.manifold.exp_map_zero(x_tan, c)
            
            # 投影确保在双曲空间内
            x_hyp = self.manifold.proj(x_hyp, c)
            
            # 双曲线性变换（简化：使用平均曲率）
            # 注意：这里可以进一步优化为类型特定的线性变换
            node_feat[ntype] = self.linear_first[i](x_hyp)
            start_idx += num_nodes
        
        # 合并所有节点特征
        X = []
        for ntype in self.node_type:
            X.append(node_feat[ntype])
        X = torch.cat(X, dim=0)
        
        V, E = self.V, self.E
        
        # 辅助函数：按节点类型处理（卷积前后使用隐藏空间基点）
        def process_by_type(X_tensor, operation='log', use_base_for_input_only=False):
            """按节点类型处理张量
            use_base_for_input_only=True 时，只在初始输入阶段使用动态基点；其余阶段使用原点
            """
            results = []
            for start, end, ntype in node_type_boundaries:
                X_type = X_tensor[start:end]
                c = self.get_curvature(ntype)
                
                if operation == 'log':
                    if self.use_dynamic_base and not use_base_for_input_only:
                        base = self.get_base_point(ntype, space='hidden')
                        result = self.manifold.log_map_x(X_type, base)
                    else:
                        result = self.manifold.log_map_zero(X_type)
                    result = self.manifold.proj_tan0(result, c)
                elif operation == 'exp':
                    if self.use_dynamic_base and not use_base_for_input_only:
                        base = self.get_base_point(ntype, space='hidden')
                        result = self.manifold.exp_map_x(X_type, base)
                    else:
                        result = self.manifold.exp_map_zero(X_type, c)
                    result = self.manifold.proj(result, c)
                else:
                    result = X_type
                
                results.append(result)
            return torch.cat(results, dim=0)
        
        # 第一层卷积
        # 如果启用动态基点，则在隐藏空间也使用对应基点；否则回退到原点
        use_origin_only = not self.use_dynamic_base
        X = process_by_type(X, 'log', use_base_for_input_only=use_origin_only)
        X = self.conv_in(X, V, E, save_attn=save_attn, attn_cache=attn_cache, layer_tag="conv_in")
        X = process_by_type(X, 'exp', use_base_for_input_only=use_origin_only)
        X = self.relu(X)
        
        # 第二层卷积
        X = process_by_type(X, 'log', use_base_for_input_only=use_origin_only)
        X = self.conv_out(X, V, E, save_attn=save_attn, attn_cache=attn_cache, layer_tag="conv_out")
        X = process_by_type(X, 'exp', use_base_for_input_only=use_origin_only)
        X = self.relu(X)
        
        # 保存双曲空间嵌入（用于可视化）
        X_hyperbolic = X.clone()  # 这是真正在双曲空间的嵌入
        
        # 最终映射回欧几里得空间（用于下游任务）
        X = process_by_type(X, 'log', use_base_for_input_only=use_origin_only)
        X = self.lin_out1(X)
        
        # 返回嵌入和曲率信息（用于监控）
        curvature_info = {
            'c_user': (F.softplus(self.c_user_raw).item() + 1e-3),
            'c_poi': (F.softplus(self.c_poi_raw).item() + 1e-3),
            'c_class': (F.softplus(self.c_class_raw).item() + 1e-3),
            'c_time': (F.softplus(self.c_time_raw).item() + 1e-3),
            'X_hyperbolic': X_hyperbolic  # 添加双曲嵌入
        }
        
        return X, curvature_info

