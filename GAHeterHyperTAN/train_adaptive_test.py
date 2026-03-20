#coding=utf-8
"""
测试脚本：在最小数据集上运行自适应双曲空间模型
"""
import pickle
import wandb
import time
import dgl
import os
import sys
import glob
import torch.nn.utils as nn_utils

# 预处理自定义参数（必须在导入 utils 之前，因为 utils.py 会调用 config.parse()）
_resume_flag = '--resume' in sys.argv
if _resume_flag:
    sys.argv.remove('--resume')

_train_id_value = None
if '--train-id' in sys.argv:
    idx = sys.argv.index('--train-id')
    if idx + 1 < len(sys.argv):
        try:
            _train_id_value = int(sys.argv[idx + 1])
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
        except:
            sys.argv.pop(idx)

from utils import *
from prepare import fetch_data, initialise
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR

args = config.parse()
# 强制使用自适应模型
args.use_adaptive = 1
# 仅曲率学习：关闭动态基点
args.use_dynamic_base = 0
# 使用NYC数据集
# args.city = 'BER'  # 使用命令行参数 --city
args.epochs = 8000
args.negative_K = 4

# wandb.init 已移至 train_id 确定后
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
if args.multi_cuda==0:
    device2=device
else:
    device2 = torch.device('cuda:' + str(eval(args.cuda) + 1) if torch.cuda.is_available() else 'cpu')

add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'
model_name = args.model_name

now_time=time.strftime("%m%d%H%M", time.localtime())

# Define node and edge types
node_type=['user','poi','poi_class','time_point']
edge_type=['friend','check_in','trajectory']

# ========== 文件命名 + train_id 版本管理 ==========
# 使用预处理的参数
specified_train_id = _train_id_value

log_base = f'output/{args.city}_adaptive_ep{args.epochs}'
file = log_base + '.txt'  # 后续会根据train_id更新
embedding_file = f'output_embedding/{args.city}_adaptive_ep{args.epochs}.pkl'  # 后续更新
# 确保输出目录存在
os.makedirs('output', exist_ok=True)
os.makedirs('output_embedding', exist_ok=True)
print(f"Output file: {file}")
print(f"Embedding file: {embedding_file}")

G, node_attr, friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,k =fetch_data(args)
model, optimizer,G = initialise(G, node_attr , args,node_type,edge_type)

# ===== 优化器重构：为曲率/基点设置独立学习率并加入 warmup =====
base_lr = getattr(args, 'lr', 0.001)
weight_decay = getattr(args, 'weight_decay', 0.0)
curvature_lr_factor = 0.05
base_point_lr_factor = 0.4

curvature_params = []
base_point_params = []
main_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'c_' in name and name.endswith('_raw'):
        curvature_params.append(param)
    elif 'base_' in name:
        base_point_params.append(param)
    else:
        main_params.append(param)

param_groups = []
if main_params:
    param_groups.append({"params": main_params, "lr": base_lr, "weight_decay": weight_decay})
if curvature_params:
    param_groups.append({
        "params": curvature_params,
        "lr": base_lr * curvature_lr_factor,
        "weight_decay": 0.0
    })
if base_point_params:
    param_groups.append({
        "params": base_point_params,
        "lr": base_lr * base_point_lr_factor,
        "weight_decay": 0.0
    })

if param_groups:
    optimizer = torch.optim.Adam(param_groups)

warmup_epochs = 200
total_epochs = int(args.epochs)
cosine_tmax = max(1, total_epochs - warmup_epochs)

def _warmup_lambda(current_epoch: int):
    if current_epoch < warmup_epochs:
        return max(0.1, float(current_epoch + 1) / float(warmup_epochs))
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=_warmup_lambda)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_tmax)
friend_edge_train_all_label=(torch.tensor( friend_edge_train_all_label, dtype=torch.float32)).to(device2)

friend_edge_train_list = []
for i in range(len(friend_edge_train)):
    f = list(friend_edge_train[i])
    friend_edge_train_list.append((f[0], f[1]))
    friend_edge_train_list.append((f[1], f[0]))

friend_edge_train_list = np.array(friend_edge_train_list)
friend_edge_train_list = torch.tensor(friend_edge_train_list, dtype=torch.long).t().contiguous()
g = dgl.graph((friend_edge_train_list[0], friend_edge_train_list[1]))
g=g.to(device2)

tanh=torch.nn.Tanh()

for i in node_type:
    node_attr[i]= torch.tensor(node_attr[i],).to(device)

best_test_auc, test_auc, Z = 0, 0, None
tic_epoch = time.time()

# 记录曲率变化
curvature_history = {
    'c_user': [],
    'c_poi': [],
    'c_class': [],
    'c_time': []
}

curvature_targets = {
    'user': 1.0,
    'poi': 1.0,
    'poi_class': 1.5,
    'time_point': 0.3,
}
curvature_reg_weight = 1e-3
curvature_div_weight = 3e-4
freeze_epochs = 300

# ========== 断点续训 + 版本管理 ==========
ckpt_dir = Path('checkpoints_adaptive')
ckpt_dir.mkdir(parents=True, exist_ok=True)
start_epoch = 0

# 检查是否加载checkpoint (--resume 参数，默认False)
resume = _resume_flag

# ===== train_id 确定逻辑 =====
train_id = 1

# 1. 如果用户指定了 --train-id，优先使用
if specified_train_id is not None:
    train_id = specified_train_id
    print(f"📌 [Specified] Using train_id={train_id} from --train-id parameter")
elif resume:
    # 2. --resume 模式：加载最新版本
    existing_files = glob.glob(str(ckpt_dir / f"{args.city}_adaptive_train*.pt"))
    if existing_files:
        train_numbers = []
        for f in existing_files:
            try:
                num = int(f.split('train')[-1].replace('.pt', ''))
                train_numbers.append(num)
            except:
                pass
        if train_numbers:
            train_id = max(train_numbers)
    print(f"📖 [Resume] Continuing training session: train_id={train_id}")
else:
    # 3. 默认新训练：创建新版本号
    existing_files = glob.glob(str(ckpt_dir / f"{args.city}_adaptive_train*.pt"))
    if existing_files:
        train_numbers = []
        for f in existing_files:
            try:
                num = int(f.split('train')[-1].replace('.pt', ''))
                train_numbers.append(num)
            except:
                pass
        if train_numbers:
            train_id = max(train_numbers) + 1
        print(f"🆕 [New] Starting new training session: train_id={train_id}")
    else:
        print(f"🆕 [First] First training session: train_id={train_id}")

# ===== 更新所有输出文件路径（统一使用train_id） =====
file = f'{log_base}_train{train_id}.txt'
embedding_file = f'output_embedding/{args.city}_adaptive_ep{args.epochs}_train{train_id}.pkl'
versioned_ckpt_path = ckpt_dir / f"{args.city}_adaptive_train{train_id}.pt"
best_ckpt_path = ckpt_dir / f"{args.city}_adaptive_best_train{train_id}.pt"

# ===== 初始化/恢复 wandb =====
wandb_run_id = None
if resume or specified_train_id is not None:
    # 尝试从已有 checkpoint 加载 wandb_run_id
    temp_ckpt_path = versioned_ckpt_path if versioned_ckpt_path.exists() else None
    if temp_ckpt_path:
        try:
            temp_ckpt = torch.load(temp_ckpt_path, map_location='cpu')
            wandb_run_id = temp_ckpt.get('wandb_run_id', None)
            if wandb_run_id:
                print(f"🔄 Found wandb run_id: {wandb_run_id}, will resume logging")
        except:
            pass

# 过滤掉不能序列化的参数（排除tensor和大对象）
wandb_config = {}
for _k, _v in vars(args).items():
    if hasattr(_v, 'is_sparse') and _v.is_sparse:
        continue
    if hasattr(_v, 'shape'):  # 跳过所有tensor
        continue
    if isinstance(_v, (dict, list)) and len(str(_v)) > 10000:  # 跳过大对象
        continue
    wandb_config[_k] = _v

# 设置 wandb API key
import os
os.environ['WANDB_API_KEY'] = 'caf6611055ea7599c4cf82575c30b8b30d2b373e'

if wandb_run_id:
    # 续接已有的 wandb run
    wandb.init(
        id=wandb_run_id,
        resume="must",
        entity="yztyzt799-china-university-of-geosciences",
        project="test",
        config=wandb_config,
    )
else:
    # 创建新的 wandb run
    wandb.init(
        name=f"{args.city}_adaptive_train{train_id}",
        entity="yztyzt799-china-university-of-geosciences",
        project="test",
        config=wandb_config,
    )
    wandb_run_id = wandb.run.id
    print(f"🆕 New wandb run created: {wandb_run_id}")

print(f"\n📋 Training Configuration:")
print(f"   City: {args.city} | Epochs: {args.epochs} | Version: train{train_id}")
print(f"   Checkpoint: {versioned_ckpt_path}")
print(f"   Log file:   {file}")
print(f"   Embedding:  {embedding_file}\n")


# ===== 加载checkpoint =====
load_path = versioned_ckpt_path if versioned_ckpt_path.exists() else None
best_test_ap = 0.0
best_acc_test = 0.0
best_top_k = [0.0] * 5
best_epoch = 0

if load_path and load_path.exists() and (resume or specified_train_id is not None):
    try:
        ckpt = torch.load(load_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_test_auc = float(ckpt.get('best_test_auc', 0.0))
        best_test_ap = float(ckpt.get('best_test_ap', 0.0))
        best_acc_test = float(ckpt.get('best_acc_test', 0.0))
        best_top_k = ckpt.get('best_top_k', [0.0]*5)
        best_epoch = int(ckpt.get('best_epoch', 0))
        print(f"✅ Loaded checkpoint from {load_path}")
        print(f"   Start epoch: {start_epoch} | Best AUC: {best_test_auc:.4f} @ epoch {best_epoch}\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}. Start from scratch.\n")

target_end_epoch = start_epoch + int(args.epochs)

for epoch in range(start_epoch, target_end_epoch):
    optimizer.zero_grad()
    model.train()
    # 冻结-解冻策略：前 freeze_epochs 冻结曲率与基点参数
    if epoch == start_epoch:
        for name, p in model.named_parameters():
            if 'c_' in name and name.endswith('_raw'):
                p.requires_grad = False
            if 'base_' in name:
                p.requires_grad = False
    if epoch == freeze_epochs:
        for name, p in model.named_parameters():
            if 'c_' in name and name.endswith('_raw'):
                p.requires_grad = True
            if 'base_' in name:
                p.requires_grad = True
    
    # 模型前向传播（自适应模型返回curvature_info）
    result = model(node_attr)
    if isinstance(result, tuple) and len(result) == 2:
        Z, curvature_info = result
        # 记录曲率
        if epoch % 50 == 0:
            curvature_history['c_user'].append(curvature_info.get('c_user', 0))
            curvature_history['c_poi'].append(curvature_info.get('c_poi', 0))
            curvature_history['c_class'].append(curvature_info.get('c_class', 0))
            curvature_history['c_time'].append(curvature_info.get('c_time', 0))
    else:
        # 原版模型返回 (Z, c)
        Z, c = result
    
    predic_label = F.cosine_similarity( Z[friend_edge_train_all[0] ], Z[friend_edge_train_all[1] ])

    loss_cross = F.binary_cross_entropy_with_logits( predic_label, friend_edge_train_all_label)
    loss_margin = margin_loss( predic_label[: k], predic_label[k: ]  )
    con_loss = contrastive_loss(Z,g)
    loss= con_loss* args.lam_1 + loss_cross* args.lam_2 + loss_margin * args.lam_3

    curvature_reg = torch.tensor(0.0, device=loss.device)
    if hasattr(model, "c_dict") and model.c_dict:
        reg_accum = torch.tensor(0.0, device=loss.device)
        c_values = []
        for ntype, param in model.c_dict.items():
            target = curvature_targets.get(ntype, 1.0)
            c_value = F.softplus(param) + 1e-3
            reg_accum = reg_accum + (c_value - target) ** 2
            c_values.append(c_value)
        curvature_reg = curvature_reg_weight * reg_accum
        # 差异化正则：鼓励不同类型曲率拉开
        curvature_div = torch.tensor(0.0, device=loss.device)
        if len(c_values) >= 2:
            for i in range(len(c_values)):
                for j in range(i + 1, len(c_values)):
                    curvature_div = curvature_div + torch.abs(c_values[i] - c_values[j])
        loss = loss + curvature_reg + curvature_div_weight * curvature_div

    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    # 学习率调度：warmup 后切换到 cosine
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    # 每100轮打印一次loss并写入文件
    if isinstance(curvature_reg, torch.Tensor):
        curvature_reg_value = curvature_reg.item()
    else:
        curvature_reg_value = float(curvature_reg)

    if epoch % 100 == 0:
        loss_msg = f"Epoch: {epoch} loss_total: {loss.item():.6f} " \
                   f"loss_cross: {loss_cross.item():.6f} " \
                   f"loss_margin: {loss_margin.item():.6f} " \
                   f"loss_contrastive: {con_loss.item():.6f} " \
                   f"loss_curvature_reg: {curvature_reg_value:.6f}"
        print(loss_msg)
        with open(file, 'a') as f:
            f.write(loss_msg + '\n')
        wandb.log({
            "epoch": epoch,
            "loss_total": loss.item(),
            "loss_cross": loss_cross.item(),
            "loss_margin": loss_margin.item(),
            "loss_contrastive": con_loss.item(),
            "loss_curvature_reg": curvature_reg_value,
            "lr_group_0": optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        }, step=epoch)

    if epoch % 100 == 0:
        end_time = time.time()
        execution_time = end_time - tic_epoch
        print(f"Execution time for 100 epochs: {execution_time} seconds")
        tic_epoch = time.time()
        
        # 打印曲率信息
        if isinstance(result, tuple) and len(result) == 2:
            Z, curvature_info = result
            print(f"Curvatures - User: {curvature_info.get('c_user', 0):.4f}, "
                  f"POI: {curvature_info.get('c_poi', 0):.4f}, "
                  f"Class: {curvature_info.get('c_class', 0):.4f}, "
                  f"Time: {curvature_info.get('c_time', 0):.4f}")
            try:
                with open(file, 'a') as f:
                    f.write(f"epoch{epoch} curvatures user:{curvature_info.get('c_user', 0):.6f} "
                            f"poi:{curvature_info.get('c_poi', 0):.6f} "
                            f"class:{curvature_info.get('c_class', 0):.6f} "
                            f"time:{curvature_info.get('c_time', 0):.6f}\n")
            except Exception as e:
                print(f"[Warn] Failed to write curvature line: {e}")
            wandb.log({
                "epoch": epoch,
                "curvature_user": curvature_info.get('c_user', 0),
                "curvature_poi": curvature_info.get('c_poi', 0),
                "curvature_class": curvature_info.get('c_class', 0),
                "curvature_time": curvature_info.get('c_time', 0),
            }, step=epoch)

    if epoch >= 100:  # 提前开始测试
        if epoch%100 == 0:
            auc, ap, top_k, acc_test = test( Z, test_label, g, friend_edge_test)

            print(f"\n{'─'*50}")
            print(f"📈 Test @Epoch {epoch}")
            print(f"   AUC: {auc:.4f}  |  AP: {ap:.4f}  |  Acc: {acc_test:.4f}")
            print(f"   Top@1/5/10/15/20: {top_k[0]:.4f}/{top_k[1]:.4f}/{top_k[2]:.4f}/{top_k[3]:.4f}/{top_k[4]:.4f}")
            # 更新所有最优指标
            if auc > best_test_auc:
                best_test_auc = auc
                best_test_ap = ap
                best_acc_test = acc_test
                best_top_k = list(top_k)
                best_epoch = epoch
                print(f"🎯 New Best! AUC={best_test_auc:.4f}, AP={best_test_ap:.4f}, Acc={best_acc_test:.4f}")
                # 保存欧几里得嵌入（用于下游任务）
                f=open(embedding_file,'wb')
                pickle.dump(Z[:args.user_number], f)
                f.close()
                # 保存双曲空间嵌入（用于可视化）
                if 'X_hyperbolic' in curvature_info:
                    hyperbolic_file = embedding_file.replace('.pkl', '_hyperbolic.pkl')
                    with open(hyperbolic_file, 'wb') as f_hyp:
                        pickle.dump(curvature_info['X_hyperbolic'][:args.user_number].detach().cpu(), f_hyp)
                    print(f"   📦 Saved hyperbolic embedding: {hyperbolic_file}")
                
                # 立即保存最优模型（版本化）
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_auc': best_test_auc,
                    'best_test_ap': best_test_ap,
                    'best_acc_test': best_acc_test,
                    'best_top_k': best_top_k,
                    'best_epoch': best_epoch,
                    'wandb_run_id': wandb_run_id,
                    'curvatures': {
                        'c_user': curvature_history['c_user'][-1] if curvature_history['c_user'] else 1.0,
                        'c_poi': curvature_history['c_poi'][-1] if curvature_history['c_poi'] else 1.0,
                        'c_class': curvature_history['c_class'][-1] if curvature_history['c_class'] else 1.0,
                        'c_time': curvature_history['c_time'][-1] if curvature_history['c_time'] else 1.0,
                    }
                }, str(best_ckpt_path))
                print(f"⭐ Best model saved to {best_ckpt_path}")
            print(f"📊 Best @Epoch {best_epoch}:")
            print(f"   AUC={best_test_auc:.4f}  AP={best_test_ap:.4f}  Acc={best_acc_test:.4f}")
            print(f"   Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}")
            print(f"{'─'*50}\n")
            # 写入评估结果到txt（格式化）
            log_line = (
                f"[Epoch {epoch:5d}] "
                f"AUC={auc:.4f} AP={ap:.4f} Acc={acc_test:.4f} "
                f"Top@1/5/10/15/20: {top_k[0]:.4f}/{top_k[1]:.4f}/{top_k[2]:.4f}/{top_k[3]:.4f}/{top_k[4]:.4f} | "
                f"Best @{best_epoch}: AUC={best_test_auc:.4f} AP={best_test_ap:.4f} Acc={best_acc_test:.4f} "
                f"Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}\n"
            )
            with open(file, 'a') as f:
                f.write(log_line)
            wandb.log({
                "epoch": epoch,
                "test_auc": auc,
                "test_ap": ap,
                "test_acc": acc_test,
                "top1": top_k[0],
                "top5": top_k[1],
                "top10": top_k[2],
                "top15": top_k[3],
                "top20": top_k[4],
                "best_test_auc": best_test_auc,
                "best_test_ap": best_test_ap,
                "best_test_acc": best_acc_test,
                "best_top1": best_top_k[0],
                "best_top5": best_top_k[1],
                "best_top10": best_top_k[2],
                "best_top15": best_top_k[3],
                "best_top20": best_top_k[4],
            }, step=epoch)

    # 保存checkpoint到版本化文件
    # 定期保存（每1000 epoch）或训练结束时
    if epoch % 1000 == 0 or epoch == target_end_epoch - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_auc': best_test_auc,
            'best_test_ap': best_test_ap,
            'best_acc_test': best_acc_test,
            'best_top_k': best_top_k,
            'best_epoch': best_epoch,
            'train_id': train_id,
            'wandb_run_id': wandb_run_id,
            'curvatures': {
                'c_user': curvature_history['c_user'][-1] if curvature_history['c_user'] else 1.0,
                'c_poi': curvature_history['c_poi'][-1] if curvature_history['c_poi'] else 1.0,
                'c_class': curvature_history['c_class'][-1] if curvature_history['c_class'] else 1.0,
                'c_time': curvature_history['c_time'][-1] if curvature_history['c_time'] else 1.0,
            }
        }, str(versioned_ckpt_path))
        print(f"💾 Checkpoint [train{train_id}] saved @epoch{epoch}: Best AUC={best_test_auc:.4f}, AP={best_test_ap:.4f}")
        # 也保存一次当前最好用户嵌入
        try:
            with open(embedding_file, 'wb') as f:
                pickle.dump(Z[:args.user_number], f)
        except Exception as e:
            print(f"[Warn] Failed to save embeddings: {e}")

print("\n" + "=" * 70)
print("🎉 Training Completed!")
print("=" * 70)
print(f"\n📊 Best Results @Epoch {best_epoch}:")
print(f"   AUC:  {best_test_auc:.4f}")
print(f"   AP:   {best_test_ap:.4f}")
print(f"   Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}")

# 写入最终总结到txt
with open(file, 'a') as f:
    f.write("\n" + "="*100 + "\n")
    f.write("📊 FINAL RESULTS\n")
    f.write("="*100 + "\n")
    f.write(f"Dataset:          {args.city.upper()}\n")
    f.write(f"Total Epochs:     {total_epochs}\n")
    f.write(f"Best Epoch:       {best_epoch}\n")
    f.write(f"Best AUC:         {best_test_auc:.4f}\n")
    f.write(f"Best AP:          {best_test_ap:.4f}\n")
    f.write(f"Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}\n")
    f.write("="*100 + "\n")
    
print(f"\n📁 Files saved:")
print(f"   • {file}")
print(f"   • {embedding_file}")
print(f"   • {versioned_ckpt_path}")
print(f"   • {best_ckpt_path}")
print(f"\n📐 Final Curvatures:")
if curvature_history['c_user']:
    print(f"  User: {curvature_history['c_user'][-1]:.4f}")
    print(f"  POI: {curvature_history['c_poi'][-1]:.4f}")
    print(f"  Class: {curvature_history['c_class'][-1]:.4f}")
    print(f"  Time: {curvature_history['c_time'][-1]:.4f}")

