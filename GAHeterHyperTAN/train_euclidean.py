#coding=utf-8
"""
欧式空间模型训练脚本
参考 train_adaptive_test.py 的版本管理和输出格式
"""
import pickle
import wandb
import time
import dgl
import os
import sys
import glob
import torch.nn.utils as nn_utils

# 预处理自定义参数
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
# 设置为欧式空间
args.manifold_name = 'euclidean'
args.use_adaptive = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
if args.multi_cuda==0:
    device2=device
else:
    device2 = torch.device('cuda:' + str(eval(args.cuda) + 1) if torch.cuda.is_available() else 'cpu')

now_time=time.strftime("%m%d%H%M", time.localtime())

# Define node and edge types
node_type=['user','poi','poi_class','time_point']
edge_type=['friend','check_in','trajectory']

# ========== 版本管理 ==========
specified_train_id = _train_id_value
log_base = f'output/{args.city}_euclidean_ep{args.epochs}'
resume = _resume_flag

# 确保输出目录存在
os.makedirs('output', exist_ok=True)
os.makedirs('output_embedding', exist_ok=True)
ckpt_dir = Path('checkpoints_euclidean')
ckpt_dir.mkdir(parents=True, exist_ok=True)

# ===== train_id 确定逻辑 =====
train_id = 1

if specified_train_id is not None:
    train_id = specified_train_id
    print(f"📌 [Specified] Using train_id={train_id} from --train-id parameter")
elif resume:
    existing_files = glob.glob(str(ckpt_dir / f"{args.city}_euclidean_train*.pt"))
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
    existing_files = glob.glob(str(ckpt_dir / f"{args.city}_euclidean_train*.pt"))
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
    else:
        print(f"🆕 [First] First training session: train_id={train_id}")

# ===== 更新所有输出文件路径 =====
file = f'{log_base}_train{train_id}.txt'
embedding_file = f'output_embedding/{args.city}_euclidean_ep{args.epochs}_train{train_id}.pkl'
versioned_ckpt_path = ckpt_dir / f"{args.city}_euclidean_train{train_id}.pt"
best_ckpt_path = ckpt_dir / f"{args.city}_euclidean_best_train{train_id}.pt"

print(f"\n📋 Training Configuration:")
print(f"   City: {args.city} | Epochs: {args.epochs} | Version: train{train_id}")
print(f"   Checkpoint: {versioned_ckpt_path}")
print(f"   Log file:   {file}")
print(f"   Embedding:  {embedding_file}\n")

# ===== 数据加载和初始化 =====
G, node_attr, friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,k =fetch_data(args)
model, optimizer, G = initialise(G, node_attr , args, node_type, edge_type)

# ===== 学习率调度 =====
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

# ===== 加载 checkpoint =====
start_epoch = 0
best_test_ap = 0.0
best_acc_test = 0.0
best_top_k = [0.0] * 5
best_epoch = 0
wandb_run_id = None

load_path = versioned_ckpt_path if versioned_ckpt_path.exists() else None
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
        wandb_run_id = ckpt.get('wandb_run_id', None)
        print(f"✅ Loaded checkpoint from {load_path}")
        print(f"   Start epoch: {start_epoch} | Best AUC: {best_test_auc:.4f} @ epoch {best_epoch}\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}. Start from scratch.\n")

# ===== 过滤配置中不能序列化的张量 =====
wandb_config = {}
for key, val in vars(args).items():
    try:
        if isinstance(val, torch.Tensor):
            continue
        if hasattr(val, 'is_sparse') and val.is_sparse:
            continue
        if isinstance(val, (dict, list)) and len(str(val)) > 10000:
            continue
        wandb_config[key] = val
    except:
        pass

# ===== wandb 初始化 =====
if wandb_run_id:
    wandb.init(
        id=wandb_run_id,
        resume="must",
        project="hypergraph",
        config=wandb_config,
    )
else:
    wandb.init(
        name=f"{args.city}_euclidean_train{train_id}",
        project="hypergraph",
        config=wandb_config,
    )
    wandb_run_id = wandb.run.id
    print(f"🆕 New wandb run created: {wandb_run_id}")

target_end_epoch = start_epoch + int(args.epochs)

# ===== 主训练循环 =====
for epoch in range(start_epoch, target_end_epoch):
    optimizer.zero_grad()
    model.train()
    
    # 模型前向传播
    result = model(node_attr)
    if isinstance(result, tuple):
        Z, c = result
    else:
        Z = result
        c = 1.0
    
    predic_label = F.cosine_similarity(Z[friend_edge_train_all[0]], Z[friend_edge_train_all[1]])
    loss_cross = F.binary_cross_entropy_with_logits(predic_label, friend_edge_train_all_label)
    loss_margin = margin_loss(predic_label[:k], predic_label[k:])
    con_loss = contrastive_loss(Z, g)
    loss = con_loss * args.lam_1 + loss_cross * args.lam_2 + loss_margin * args.lam_3

    loss.backward()
    nn_utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    # 学习率调度
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    # 每100轮打印一次loss
    if epoch % 100 == 0:
        loss_msg = f"Epoch: {epoch} loss_total: {loss.item():.6f} " \
                   f"loss_cross: {loss_cross.item():.6f} " \
                   f"loss_margin: {loss_margin.item():.6f} " \
                   f"loss_contrastive: {con_loss.item():.6f}"
        print(loss_msg)
        with open(file, 'a') as f:
            f.write(loss_msg + '\n')
        
        c_val = c.item() if isinstance(c, torch.Tensor) else c
        wandb.log({
            "epoch": epoch,
            "loss_total": loss.item(),
            "loss_cross": loss_cross.item(),
            "loss_margin": loss_margin.item(),
            "loss_contrastive": con_loss.item(),
            "curvature": c_val,
            "lr": optimizer.param_groups[0]["lr"]
        }, step=epoch)
        
        end_time = time.time()
        execution_time = end_time - tic_epoch
        print(f"Execution time for 100 epochs: {execution_time} seconds")
        tic_epoch = time.time()

    # 每100轮进行一次测试
    if epoch >= 100:
        if epoch % 100 == 0:
            auc, ap, top_k, acc_test = test(Z, test_label, g, friend_edge_test)

            print(f"\n{'─'*50}")
            print(f"📈 Test @Epoch {epoch}")
            print(f"   AUC: {auc:.4f}  |  AP: {ap:.4f}  |  Acc: {acc_test:.4f}")
            print(f"   Top@1/5/10/15/20: {top_k[0]:.4f}/{top_k[1]:.4f}/{top_k[2]:.4f}/{top_k[3]:.4f}/{top_k[4]:.4f}")
            
            # 更新最优指标
            if auc > best_test_auc:
                best_test_auc = auc
                best_test_ap = ap
                best_acc_test = acc_test
                best_top_k = list(top_k)
                best_epoch = epoch
                print(f"🎯 New Best! AUC={best_test_auc:.4f}, AP={best_test_ap:.4f}, Acc={best_acc_test:.4f}")
                
                # 保存最优嵌入
                with open(embedding_file, 'wb') as f:
                    pickle.dump(Z[:args.user_number], f)
                
                # 保存最优模型
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
                }, str(best_ckpt_path))
                print(f"⭐ Best model saved to {best_ckpt_path}")
            
            print(f"📊 Best @Epoch {best_epoch}:")
            print(f"   AUC={best_test_auc:.4f}  AP={best_test_ap:.4f}  Acc={best_acc_test:.4f}")
            print(f"   Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}")
            print(f"{'─'*50}\n")
            
            # 写入评估结果
            log_line = (
                f"[Epoch {epoch:5d}] "
                f"AUC={auc:.4f} AP={ap:.4f} Acc={acc_test:.4f} "
                f"Top@1/5/10/15/20: {top_k[0]:.4f}/{top_k[1]:.4f}/{top_k[2]:.4f}/{top_k[3]:.4f}/{top_k[4]:.4f} | "
                f"Best @{best_epoch}: AUC={best_test_auc:.4f} AP={best_test_ap:.4f} Acc={best_acc_test:.4f} "
                f"Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}\n"
            )
            with open(file, 'a') as f:
                f.write(log_line)
            
            c_val = c.item() if isinstance(c, torch.Tensor) else c
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
                "curvature": c_val
            }, step=epoch)

    # 定期保存 checkpoint
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
        }, str(versioned_ckpt_path))
        print(f"💾 Checkpoint [train{train_id}] saved @epoch{epoch}: Best AUC={best_test_auc:.4f}, AP={best_test_ap:.4f}")

print("\n" + "=" * 70)
print("🎉 Training Completed!")
print("=" * 70)
print(f"\n📊 Best Results @Epoch {best_epoch}:")
print(f"   AUC:  {best_test_auc:.4f}")
print(f"   AP:   {best_test_ap:.4f}")
print(f"   Acc:  {best_acc_test:.4f}")
print(f"   Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}")

# 写入最终总结
with open(file, 'a') as f:
    f.write("\n" + "="*100 + "\n")
    f.write("📊 FINAL RESULTS\n")
    f.write("="*100 + "\n")
    f.write(f"Dataset:          {args.city.upper()}\n")
    f.write(f"Model:            Euclidean (train{train_id})\n")
    f.write(f"Total Epochs:     {total_epochs}\n")
    f.write(f"Best Epoch:       {best_epoch}\n")
    f.write(f"Best AUC:         {best_test_auc:.4f}\n")
    f.write(f"Best AP:          {best_test_ap:.4f}\n")
    f.write(f"Best Acc:         {best_acc_test:.4f}\n")
    f.write(f"Top@1/5/10/15/20: {best_top_k[0]:.4f}/{best_top_k[1]:.4f}/{best_top_k[2]:.4f}/{best_top_k[3]:.4f}/{best_top_k[4]:.4f}\n")
    f.write("="*100 + "\n")

print(f"\n📁 Files saved:")
print(f"   • {file}")
print(f"   • {embedding_file}")
print(f"   • {versioned_ckpt_path}")
print(f"   • {best_ckpt_path}")

wandb.finish()
