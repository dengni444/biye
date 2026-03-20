# 基于Lite-GRU嵌入和VAE增强的异构注意力神经网络

## 版本要求

| 包 | 最低版本 |
|----|---------|
| Python | 3.7+ |
| PyTorch | 1.9+ |
| DGL | 0.4+ |
| scikit-learn | 0.24+ |
| numpy | 1.19+ |

## 快速开始

```bash
python jianhua_main.py --city IST --epochs 5010 --lr 1e-4 --multihead 5 --lambda_1 1 --lambda_2 2 --lambda_3 3
```

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--city` | 数据集(IST/JK/KL/NYC/SP/TKY) | IST |
| `--epochs` | 训练轮数 | 5010 |
| `--lr` | 学习率 | 1e-4 |
| `--multihead` | 多头数 | 5 |
| `--lambda_1/2/3` | 损失权重 | 1/2/3 |
| `--cuda` | GPU索引 | 0 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 命令行参数解析 |
| `jianhua_dataset.py` | 数据加载。 |
| `model.py` | GEVEHAN模型 + 链接预测器。 |
| `utils.py` | 损失函数、评估指标、对抗扰动。 |
| jianhua_main.py` | 训练脚本 |
| `test.py` | 模型评估脚本。 |



## 模型架构

异构图 → VAE特征重构 → (异构消息传递) → 节点嵌入 → 链接预测

## 输出文件

- `pth/best_model.pth` - 最优模型权重
- `data/save_user_embedding/{city}/*.npy` - 用户嵌入向量
- `output/{city}-*.txt` - 评估结果日志
