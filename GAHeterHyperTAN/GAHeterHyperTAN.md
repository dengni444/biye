# 基于几何自适应的异构双曲超图类型感知注意力神经网络（GAHeterHyperTAN）

在Poincaré双曲空间中进行好友链路预测  
**核心**：几何自适应曲率+动态基点对齐 + 类型感知注意力 + 对比学习

## ⚡ 快速开始

```bash
# 训练
python train_adaptive_test.py --dataset NYC --epochs 200
# 对比实验
python train_euclidean.py --dataset NYC
# 绘图
python paper_figures_three_comparisons_cn.py
```

## 📦 依赖

```
pip install torch dgl scikit-learn numpy pandas matplotlib scipy
torch>=1.9.0,<2.0
dgl>=0.6.0,<0.7
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## 📂 核心文件

**prepare.py** - 数据加载与异构超图构建（4类节点+8条边），返回DGL图和样本对。

**model_Poincare_adaptive.py** ⭐ - 多头GNN提取特征，自适应曲率映射到Poincaré双曲空间。

**train_adaptive_test.py** 🌟 - 对比学习+边界损失+曲率正则化，多城市训练脚本。

**config.py** - 模型超参数和训练参数配置。

**utils.py** - AUC/AP评估指标和模型推理函数。








