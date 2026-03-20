#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文三个关键对比图：
1. PCA投影对比 - 结构可视化
2A. 聚类稳定性可视化 - 鲁棒性验证 (k=3,5,7)
2B. 聚类质量指标对比 - 固有结构分析 (k=2-15)
3. 范数分布对比 - 层级编码能力

使用方法:
    python paper_figures_three_comparisons_cn.py --city NYC
    python paper_figures_three_comparisons_cn.py --city SP
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import gaussian_kde
import seaborn as sns
from pathlib import Path
from glob import glob
import argparse

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='Generate comparison figures by city')
parser.add_argument('--city', type=str, default='NYC', help='City name (NYC, SP, BER, CHI, JK, KL)')
parser.add_argument('--euc_ep', type=int, default=8100, help='Euclidean model epoch')
parser.add_argument('--hyp_ep', type=int, default=8000, help='Hyperbolic model epoch')
args = parser.parse_args()

city = args.city.upper()
euc_ep = args.euc_ep
hyp_ep = args.hyp_ep

# ============================================================================
# 1. 加载嵌入
# ============================================================================

def load_embedding(path):
    """Load embedding from pickle file"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'embedding' in data:
        return data['embedding']
    return data

print(f"📂 Loading embeddings for {city}...")

euc_files = glob(f'./output_embedding/{city}_euclidean_ep*_train1.pkl')
if euc_files:
    euclidean_path = sorted(euc_files)[-1]
else:
    euclidean_path = f'./output_embedding/{city}_euclidean_ep{euc_ep}_train1.pkl'

hyp_files = glob(f'./output_embedding/{city}_adaptive_ep*_hyperbolic.pkl')
if not hyp_files:
    hyp_files = glob(f'./output_embedding/{city}_adaptive_ep*train*_hyperbolic.pkl')
if not hyp_files:
    hyp_files = glob(f'./output_embedding/{city}_adaptive_ep*.pkl')

if hyp_files:
    hyperbolic_path = sorted(hyp_files)[-1]
else:
    hyperbolic_path = f'./output_embedding/{city}_adaptive_ep{hyp_ep}_train4_hyperbolic.pkl'

print(f"  欧几里得: {euclidean_path}")
print(f"  双曲: {hyperbolic_path}")

X_euc = load_embedding(euclidean_path)
X_hyp = load_embedding(hyperbolic_path)

if hasattr(X_euc, 'cpu'):
    X_euc = X_euc.detach().cpu().numpy()
if isinstance(X_euc, np.ndarray) == False:
    X_euc = np.array(X_euc)
    
if hasattr(X_hyp, 'cpu'):
    X_hyp = X_hyp.detach().cpu().numpy()
if isinstance(X_hyp, np.ndarray) == False:
    X_hyp = np.array(X_hyp)

print(f"✅ 欧几里得嵌入: {X_euc.shape}")
print(f"✅ 双曲嵌入: {X_hyp.shape}")

n_users = X_euc.shape[0]

# ============================================================================
# 2. 计算范数
# ============================================================================

norms_euc = np.linalg.norm(X_euc, axis=1)
norms_hyp = np.linalg.norm(X_hyp, axis=1)

print(f"\n📊 欧几里得范数 - 均值: {norms_euc.mean():.4f}, 标准差: {norms_euc.std():.4f}")
print(f"📊 双曲范数 - 均值: {norms_hyp.mean():.4f}, 标准差: {norms_hyp.std():.4f}")

# ============================================================================
# 3. PCA投影 (2D)
# ============================================================================

print("\n🔄 计算PCA投影...")
pca_euc = PCA(n_components=2)
X_euc_2d = pca_euc.fit_transform(X_euc)

pca_hyp = PCA(n_components=2)
X_hyp_2d = pca_hyp.fit_transform(X_hyp)

# ============================================================================
# 4. 聚类稳定性 - K值为 [3, 5, 7] (for Figure 2A)
# ============================================================================

print("\n🔄 计算K-means聚类，k=3,5,7 (图2A)...")
k_values_visual = [3, 5, 7]
clustering_euc_visual = {}
clustering_hyp_visual = {}

for k in k_values_visual:
    kmeans_euc = KMeans(n_clusters=k, random_state=42, n_init=10)
    clustering_euc_visual[k] = kmeans_euc.fit_predict(X_euc)
    
    kmeans_hyp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clustering_hyp_visual[k] = kmeans_hyp.fit_predict(X_hyp)

# ============================================================================
# 5. 聚类质量指标 - K值从2到15 (for Figure 2B)
# ============================================================================

print("\n🔄 计算聚类质量指标，k=2-15 (图2B)...")
k_values_metrics = range(2, 16)

metrics_euc = {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}
metrics_hyp = {'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}

for k in k_values_metrics:
    kmeans_euc = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_euc = kmeans_euc.fit_predict(X_euc)
    
    kmeans_hyp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_hyp = kmeans_hyp.fit_predict(X_hyp)
    
    # Euclidean metrics
    metrics_euc['Silhouette'].append(silhouette_score(X_euc, labels_euc))
    metrics_euc['Davies-Bouldin'].append(davies_bouldin_score(X_euc, labels_euc))
    metrics_euc['Calinski-Harabasz'].append(calinski_harabasz_score(X_euc, labels_euc))
    
    # Hyperbolic metrics
    metrics_hyp['Silhouette'].append(silhouette_score(X_hyp, labels_hyp))
    metrics_hyp['Davies-Bouldin'].append(davies_bouldin_score(X_hyp, labels_hyp))
    metrics_hyp['Calinski-Harabasz'].append(calinski_harabasz_score(X_hyp, labels_hyp))

# ============================================================================
# 开始绘图
# ============================================================================

output_dir = Path('./paper_visualization_cn')
output_dir.mkdir(exist_ok=True)

# ========== 图1: PCA投影对比 ==========
print("\n📈 生成图1: PCA投影对比...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

norm_vmin = min(norms_euc.min(), norms_hyp.min())
norm_vmax = max(norms_euc.max(), norms_hyp.max())

scatter1 = axes[0].scatter(X_euc_2d[:, 0], X_euc_2d[:, 1], 
                           c=norms_euc, cmap='Blues', s=50, alpha=0.85, edgecolors='none',
                           vmin=norm_vmin, vmax=norm_vmax)
axes[0].set_title(f'欧式空间\n(PCA主成分1: {pca_euc.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
axes[0].set_xlabel(f'主成分1 ({pca_euc.explained_variance_ratio_[0]:.1%})', fontsize=11)
axes[0].set_ylabel(f'主成分2 ({pca_euc.explained_variance_ratio_[1]:.1%})', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle='--')
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('用户范数', fontsize=10)

scatter2 = axes[1].scatter(X_hyp_2d[:, 0], X_hyp_2d[:, 1], 
                           c=norms_hyp, cmap='Reds', s=50, alpha=0.85, edgecolors='none',
                           vmin=norm_vmin, vmax=norm_vmax)
axes[1].set_title(f'双曲空间\n(PCA主成分1: {pca_hyp.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
axes[1].set_xlabel(f'主成分1 ({pca_hyp.explained_variance_ratio_[0]:.1%})', fontsize=11)
axes[1].set_ylabel(f'主成分2 ({pca_hyp.explained_variance_ratio_[1]:.1%})', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle='--')
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('用户范数', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / f'{city}_Figure1_PCA_Projection_Comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {city}_Figure1_PCA_Projection_Comparison.png")
plt.close()

# ========== 图2A: 聚类稳定性可视化 (k=3,5,7) ==========
print("\n📈 生成图2A: 聚类稳定性可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, k in enumerate(k_values_visual):
    ax = axes[0, idx]
    scatter = ax.scatter(X_euc_2d[:, 0], X_euc_2d[:, 1], 
                        c=clustering_euc_visual[k], cmap='tab10', s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
    ax.set_title(f'欧式空间 (k={k})', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'主成分1', fontsize=10)
    ax.set_ylabel(f'主成分2', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

for idx, k in enumerate(k_values_visual):
    ax = axes[1, idx]
    scatter = ax.scatter(X_hyp_2d[:, 0], X_hyp_2d[:, 1], 
                        c=clustering_hyp_visual[k], cmap='tab10', s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
    ax.set_title(f'双曲空间 (k={k})', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'主成分1', fontsize=10)
    ax.set_ylabel(f'主成分2', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / f'{city}_Figure2A_Clustering_Stability_Visual.png', dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {city}_Figure2A_Clustering_Stability_Visual.png")
plt.close()

# ========== 图2B: 聚类质量指标对比 (k=2-15，折线图) ==========
print("\n📈 生成图2B: 聚类质量指标对比...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

k_range = list(k_values_metrics)

# Silhouette Score
ax = axes[0]
ax.plot(k_range, metrics_euc['Silhouette'], marker='o', linewidth=2.5, markersize=8, 
        label='欧式空间', color='#0066CC', alpha=0.8)
ax.plot(k_range, metrics_hyp['Silhouette'], marker='s', linewidth=2.5, markersize=8, 
        label='双曲空间', color='#CC0000', alpha=0.8)
ax.set_xlabel('聚类数 (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('轮廓系数', fontsize=12, fontweight='bold')
ax.set_title('(a) 轮廓系数 ', fontsize=11, fontweight='bold')
ax.set_xticks(k_range)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# Davies-Bouldin Index
ax = axes[1]
ax.plot(k_range, metrics_euc['Davies-Bouldin'], marker='o', linewidth=2.5, markersize=8, 
        label='欧式空间', color='#0066CC', alpha=0.8)
ax.plot(k_range, metrics_hyp['Davies-Bouldin'], marker='s', linewidth=2.5, markersize=8, 
        label='双曲空间', color='#CC0000', alpha=0.8)
ax.set_xlabel('聚类数 (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Davies-Bouldin指数', fontsize=12, fontweight='bold')
ax.set_title('(b) Davies-Bouldin指数 ', fontsize=11, fontweight='bold')
ax.set_xticks(k_range)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

# Calinski-Harabasz Score
ax = axes[2]
ax.plot(k_range, metrics_euc['Calinski-Harabasz'], marker='o', linewidth=2.5, markersize=8, 
        label='欧式空间', color='#0066CC', alpha=0.8)
ax.plot(k_range, metrics_hyp['Calinski-Harabasz'], marker='s', linewidth=2.5, markersize=8, 
        label='双曲空间', color='#CC0000', alpha=0.8)
ax.set_xlabel('聚类数 (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Calinski-Harabasz分数', fontsize=11, fontweight='bold')
ax.set_title('(c) Calinski-Harabasz分数 ', fontsize=11, fontweight='bold')
ax.set_xticks(k_range)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / f'{city}_Figure2B_Clustering_Quality_Metrics.png', dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {city}_Figure2B_Clustering_Quality_Metrics.png")
plt.close()

# ========== 图3: 范数分布对比 ==========
print("\n📈 生成图3: 范数分布对比...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
kde_euc = gaussian_kde(norms_euc)
kde_hyp = gaussian_kde(norms_hyp)
x_range = np.linspace(0, max(norms_euc.max(), norms_hyp.max()), 200)
ax.fill_between(x_range, kde_euc(x_range), alpha=0.5, label='欧式空间', color='#0066CC')
ax.fill_between(x_range, kde_hyp(x_range), alpha=0.5, label='双曲空间', color='#CC0000')
ax.set_xlabel('用户范数 ||u||', fontsize=11, fontweight='bold')
ax.set_ylabel('密度', fontsize=11, fontweight='bold')
ax.set_title('(a) 范数分布 (核密度估计)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Pareto曲线
ax = axes[1]
contrib_euc = (norms_euc / norms_euc.sum()) * 100
contrib_hyp = (norms_hyp / norms_hyp.sum()) * 100
contrib_euc_sorted = np.sort(contrib_euc)[::-1]
contrib_hyp_sorted = np.sort(contrib_hyp)[::-1]
cumsum_euc = np.cumsum(contrib_euc_sorted)
cumsum_hyp = np.cumsum(contrib_hyp_sorted)
user_pct = np.arange(1, len(contrib_euc_sorted) + 1) / len(contrib_euc_sorted) * 100

ax.plot(user_pct, cumsum_euc, linewidth=2.5, label='欧式空间', color='#0066CC', marker='o', markersize=3, alpha=0.8)
ax.plot(user_pct, cumsum_hyp, linewidth=2.5, label='双曲空间', color='#CC0000', marker='s', markersize=3, alpha=0.8)
ax.axhline(y=40, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='40%阈值')
ax.axvline(x=20, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

idx_20_euc = np.argmax(user_pct >= 20)
idx_20_hyp = np.argmax(user_pct >= 20)
val_euc_20 = cumsum_euc[idx_20_euc]
val_hyp_20 = cumsum_hyp[idx_20_hyp]

ax.text(22, val_euc_20, f'{val_euc_20:.1f}%', fontsize=9, ha='left', color='#0066CC', fontweight='bold')
ax.text(22, val_hyp_20, f'{val_hyp_20:.1f}%', fontsize=9, ha='left', color='#CC0000', fontweight='bold')

ax.set_xlabel('用户占比 (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('累计贡献率 (%)', fontsize=12, fontweight='bold')
ax.set_title('(b) 帕累托分析', fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0, 100])
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(output_dir / f'{city}_Figure3_Norm_Distribution_Comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {city}_Figure3_Norm_Distribution_Comparison.png")
plt.close()

print("\n" + "="*70)
print("✅ 所有图表已成功生成!")
print("="*70)
print(f"📁 输出目录: {output_dir.absolute()}")
print(f"🏙️  城市: {city}")
print(f"   - {city}_Figure1_PCA_Projection_Comparison.png")
print(f"   - {city}_Figure2A_Clustering_Stability_Visual.png")
print(f"   - {city}_Figure2B_Clustering_Quality_Metrics.png")
print(f"   - {city}_Figure3_Norm_Distribution_Comparison.png")
