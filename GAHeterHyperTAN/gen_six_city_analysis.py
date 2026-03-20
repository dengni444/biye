import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from collections import Counter

# Set professional style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'legend.fontsize': 12})

cities = ['NYC', 'SP', 'JK', 'KL', 'BER', 'CHI']
data_root = '/root/autodl-tmp/H3GNN-main/data'
raw_root = '/root/autodl-tmp/H3GNN-main/data_preprocess/yongliu_gowalla_data'
output_dir = '/root/autodl-tmp/H3GNN-main/paper_visualizations_v3'
os.makedirs(output_dir, exist_ok=True)

def get_degree_data(city):
    path = os.path.join(data_root, city, 'visit_list_edge_tensor.pkl')
    with open(path, 'rb') as f:
        visit_data = pickle.load(f)
    
    # Extract POI IDs (second element in the 4-tuple or values in the dict)
    poi_ids = []
    if isinstance(visit_data, dict):
        for edge_id, nodes in visit_data.items():
            # In your dataset, POI is usually the second node or we can just count all node frequencies
            # However, to be precise, let's treat the whole edge as a visit
            # Actually, most degrees follow power law in these graphs.
            for n in nodes:
                poi_ids.append(n)
    else:
        for nodes in visit_data:
            for n in nodes:
                poi_ids.append(n)
    
    counts = Counter(poi_ids)
    freqs = Counter(counts.values())
    
    # Sort for plotting
    x = sorted(freqs.keys())
    y = [freqs[k] for k in x]
    return x, y

# --- Figure 1: Degree Distribution for All 6 Cities ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

colors = sns.color_palette("husl", 6)

for i, city in enumerate(cities):
    x, y = get_degree_data(city)
    ax = axes[i]
    
    ax.scatter(x, y, color=colors[i], alpha=0.7, edgecolors='w', s=60, label=city)
    
    # Add a trend line (simple linear fit in log-log for power law)
    log_x, log_y = np.log10(x), np.log10(y)
    z = np.polyfit(log_x, log_y, 1)
    p = np.poly1d(z)
    ax.plot(x, 10**p(np.log10(x)), color='black', linestyle='--', alpha=0.5, label=f'Slope: {z[0]:.2f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Degree Distribution: {city}', fontweight='bold')
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Frequency P(k)')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_cities_degree_loglog.png'), dpi=300)
print(f"Saved: all_cities_degree_loglog.png")

# --- Figure 2: Professional Geo Heatmap for BER and CHI ---
# (Only cities with raw CSV coordinates)
city_csvs = {
    'Berlin': os.path.join(raw_root, 'Berlin/berlin_poi_incheckin_and_friend.csv'),
    'Chicago': os.path.join(raw_root, 'Chicago/chi_poi_incheckin_and_friend.csv')
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
cmap = sns.color_palette("magma", as_cmap=True)

for i, (city, path) in enumerate(city_csvs.items()):
    df = pd.read_csv(path)
    ax = axes[i]
    
    # Use hexbin with better parameters for a professional "paper" look
    hb = ax.hexbin(df['lng'], df['lat'], gridsize=60, cmap=cmap, bins='log', mincnt=1, edgecolors='none')
    ax.set_title(f'POI Geographic Density: {city}', fontsize=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    
    # Add a nice colorbar
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('log10(Count)', fontsize=14)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ber_chi_geo_density_professional.png'), dpi=300)
print(f"Saved: ber_chi_geo_density_professional.png")

