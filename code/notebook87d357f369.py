#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install kagglehub[pandas-datasets]')
get_ipython().system('pip install torch-geometric')


# In[ ]:


import kagglehub
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from kagglehub import KaggleDatasetAdapter


# # Phase 1 : Setup and Data Loading

# In[ ]:


KAGGLE_ROOT = "/kaggle/input/elliptic-data-set/"
SUBFOLDER = "elliptic_bitcoin_dataset/"
FINAL_PATH_ROOT = KAGGLE_ROOT + SUBFOLDER
DELIMITER = ","
TIME_STEP_COLUMN_NAME = "Time_step"
FEATURE_COLUMNS = ["txId", TIME_STEP_COLUMN_NAME] + [f"feature_{i}" for i in range(1,166)]


# ## 1. Loading the features file

# In[ ]:


try :
    features_file = FINAL_PATH_ROOT + "elliptic_txs_features.csv"
    print(f"Reading {features_file}")
    data = []
    with open(features_file,'r') as f:
        for line in f :
            parts = line.strip().split(DELIMITER)
            if len(parts) == 167:
                data.append(parts)
            else:
                print(f"Warning: Skipped line due to incorrect part count ({len(parts)}).")
    df_features = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    # Set the transaction ID as index
    df_features = df_features.set_index('txId')
    # Convert the time step column into an integer
    df_features[TIME_STEP_COLUMN_NAME] = pd.to_numeric(df_features[TIME_STEP_COLUMN_NAME], errors='coerce').astype(np.int64)

    TIME_STEPS = sorted(df_features[TIME_STEP_COLUMN_NAME].unique())
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# In[ ]:


df_features.head()


# ## 2. Loading the classes file

# In[ ]:


classes_file = FINAL_PATH_ROOT + "elliptic_txs_classes.csv"
df_classes = pd.read_csv(classes_file, index_col='txId')


# In[ ]:


df_classes.head()


# ## 3. Loading the Edges List

# In[ ]:


edgelist_file = FINAL_PATH_ROOT + "elliptic_txs_edgelist.csv"
df_edges = pd.read_csv(edgelist_file)


# In[ ]:


df_edges.head()


# ## 4. Data Inspection

# In[ ]:


print(f"Nodes (Features): {df_features.shape}, Labels: {df_classes.shape}, Edges: {df_edges.shape}")

print(f"\nAvailable Time Steps: {TIME_STEPS[0]} to {TIME_STEPS[-1]} (Total {len(TIME_STEPS)} steps)")


# In[ ]:


print("\n--- df_features (Transaction Nodes and Time_step) ---")
print(df_features.head()[['Time_step']].merge(df_classes, left_index=True, right_index=True, how='left'))


# # Phase 2 : Data Preprocessing and Merging

# ## 1. Merge the features and labels dataframes

# In[ ]:


# Convert df_classes index (txId) to string to match df_features' index format.
df_classes.index = df_classes.index.astype(str)

# Merges df_features and df_classes on their index (txId).
df_nodes = df_features.merge(df_classes, left_index=True, right_index=True, how='left')

# This ensures compatibility with the edge list (df_edges) later.
df_nodes.index = df_nodes.index.astype(str)

# Fills any missing labels (should now be fewer) with 'unknown' (0)
df_nodes['class'] = df_nodes['class'].fillna('unknown')
# Replaces class strings with standardized names and numeric codes
df_nodes['class'] = df_nodes['class'].replace({'unknown': 'Unknown','1': 'Illicit', '2': 'Licit'})
df_nodes['class_numeric'] = df_nodes['class'].replace({'Unknown': 0, 'Illicit': 1, 'Licit': 2}).astype(int)

# Ensures Time_step is a clean integer for proper temporal ordering
df_nodes[TIME_STEP_COLUMN_NAME] = df_nodes[TIME_STEP_COLUMN_NAME].fillna(-1).astype(np.int64)


# In[ ]:


df_nodes.head()


# ## 2. Merge the time and edge data

# In[ ]:


# Rename columns in df_edges (txId1 and txId2)
df_edges_renamed = df_edges.rename(columns={'txId1': 'Source', 'txId2': 'Target'})

# Ensure the merge keys in the edge list are also string/object type.
df_edges_renamed['Source'] = df_edges_renamed['Source'].astype(str)
df_edges_renamed['Target'] = df_edges_renamed['Target'].astype(str)


# Merge the time data onto the edge list.
df_edges_temporal = df_edges_renamed.merge(
    df_nodes[[TIME_STEP_COLUMN_NAME]],
    left_on='Source',
    right_index=True,
    how='left'
).rename(columns={TIME_STEP_COLUMN_NAME: 'Time_Step_Edge'})

# Drop edges where the source transaction's time step was -1 (unknown)
df_edges_temporal = df_edges_temporal[df_edges_temporal['Time_Step_Edge'] != -1]


# In[ ]:


df_edges_temporal.head()


# In[ ]:


TIME_STEPS = [int(t) for t in df_nodes[TIME_STEP_COLUMN_NAME].unique()]
TIME_STEPS.sort()
if TIME_STEPS and TIME_STEPS[0] == -1:
    TIME_STEPS.pop(0)

print(f"\n Data Merging Complete.")

# Select a subset of time steps for the demonstration loop (Phase 3)
ANALYSIS_STEPS = TIME_STEPS[::5]
print(f"Sampling {len(ANALYSIS_STEPS)} steps for dynamic analysis: {ANALYSIS_STEPS}")


# # Phase 3 : Dynamic Temporal Analysis Loop

# In[ ]:


# Ensure ANALYSIS_STEPS is defined from Phase 2
if 'ANALYSIS_STEPS' not in locals() and 'ANALYSIS_STEPS' not in globals():
    raise NameError("ANALYSIS_STEPS variable not found. Please run Phase 2 first.")

# --- OPTIMIZATION STEP: SAMPLE THE EDGE DATA ---
SAMPLING_RATE = 0.60 # Sample 60% of the total edges 
print(f"Optimizing: Sampling {SAMPLING_RATE*100:.0f}% of the temporal edge list...")

# Create the sampled edge list
df_edges_sampled = df_edges_temporal.sample(frac=SAMPLING_RATE, random_state=42)
print(f"Original Edges: {df_edges_temporal.shape[0]} | Sampled Edges: {df_edges_sampled.shape[0]}")


temporal_results = []
TIME_STEP_COLUMN_NAME = 'Time_step'

print(f"\nStarting Temporal Analysis for {len(ANALYSIS_STEPS)} Snapshots on Sampled Data...")

for t in ANALYSIS_STEPS:
    print(f"  -> Processing Time Step {t}...")
    
    # 1. Filter Edges for the current Time Step (Snapshot)
    # NOW WE FILTER THE SMALLER, SAMPLED DATAFRAME: df_edges_sampled
    current_edges = df_edges_sampled[df_edges_sampled['Time_Step_Edge'] == t]
    
    # 2. Build the Directed Graph (Snapshot G_t)
    G_t = nx.from_pandas_edgelist(
        current_edges, 
        source='Source', 
        target='Target', 
        create_using=nx.DiGraph()
    )
    
    # 3. Add node attributes (Class label)
    snapshot_node_ids = set(G_t.nodes())
    snapshot_nodes_data = df_nodes[df_nodes.index.isin(snapshot_node_ids)]
    node_labels = snapshot_nodes_data['class'].to_dict()
    nx.set_node_attributes(G_t, node_labels, 'label')

    # 4. Calculate Centrality Measures (The Temporal Metric)
    try:
        # PageRank complexity is O(I * E), so 10% of E means a huge speedup.
        pr_scores = nx.pagerank(G_t) 
        in_degree = dict(G_t.in_degree())
    except nx.NetworkXError:
        pr_scores = {}
        in_degree = {}
        
    # 5. Store the Results
    for node_id in snapshot_node_ids:
        if node_id in node_labels: 
            temporal_results.append({
                'Time_Step': t,
                'Node_ID': node_id,
                'PageRank': pr_scores.get(node_id, 0),
                'In_Degree': in_degree.get(node_id, 0),
                'Class': node_labels[node_id]
            })

# Convert results to a DataFrame for easy analysis and plotting
df_temporal_metrics = pd.DataFrame(temporal_results)
print("Analysis Loop Finished.")
print(f"Total metrics recorded: {df_temporal_metrics.shape[0]} transaction metrics.")

# Display a sample of the results
print("\n--- Sample of Temporal Metrics ---")
print(df_temporal_metrics.head())


# # Phase 4 : Visualization of dynamics

# In[ ]:


# df_temporal_metrics is assumed to be defined from Phase 3

# --- 1. Aggregate Metrics by Class and Time Step ---
# Calculate the average PageRank for each class at each time step.
df_plot = df_temporal_metrics.groupby(['Time_Step', 'Class'])['PageRank'].mean().reset_index()

# Filter out the 'unknown' class ('0') for a clearer comparison of labeled activity
df_plot_labeled = df_plot[df_plot['Class'] != '0']

# --- 2. Visualize the Dynamic Centrality ---
plt.figure(figsize=(12, 6))

if df_plot_labeled.empty:
    print("WARNING: Licit and Illicit samples are missing from the data. Plotting average PageRank only.")
    # Fallback plot for total network activity
    df_plot_total = df_temporal_metrics.groupby('Time_Step')['PageRank'].mean().reset_index()
    sns.lineplot(data=df_plot_total, x='Time_Step', y='PageRank', color='gray', label='Total Avg PageRank')
    
else:
    # This is the intended plot: Licit vs. Illicit
    sns.lineplot(
        data=df_plot_labeled, 
        x='Time_Step', 
        y='PageRank', 
        hue='Class',
        style='Class', 
        markers=True,
        dashes=False
    )
    plt.legend(title='Transaction Class')

plt.title('Temporal Evolution of Average PageRank by Transaction Class', fontsize=16)
plt.xlabel('Time Step (Snapshot)', fontsize=12)
plt.ylabel('Average PageRank Score (Temporal Influence)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()


# --- 3. Identify and Track the Most Central Node Dynamically ---
# Output confirmed: ID 196107869, Class: 0 (unknown)
df_top_node = df_temporal_metrics.sort_values(by='PageRank', ascending=False).iloc[0]
top_node_id = df_top_node['Node_ID']
top_node_class = df_top_node['Class']
print(f"\nMost Influential Transaction Overall: ID {top_node_id}, Class: {top_node_class}, Max PageRank: {df_top_node['PageRank']:.5f}")

# Plot the PageRank trajectory of this single node.
df_node_trajectory = df_temporal_metrics[df_temporal_metrics['Node_ID'] == top_node_id]

plt.figure(figsize=(10, 4))
plt.plot(df_node_trajectory['Time_Step'], df_node_trajectory['PageRank'], marker='o', linestyle='-', color='purple')
plt.title(f"PageRank Trajectory of Top Node (ID: {top_node_id}, Class: {top_node_class})", fontsize=14)
plt.xlabel('Time Step (Snapshot)', fontsize=12)
plt.ylabel('PageRank Score (Temporal Influence)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()


print("Visualization Phase Complete.")


# In[ ]:


# 1. Filter out the unknown class ('0')
df_labeled_metrics = df_temporal_metrics[df_temporal_metrics['Class'] != 'Unknown'].copy()

# 2. Check if both Licit and Illicit are present in the final sample
if len(df_labeled_metrics['Class'].unique()) < 2:
    print("\n Only one labeled class was found in the 60% sample. Cannot plot comparison.")
    print(f"Captured classes: {df_labeled_metrics['Class'].unique()}")
else:
    # 3. Aggregate the mean centrality measures across ALL time steps
    df_comparison = df_labeled_metrics.groupby('Class')[['PageRank', 'In_Degree']].mean().reset_index()

    # 4. Create the Bar Plot Visualization
    df_comparison_long = df_comparison.melt(id_vars='Class', 
                                            var_name='Metric', 
                                            value_name='Average Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_comparison_long,
        x='Class',
        y='Average Score',
        hue='Metric',
        palette={'PageRank': 'darkred', 'In_Degree': 'darkblue'}
    )
    plt.title('Average Centrality of Licit vs. Illicit Transactions (Aggregated)', fontsize=16)
    plt.xlabel('Transaction Class', fontsize=12)
    plt.ylabel('Average Centrality Score', fontsize=12)
    plt.grid(axis='y', linestyle='--')
    plt.show()
    
    print("Overall Class Comparison Plot Complete.")


# # Phase 5 : Snapshot Graph Visualization

# In[ ]:


# Ensure variables are available (VIS_TIME_STEP, df_nodes, df_edges_temporal, ANALYSIS_STEPS)
# Assume VIS_TIME_STEP is defined, e.g., VIS_TIME_STEP = ANALYSIS_STEPS[0]

print("\n--- Starting Snapshot Graph Visualization ---")

# Re-define necessary variables from the previous run to ensure robustness
VIS_TIME_STEP = ANALYSIS_STEPS[0]
color_map = {'Unknown': 'lightgray', 'Illicit': 'red', 'Licit': 'blue'}

current_edges = df_edges_temporal[df_edges_temporal['Time_Step_Edge'] == VIS_TIME_STEP]
active_nodes_in_snapshot = set(current_edges['Source']).union(set(current_edges['Target']))
snapshot_nodes_data = df_nodes[df_nodes.index.isin(active_nodes_in_snapshot)]

G_vis = nx.from_pandas_edgelist(
    current_edges, 
    source='Source', 
    target='Target', 
    create_using=nx.DiGraph()
)

# Prepare Node Colors (using the robust method from before)
node_colors = []
for node_id in G_vis.nodes():
    if node_id in snapshot_nodes_data.index:
        node_colors.append(color_map.get(snapshot_nodes_data.loc[node_id, 'class'], 'lightgray'))
    else:
        node_colors.append('lightgray')

# --- 1. Draw the Graph and Legend on ONE Figure ---
plt.figure(figsize=(15, 12)) 
pos = nx.spring_layout(G_vis, k=0.15, iterations=50, seed=42)

# Draw Nodes
nx.draw_networkx_nodes(
    G_vis, 
    pos, 
    node_color=node_colors, 
    node_size=50,
    alpha=0.8
)
# Draw Edges
nx.draw_networkx_edges(
    G_vis, 
    pos, 
    edge_color='gray', 
    width=0.5,
    alpha=0.3,
    arrows=True,
    arrowstyle='->',
    arrowsize=10
)

# --- 2. Create and Place the Custom Legend ---
# Note: Legend creation is done on the current, active plt.figure
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Unknown (0)', markerfacecolor='lightgray', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Illicit (1)', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Licit (2)', markerfacecolor='blue', markersize=10),
]

plt.legend(
    handles=legend_elements, 
    loc='upper right', 
    bbox_to_anchor=(1.25, 1), # Places the legend outside the main plot area
    title="Node Class"
)

# Set final plot attributes
plt.title(f'Network Snapshot at Time Step {VIS_TIME_STEP} (Nodes Colored by Class)', fontsize=18)
plt.axis('off')
plt.tight_layout()

# CRITICAL FIX: Only call plt.show() once at the end!
plt.show()

print("Snapshot Graph Visualization Complete.")


# In[ ]:


# ==============================================================================
# CENTRALITY METRIC COMPARISON
# Compares the average PageRank and In-Degree across the entire sampled network.
# ==============================================================================

# 1. Aggregate the mean centrality measures across ALL sampled nodes
# This is required because Licit/Illicit data was too sparse.
df_comparison = df_temporal_metrics[['PageRank', 'In_Degree']].mean().reset_index()
df_comparison.columns = ['Metric', 'Average Score']

# 2. Create the Bar Plot Visualization
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_comparison,
    x='Metric',
    y='Average Score',
    palette={'PageRank': 'darkred', 'In_Degree': 'darkblue'},
    hue='Metric',
    dodge=False,
)

# 3. Add Legend (Note: Seaborn automatically handles the legend if 'hue' is used)
# Calling plt.legend() here ensures the legend appears correctly based on the 'hue' argument.
plt.legend(title='Centrality Metric') 

plt.title('Average Centrality of Sampled Transactions (60% Sample)', fontsize=16)
plt.xlabel('Centrality Metric', fontsize=12)
plt.ylabel('Average Score (Across All Time Steps)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()


print("Final Centrality Metric Comparison Plot Complete.")


# # Phase 6 : Prediction using Graphs
# #### 1. GNN
# #### 1.1. Data Processing

# In[ ]:


get_ipython().system('pip install torch torch-geometric')
get_ipython().system('pip install scikit-learn')


# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# In[ ]:


# --- 1. Prepare Node Features (X) ---
df_features_only = df_nodes.iloc[:, 1:166].copy()
for col in df_features_only.columns:
    df_features_only[col] = pd.to_numeric(df_features_only[col], errors='coerce')
df_features_only = df_features_only.fillna(0.0)

x = torch.tensor(df_features_only.values, dtype=torch.float)


# In[ ]:


# --- 2. Prepare Node Labels (Y) ---
df_nodes['gnn_label'] = df_nodes['class_numeric'].replace({1: 0, 2: 1, 0: -1})
y = torch.tensor(df_nodes['gnn_label'].values, dtype=torch.long)

# --- 3. Prepare Edge Index and PyG Data Object ---
txid_to_index = {txid: i for i, txid in enumerate(df_nodes.index)}
source_indices = df_edges_temporal['Source'].map(txid_to_index).dropna().astype(int)
target_indices = df_edges_temporal['Target'].map(txid_to_index).dropna().astype(int)
edge_index = torch.tensor([source_indices.values, target_indices.values], dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)

print(f"PyG Data Object Created: {data}")
print(f"Number of Nodes: {data.num_nodes}, Number of Edges: {data.num_edges}")


# In[ ]:


# --- 5. Prepare Training / Validation / Test Masks (Temporal Split) ---

TRAIN_END_TIME = 30
VAL_END_TIME = 34

time_steps_np = df_nodes['Time_step'].values
gnn_labels_np = df_nodes['gnn_label'].values

# Training: labeled & early time
train_idx = np.where(
    (time_steps_np <= TRAIN_END_TIME) & (gnn_labels_np >= 0)
)[0]

# Validation: labeled & mid time
val_idx = np.where(
    (time_steps_np > TRAIN_END_TIME) &
    (time_steps_np <= VAL_END_TIME) &
    (gnn_labels_np >= 0)
)[0]

# Test: labeled & future time
test_idx = np.where(
    (time_steps_np > VAL_END_TIME) & (gnn_labels_np >= 0)
)[0]

# Create boolean masks
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx]     = True
data.test_mask[test_idx]   = True

print(f"Train nodes: {data.train_mask.sum().item()}")
print(f"Val nodes:   {data.val_mask.sum().item()}")
print(f"Test nodes:  {data.test_mask.sum().item()}")


# ### 1.1.2. Model Definition (GNN)

# In[ ]:


from torch_geometric.nn import GCNConv
import torch.nn as nn

class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        # 1. First Graph Convolutional Layer
        self.conv1 = GCNConv(num_features, hidden_channels)
        # 2. Second Graph Convolutional Layer
        self.conv2 = GCNConv(hidden_channels, num_classes)
        # 3. Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        # 1. Pass through Conv1, ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # 2. Pass through Conv2 (output layer)
        x = self.conv2(x, edge_index)
        # The output does not use a final activation (like softmax) because
        # the CrossEntropyLoss function handles it internally.
        return x


# In[ ]:


NUM_FEATURES = data.num_features # 167 features + time step
NUM_CLASSES = 2                 # Illicit (0) or Licit (1)
HIDDEN_CHANNELS = 32

model = GNNModel(NUM_FEATURES, HIDDEN_CHANNELS, NUM_CLASSES)
print("\nGNN Model Architecture:")
print(model)


# ### 1.1.3. Model Training

# In[ ]:


from sklearn.metrics import f1_score, accuracy_score

# Define Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward pass
    out = model(data.x, data.edge_index)
    
    # 2. Compute Loss only on the training nodes (Licit/Illicit)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 3. Backward pass and optimization
    loss.backward()
    optimizer.step()
    return loss.item()


# In[ ]:


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    
    # 1. Get predictions for test set
    pred = out.argmax(dim=1)
    
    # 2. Compute metrics
    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = pred[data.test_mask].cpu().numpy()
    
    acc = accuracy_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred, average='binary', zero_division=0) # Added zero_division=0
    
    return acc, f1


# In[ ]:


from sklearn.metrics import f1_score, precision_score, recall_score

num_epochs = 200
best_val_f1 = 0.0
best_epoch = -1

history = {
    "train_loss": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": []
}

for epoch in range(1, num_epochs + 1):
    # =====================
    # Train
    # =====================
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    history["train_loss"].append(loss.item())

    # =====================
    # Validate
    # =====================
    model.eval()
    with torch.no_grad():
        val_logits = out[data.val_mask]
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_true = data.y[data.val_mask].cpu().numpy()

        val_f1 = f1_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds)

        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)

        # Model selection ONLY on validation
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), "best_gnn_model.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"P: {val_precision:.4f} | "
            f"R: {val_recall:.4f}"
        )

print(f"\n Best model selected at epoch {best_epoch} (Val F1 = {best_val_f1:.4f})")


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# Load best model
model.load_state_dict(torch.load("best_gnn_model.pt"))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()

print("\n=== FINAL TEST PERFORMANCE ===")
print(classification_report(
    test_true,
    test_preds,
    target_names=["Illicit", "Licit"]
))


# In[ ]:


from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(test_true, test_preds)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Illicit", "Licit"],
    yticklabels=["Illicit", "Licit"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()


# ### Testing a hypothesis 

# In[ ]:


# Count training labels
train_labels = data.y[data.train_mask].cpu().numpy()

num_illicit = (train_labels == 0).sum()
num_licit = (train_labels == 1).sum()

# Inverse frequency weighting
w_illicit = num_licit / (num_illicit + num_licit)
w_licit = num_illicit / (num_illicit + num_licit)

class_weights = torch.tensor(
    [w_illicit, w_licit],
    dtype=torch.float,
    device=data.x.device
)

print("Class weights:", class_weights)


# In[ ]:


model1 = GNNModel(NUM_FEATURES, HIDDEN_CHANNELS, NUM_CLASSES)


# In[ ]:


# Define Optimizer and Loss Function
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def train():
    model1.train()
    optimizer.zero_grad()
    
    # 1. Forward pass
    out = model1(data.x, data.edge_index)
    
    # 2. Compute Loss only on the training nodes (Licit/Illicit)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 3. Backward pass and optimization
    loss.backward()
    optimizer.step()
    return loss.item()


# In[ ]:


@torch.no_grad()
def test():
    model1.eval()
    out = model1(data.x, data.edge_index)
    
    # 1. Get predictions for test set
    pred = out.argmax(dim=1)
    
    # 2. Compute metrics
    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred = pred[data.test_mask].cpu().numpy()
    
    acc = accuracy_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred, average='binary', zero_division=0) # Added zero_division=0
    
    return acc, f1


# In[ ]:


criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

num_epochs = 200
best_val_f1 = 0.0
best_epoch = -1

for epoch in range(1, num_epochs + 1):

    # ===== TRAIN =====
    model1.train()
    optimizer.zero_grad()

    out = model1(data.x, data.edge_index)
    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    # ===== VALIDATE =====
    model1.eval()
    with torch.no_grad():
        val_logits = out[data.val_mask]
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_true = data.y[data.val_mask].cpu().numpy()

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_recall = recall_score(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model1.state_dict(), "best_gnn_model_weighted.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"P: {val_precision:.4f} | "
            f"R: {val_recall:.4f}"
        )

print(f"\n Best model at epoch {best_epoch} (Val F1 = {best_val_f1:.4f})")


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

model1.load_state_dict(torch.load("best_gnn_model_weighted.pt"))
model1.eval()

with torch.no_grad():
    out = model1(data.x, data.edge_index)
    test_preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()

print(classification_report(
    test_true,
    test_preds,
    target_names=["Illicit", "Licit"]
))


# In[ ]:


from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(test_true, test_preds)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Illicit", "Licit"],
    yticklabels=["Illicit", "Licit"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()


# In[ ]:


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# In[ ]:


model = GNNModel(NUM_FEATURES, HIDDEN_CHANNELS, NUM_CLASSES)
model = model.to(data.x.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005,
    weight_decay=5e-4
)

criterion = FocalLoss(alpha=0.75, gamma=2.0)


# In[ ]:


num_epochs = 200
best_val_f1 = 0.0
best_epoch = -1

for epoch in range(1, num_epochs + 1):

    # ===== TRAIN =====
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    train_loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    train_loss.backward()
    optimizer.step()

    # ===== VALIDATE =====
    model.eval()
    with torch.no_grad():
        val_logits = out[data.val_mask]
        val_probs = torch.softmax(val_logits, dim=1)[:, 0]  # Illicit prob
        val_true = data.y[data.val_mask].cpu().numpy()

        # Default threshold 0.5 during training
        val_preds = (val_probs > 0.5).long().cpu().numpy()

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_recall = recall_score(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), "best_gnn_focal.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_loss.item():.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"P: {val_precision:.4f} | "
            f"R: {val_recall:.4f}"
        )

print(f"\n Best model at epoch {best_epoch} (Val F1 = {best_val_f1:.4f})")


# In[ ]:


model.load_state_dict(torch.load("best_gnn_focal.pt"))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    val_probs = torch.softmax(out[data.val_mask], dim=1)[:, 0].cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()

thresholds = np.arange(0.05, 0.95, 0.05)
best_thr = 0.5
best_thr_f1 = 0.0

for thr in thresholds:
    preds = (val_probs > thr).astype(int)
    f1 = f1_score(val_true, preds, zero_division=0)

    if f1 > best_thr_f1:
        best_thr_f1 = f1
        best_thr = thr

print(f"\n Best threshold from validation = {best_thr:.2f}")
print(f" Validation F1 @ threshold = {best_thr_f1:.4f}")


# In[ ]:


with torch.no_grad():
    test_probs = torch.softmax(out[data.test_mask], dim=1)[:, 0].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()

test_preds = (test_probs > best_thr).astype(int)

print("\n=== FINAL TEST PERFORMANCE (Focal + Threshold) ===")
print(classification_report(
    test_true,
    test_preds,
    target_names=["Illicit", "Licit"],
    zero_division=0
))


# In[ ]:


from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(test_true, test_preds)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Illicit", "Licit"],
    yticklabels=["Illicit", "Licit"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()


# #### We should retain the GNN model saved at best_gnn_model_weighted.pt

# In[ ]:


RANDOM_STATE = 42
NUM_EPOCHS = 100


# In[ ]:


def set_seed_for_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # For single-GPU.
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_seed_for_numpy(seed):
    np.random.seed(seed) 
    
def set_seed_for_random(seed):
    random.seed(seed)  


# In[ ]:


import random
set_seed_for_torch(RANDOM_STATE)
set_seed_for_numpy(RANDOM_STATE)
set_seed_for_random(RANDOM_STATE)


# In[ ]:


# ------------------------------------------- #
# Training, Evaluation and prediction methods #
# ------------------------------------------- #

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        
        correct = (pred[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()

        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return metrics

def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    return pred

def predict_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        probabilities = torch.exp(out)
    return probabilities


# In[ ]:


def train_gnn(num_epochs, data, model, optimizer, criterion):
    
    train_losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []

    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    # ----- #
    # Train #
    # ----- #

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], 
                         data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- Calculate training metrics ---
        pred_train = out[data.train_mask].argmax(dim=1)
        correct_train = (pred_train == data.y[data.train_mask]).sum()
        train_acc = int(correct_train) / int(data.train_mask.sum())
        train_accuracies.append(train_acc)

        y_true_train = data.y[data.train_mask].cpu().numpy()
        y_pred_train = pred_train.cpu().numpy()

        train_prec = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_rec = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_f1 = f1_score(y_true_train, y_pred_train, average='weighted', zero_division=0)

        train_precisions.append(train_prec)
        train_recalls.append(train_rec)
        train_f1_scores.append(train_f1)
        train_losses.append(loss.item())

        # --- Validate and calculate validation metrics ---
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred_val = out[data.val_mask].argmax(dim=1)
            correct_val = (pred_val == data.y[data.val_mask]).sum()
            val_acc = int(correct_val) / int(data.val_mask.sum())
            val_accuracies.append(val_acc)

            y_true_val = data.y[data.val_mask].cpu().numpy()
            y_pred_val = pred_val.cpu().numpy()

            val_prec = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
            val_rec = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
            val_f1 = f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)

            val_precisions.append(val_prec)
            val_recalls.append(val_rec)
            val_f1_scores.append(val_f1)

        if epoch % 10 == 0:        
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train - Acc: {train_acc:.4f} - Prec: {train_prec:.4f} - Rec: {train_rec:.4f} - F1: {train_f1:.4f}')
            print(f'                         Val   - Acc: {val_acc:.4f} - Prec: {val_prec:.4f} - Rec: {val_rec:.4f} - F1: {val_f1:.4f}')        

    return {
        'train': {
            'losses': train_losses,
            'accuracies': train_accuracies,
            'precisions': train_precisions,
            'recalls': train_recalls,
            'f1_scores': train_f1_scores,
        },
        'val': {
            'accuracies': val_accuracies,
            'precisions': val_precisions,
            'recalls': val_recalls,
            'f1_scores': val_f1_scores,            
        }
    }


# In[ ]:


elliptic_txs_features = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
elliptic_txs_classes = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
elliptic_txs_edgelist = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

elliptic_txs_features.columns = ['txId','time_step'] + [f'V{i}' for i in range(1, 166)]

print(f"""Shapes
{4*' '}Features : {elliptic_txs_features.shape[0]:8,} (rows)  {elliptic_txs_features.shape[1]:4,} (cols)
{4*' '}Classes  : {elliptic_txs_classes.shape[0]:8,} (rows)  {elliptic_txs_classes.shape[1]:4,} (cols)
{4*' '}Edgelist : {elliptic_txs_edgelist.shape[0]:8,} (rows)  {elliptic_txs_edgelist.shape[1]:4,} (cols)
""")


# In[ ]:


G = nx.from_pandas_edgelist(elliptic_txs_edgelist, 'txId1', 'txId2')


# In[ ]:


metrics_per_gnn = {
    'gcn': {
        'val': {
            'precisions': [],
            'probas': [],            
        },
        'test': {
            'licit': {
                'probas': []                
            },
            'illicit': {
                'probas': []                
            }, 
        }
    },
    'gat': {
        'val': {
            'precisions': [],
            'probas': [],            
        },
        'test': {
            'licit': {
                'probas': []                
            },
            'illicit': {
                'probas': []                
            }, 
        }
    },
    'gin': {
        'val': {
            'precisions': [],
            'probas': [],           
        },
        'test': {
            'licit': {
                'probas': []                
            },
            'illicit': {
                'probas': []                
            },            
        }
    }    
}


# In[ ]:


num_edges = elliptic_txs_edgelist.shape[0]
num_nodes = elliptic_txs_features.shape[0]

print(f'Number of edges in the graph: {num_edges:8,}')
print(f'Number of nodes in the graph: {num_nodes:8,}')


# In[ ]:


# --------------------------------------------------------- #
# Create mapping with txId as key and actual index as value #
# --------------------------------------------------------- #

tx_id_mapping = {tx_id: idx for idx, tx_id in enumerate(elliptic_txs_features['txId'])}

edges_with_features = elliptic_txs_edgelist[elliptic_txs_edgelist['txId1'].isin(list(tx_id_mapping.keys()))\
                                          & elliptic_txs_edgelist['txId2'].isin(list(tx_id_mapping.keys()))]

edges_with_features['Id1'] = edges_with_features['txId1'].map(tx_id_mapping)
edges_with_features['Id2'] = edges_with_features['txId2'].map(tx_id_mapping)

edges_with_features


# In[ ]:


edge_index = torch.tensor(edges_with_features[['Id1', 'Id2']].values.T, dtype=torch.long)
edge_index


# In[ ]:


# ------------------------------------- #
# Save node features in suitable format #
# ------------------------------------- #

node_features = torch.tensor(elliptic_txs_features.drop(columns=['txId']).values, 
                             dtype=torch.float)
print(node_features.shape)
node_features


# In[ ]:


elliptic_txs_classes['class'].value_counts()


# In[ ]:


# ------------------------ #
# Labelencode target class #
# ------------------------ #
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
class_labels = le.fit_transform(elliptic_txs_classes['class'])
node_labels = torch.tensor(class_labels, dtype=torch.long)
original_labels = le.inverse_transform(class_labels)

print(original_labels)
print(class_labels)
print(node_labels)


# In[ ]:


print(le.inverse_transform([0])) # illicit
print(le.inverse_transform([1])) # licit 
print(le.inverse_transform([2])) # unknown


# In[ ]:


# ------------------------------------ #
# Create pytorch geometric Data object #
# ------------------------------------ #

data = Data(x=node_features, 
            edge_index=edge_index, 
            y=node_labels)


# In[ ]:


known_mask   = (data.y == 0) | (data.y == 1)  # Only nodes with known labels licit or illicit
unknown_mask = data.y == 2 


# In[ ]:


# ------------------------------------------------ #
# Define size for Training, Validation and Testing #
# ------------------------------------------------ #

num_known_nodes = known_mask.sum().item()
permutations = torch.randperm(num_known_nodes)
train_size = int(0.8 * num_known_nodes)
val_size = int(0.1 * num_known_nodes)
test_size = num_known_nodes - train_size - val_size

total = np.sum([train_size, val_size, test_size])

print(f"""Number of observations per split
    Training   : {train_size:10,} ({100*train_size/total:0.2f} %)
    Validation : {val_size:10,} ({100*val_size/total:0.2f} %)
    Testing    : {test_size:10,} ({100*test_size/total:0.2f} %)
""")


# In[ ]:


# ----------------------------------------------- #
# Create mask for the indices of Train, Val, Test #
# ----------------------------------------------- #

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_indices = known_mask.nonzero(as_tuple=True)[0][permutations[:train_size]]
val_indices = known_mask.nonzero(as_tuple=True)[0][permutations[train_size:train_size + val_size]]
test_indices = known_mask.nonzero(as_tuple=True)[0][permutations[train_size + val_size:]]

data.train_mask[train_indices] = True
data.val_mask[val_indices] = True
data.test_mask[test_indices] = True

data.train_mask


# In[ ]:


# -------------------------- #
# Statistics of the datasets #
# -------------------------- #

train_licit, train_illicit = (data.y[data.train_mask] == 1).sum().item(), (data.y[data.train_mask] == 0).sum().item()
val_licit, val_illicit = (data.y[data.val_mask] == 1).sum().item(), (data.y[data.val_mask] == 0).sum().item()
test_licit, test_illicit = (data.y[data.test_mask] == 1).sum().item(), (data.y[data.test_mask] == 0).sum().item()

# Calculate total counts.
train_total = train_licit + train_illicit
val_total = val_licit + val_illicit
test_total = test_licit + test_illicit

# Calculate percentages.
train_licit_pct = (train_licit / train_total) * 100
train_illicit_pct = (train_illicit / train_total) * 100
val_licit_pct = (val_licit / val_total) * 100
val_illicit_pct = (val_illicit / val_total) * 100
test_licit_pct = (test_licit / test_total) * 100
test_illicit_pct = (test_illicit / test_total) * 100

pd.DataFrame({
    'Set': ['Training', 'Validation', 'Testing'],
    'Total Count': [train_total, val_total, test_total],
    'Licit': [train_licit, val_licit, test_licit],
    'Licit (%)': [train_licit_pct, val_licit_pct, test_licit_pct],
    'Illicit': [train_illicit, val_illicit, test_illicit],
    'Illicit (%)': [train_illicit_pct, val_illicit_pct, test_illicit_pct]
})


# In[ ]:


mapped_classes = np.array(['illicit', 'licit'])


# ## 1. GCN

# In[ ]:


# -------------- #
# Define the GCN #
# -------------- #

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ---------- #
# Initialize #
# ---------- #

model = GCN(num_node_features=data.num_features, num_classes=len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01, 
                             weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()  # Since we have a multiclass classification problem.


# In[ ]:


# ----- #
# Train #
# ----- #

train_val_metrics = train_gnn(NUM_EPOCHS, 
                              data, 
                              model, 
                              optimizer, 
                              criterion)

metrics_per_gnn['gcn']['val']['precisions'] = train_val_metrics['val']['precisions']


# In[ ]:


# -------- #
# Evaluate #
# -------- #

model.eval()
with torch.no_grad():
    
    test_metrics = evaluate(model, data, data.test_mask)
    test_acc = test_metrics.get('accuracy')
    test_prec = test_metrics.get('precision')
    test_rec = test_metrics.get('recall')
    test_f1 = test_metrics.get('f1_score')

    print(f'Test Acc: {test_acc:.4f} - Prec: {test_prec:.4f} - Rec: {test_rec:.4f} - F1: {test_f1:.4f}')


# In[ ]:


train_pred = predict(model, data)[data.train_mask]
test_pred = predict(model, data)[data.test_mask]


# In[ ]:


print(le.inverse_transform([0])) # illicit
print(le.inverse_transform([1])) # licit 
print(le.inverse_transform([2])) # unknown


# In[ ]:


#--- Classification report ---
print("Classification Report")
print("=====================\n")

# Train.
y_true_train = data.y[data.train_mask].cpu().numpy()
y_pred_train = train_pred.cpu().numpy()

report_train = classification_report(y_true_train, y_pred_train, target_names=mapped_classes)

print(f"{4*' '}TRAIN")
print("---------")
print(report_train)

# Test.
y_true_test = data.y[data.test_mask].cpu().numpy()
y_pred_test = test_pred.cpu().numpy()

report_test = classification_report(y_true_test, y_pred_test, target_names=mapped_classes)

print(f"{4*' '}TEST")
print("--------")
print(report_test)


# In[ ]:


cm = confusion_matrix(y_true_test, y_pred_test)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Illicit", "Licit"],
    yticklabels=["Illicit", "Licit"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()


# In[ ]:


torch.save(model.state_dict(), "gcn_model_weights.pt")


# ## 2. GAT

# In[ ]:


# -------------- #
# Define the GAT #
# -------------- #
from torch_geometric.nn import GATConv  
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(8 * num_heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ---------- #
# Initialize #
# ---------- #

model = GAT(num_node_features=data.num_features, num_classes=len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()  # Since we have a multiclass classification problem.



# In[ ]:


# ----- #
# Train #
# ----- #

train_val_metrics = train_gnn(NUM_EPOCHS, 
                              data, 
                              model, 
                              optimizer, 
                              criterion)

metrics_per_gnn['gat']['val']['precisions'] = train_val_metrics['val']['precisions']


# In[ ]:


# -------- #
# Evaluate #
# -------- #

model.eval()
with torch.no_grad():
    
    test_metrics = evaluate(model, data, data.test_mask)
    test_prec = test_metrics.get('precision')
    test_rec = test_metrics.get('recall')
    test_f1 = test_metrics.get('f1_score')

    print(f'Test Acc: {test_acc:.4f} - Prec: {test_prec:.4f} - Rec: {test_rec:.4f} - F1: {test_f1:.4f}')


# In[ ]:


test_pred = predict(model, data)[data.test_mask]

# --- Confusion matrix ---
cm = confusion_matrix(data.y[data.test_mask].cpu(), test_pred.cpu())

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

# Annotate each cell with the percentage of that row.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\nGraph Attention Network (GAT)')
plt.show()


# In[ ]:


print(classification_report(data.y[data.test_mask].cpu(), test_pred, target_names=mapped_classes))


# In[ ]:


torch.save(model.state_dict(), "gat_model_weights.pt")


# ## 3. GIN

# In[ ]:


# -------------- #
# Define the GIN #
# -------------- #7

from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        
        # 1st GIN layer.
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        self.conv1 = GINConv(nn1)
        
        # 2nd GIN layer.
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        self.conv2 = GINConv(nn2)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
#         x = global_add_pool(x, batch)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

# ---------- #
# Initialize #
# ---------- #

model = GIN(num_node_features=data.num_features, num_classes=len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()

# # Handle the class imbalance
# class_counts = torch.bincount(data.y)
# class_weights = 1. / class_counts.float()
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


# In[ ]:


# ----- #
# Train #
# ----- #

train_val_metrics = train_gnn(NUM_EPOCHS, 
                              data, 
                              model, 
                              optimizer, 
                              criterion)

metrics_per_gnn['gin']['val']['precisions'] = train_val_metrics['val']['precisions']


# In[ ]:


test_pred = predict(model, data)[data.test_mask]

# --- Confusion matrix ---
cm = confusion_matrix(data.y[data.test_mask].cpu(), test_pred.cpu())

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

# Annotate each cell with the percentage of that row.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\nGraph Isomorphism Network (GIN)')
plt.show()


# In[ ]:


print(classification_report(data.y[data.test_mask].cpu(), test_pred, target_names=mapped_classes))


# In[ ]:


torch.save(model.state_dict(), "gin_model_weights.pt")


# ## 4. GraphSAGE

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# In[ ]:


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
    
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2 + h1) 
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
    
        return self.lin(h2)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=128,
    out_channels=2,          # binary classification
    dropout=0.5
).to(device)

data = data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,      
    weight_decay=5e-4
)
from collections import Counter

y_train = data.y[data.train_mask].cpu().numpy()
class_counts = Counter(y_train)

# Inverse frequency weighting
w_illicit = 1.0 / class_counts[0]
w_licit = 1.0 / class_counts[1]

weights = torch.tensor([w_illicit, w_licit], device=device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)


# In[ ]:


num_epochs = 100
best_val_f1 = 0.0
best_epoch = -1

for epoch in range(1, num_epochs + 1):

    # ===== TRAIN =====
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    train_loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    train_loss.backward()
    optimizer.step()

    # ===== VALIDATE =====
    model.eval()
    with torch.no_grad():
        val_logits = out[data.val_mask]
        val_probs = torch.softmax(val_logits, dim=1)[:, 0]  # Illicit prob
        val_true = data.y[data.val_mask].cpu().numpy()

        val_preds = (val_probs > 0.2).long().cpu().numpy()

        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_recall = recall_score(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), "best_graphsage.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_loss.item():.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"P: {val_precision:.4f} | "
            f"R: {val_recall:.4f}"
        )

print(f"\n Best GraphSAGE @ epoch {best_epoch} (Val F1 = {best_val_f1:.4f})")


# In[ ]:


model.load_state_dict(torch.load("best_graphsage.pt"))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    val_probs = torch.softmax(out[data.val_mask], dim=1)[:, 0].cpu().numpy()
    val_true = data.y[data.val_mask].cpu().numpy()

thresholds = np.arange(0.05, 0.95, 0.05)
best_thr = 0.5
best_f1 = 0.0

for thr in thresholds:
    preds = (val_probs > thr).astype(int)
    f1 = f1_score(val_true, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best threshold = {best_thr:.2f}, Val F1 = {best_f1:.4f}")


# In[ ]:


with torch.no_grad():
    test_probs = torch.softmax(out[data.test_mask], dim=1)[:, 0].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()

test_preds = (test_probs > best_thr).astype(int)

print("\n=== FINAL TEST PERFORMANCE (GraphSAGE) ===")
print(classification_report(
    test_true,
    test_preds,
    target_names=["Illicit", "Licit"],
    zero_division=0
))

cm = confusion_matrix(test_true, test_preds)
fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

# Annotate each cell with the percentage of that row.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\nGraphSAGE')
plt.show()


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# GraphSAGE model
# ------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index) + h1)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        return self.lin(h2)

# ------------------------
# Prepare model, optimizer, loss
# ------------------------
model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=128,
    out_channels=2,
    dropout=0.5
).to(device)

data = data.to(device)

# Calculate class weights from training set
y_train = data.y[data.train_mask].cpu().numpy()
class_counts = Counter(y_train)
weights = torch.tensor([
    1.0 / class_counts[0],  # Illicit
    1.0 / class_counts[1]   # Licit
], device=device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# ------------------------
# Training loop
# ------------------------
num_epochs = 100
best_val_f1 = 0.0
best_epoch = -1
best_model_state = None

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = out[data.val_mask]
        val_probs = torch.softmax(val_logits, dim=1)[:, 0].cpu().numpy()  # Illicit probability
        val_true = data.y[data.val_mask].cpu().numpy()

    # Tune threshold
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_thr = 0.5
    best_thr_f1 = 0.0
    for thr in thresholds:
        val_preds = (val_probs > thr).astype(int)
        f1 = f1_score(val_true, val_preds, zero_division=0)
        if f1 > best_thr_f1:
            best_thr_f1 = f1
            best_thr = thr

    # Save best model by val F1
    if best_thr_f1 > best_val_f1:
        best_val_f1 = best_thr_f1
        best_epoch = epoch
        best_model_state = model.state_dict()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {best_thr_f1:.4f} | Threshold: {best_thr:.2f}")

print(f"\nBest GraphSAGE @ epoch {best_epoch} | Val F1: {best_val_f1:.4f} | Threshold: {best_thr:.2f}")

# ------------------------
# Testing
# ------------------------
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_probs = torch.softmax(out[data.test_mask], dim=1)[:, 0].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    test_preds = (test_probs > best_thr).astype(int)

print("\n=== FINAL TEST PERFORMANCE (GraphSAGE) ===")
print(classification_report(test_true, test_preds, target_names=["Illicit", "Licit"], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(test_true, test_preds)
print("Confusion Matrix:\n", cm)


# # Phase 7 : Prediction using Traditional ML Algorithms

# ## 1. Logistic Regression

# In[ ]:


# =============================================================================
# LOGISTIC REGRESSION USING GNN DATA
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. Extract features and labels from PyG Data
X = data.x.cpu().numpy()   # [num_nodes, num_features]
y = data.y.cpu().numpy()   # [num_nodes]

# 2. Split using the same masks
X_train = X[data.train_mask.cpu().numpy()]
y_train = y[data.train_mask.cpu().numpy()]

X_val = X[data.val_mask.cpu().numpy()]
y_val = y[data.val_mask.cpu().numpy()]

X_test = X[data.test_mask.cpu().numpy()]
y_test = y[data.test_mask.cpu().numpy()]

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Train logistic regression
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

print("Training Logistic Regression...")
lr_model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = lr_model.predict(X_test_scaled)
y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (Licit)

# 6. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_proba)

# 7. Display results
print("\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# 9. Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Illicit', 'Licit']))
fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

# Annotate each cell with the percentage of that row.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\n Logistic Regression')
plt.show()
print("\nLogistic Regression completed!")


# ## 2. Random Forest

# In[ ]:


# =============================================================================
# RANDOM FOREST CLASSIFIER USING GNN DATA
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

# 1. Train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

print("\nTraining Random Forest...")
rf_model.fit(X_train_scaled, y_train)

# 2. Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (Licit)

# 3. Calculate metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, pos_label=1)
recall_rf = recall_score(y_test, y_pred_rf, pos_label=1)
f1_rf = f1_score(y_test, y_pred_rf, pos_label=1)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

# 4. Display results
print("\nRandom Forest Performance:")
print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1-Score:  {f1_rf:.4f}")
print(f"ROC-AUC:   {roc_auc_rf:.4f}")

# 5. Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(f"[[TN={cm_rf[0,0]}  FP={cm_rf[0,1]}]")
print(f" [FN={cm_rf[1,0]}  TP={cm_rf[1,1]}]]")

# 6. Feature importance
importances = rf_model.feature_importances_
top_indices = np.argsort(importances)[-5:][::-1]
print("\nTop 5 most important features:")
for i, idx in enumerate(top_indices, 1):
    print(f" {i}. Feature_{idx+1}: {importances[idx]:.4f}")

print("\nRandom Forest completed!")


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm_rf, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
cm_normalized = cm_rf.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\n Random Forest')
plt.show()


# ## 3. Support Vector Machine (SVM)

# In[ ]:


# =============================================================================
# SUPPORT VECTOR MACHINE (SVM) USING GNN DATA
# =============================================================================

from sklearn.svm import SVC

# 1. Train model - Using RBF kernel
svm_model = SVC(
    kernel='rbf',          # Radial Basis Function kernel
    C=1.0,                 # Regularization parameter
    probability=True,      # Enable probability estimates
    random_state=42
)

print("\nTraining SVM (this may take a minute)...")
svm_model.fit(X_train_scaled, y_train)
# 2. Make predictions
y_pred_svm = svm_model.predict(X_test_scaled)
y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (Licit)

# 3. Calculate metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, pos_label=1)
recall_svm = recall_score(y_test, y_pred_svm, pos_label=1)
f1_svm = f1_score(y_test, y_pred_svm, pos_label=1)
roc_auc_svm = roc_auc_score(y_test, y_proba_svm)

# 4. Display results
print("\nSVM Performance:")
print(f"Accuracy:  {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall:    {recall_svm:.4f}")
print(f"F1-Score:  {f1_svm:.4f}")
print(f"ROC-AUC:   {roc_auc_svm:.4f}")

# 5. Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("\nConfusion Matrix:")
print(f"[[TN={cm_svm[0,0]}  FP={cm_svm[0,1]}]")
print(f" [FN={cm_svm[1,0]}  TP={cm_svm[1,1]}]]")

print("\nSVM completed!")


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(cm_svm, annot=True, fmt='g', cmap=plt.cm.Greens, 
            annot_kws={'size': 15}, 
            xticklabels=mapped_classes,
            yticklabels=mapped_classes, 
            linecolor='black', linewidth=0.5,
            ax=ax)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
cm_normalized = cm_svm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_normalized[i, j] * 100
        text = f'\n({percentage:.1f}%)'
        color = 'white' if percentage > 95 else 'black'
        ax.text(j + 0.5, i + 0.6, text,
                ha='center', va='center', fontsize=10, color=color)

plt.title('Confusion Matrix\n SVM')
plt.show()


# In[ ]:




