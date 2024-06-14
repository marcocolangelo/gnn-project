import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN

# Create a dummy dataset
dataset = MoleculeDataset(root="data/", filename="dummy.csv")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#parameters taken from BEST_PARAMETERS

# Create a dummy model
model = GNN(feature_size=dataset[0].x.shape[1], model_params={})
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Call the train_one_epoch function
epoch = 0
running_loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)

# Check if the running_loss is a float
assert isinstance(running_loss, float)

# Check if the running_loss is non-negative
assert running_loss >= 0.0