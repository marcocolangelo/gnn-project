#%% imports 
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify tracking server
mlflow.set_tracking_uri("http://localhost:5000")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, batch in enumerate(tqdm(train_loader)):  #here train_loader is a DataLoader version specific for the graph data 
        #batch is the data for the current batch in the DataLoader
        # Use GPU
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(),                       #recall the forward function from the GNN model
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_attr.float(),
                        batch.edge_index, 
                        batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")


#%% Run the training
from mango import Tuner
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE

def run_one_training(params):
    params = params[0]
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Loading the dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv") 
        test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1] #train_dataset[0] is the first molecule in the dataset
        print(f"params: {params}")
        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True) 
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

        #stampa grandezza dataset
        print(f"Train dataset size: {len(train_dataset)}")

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)  #train_dataset[i].x.shape[1] is the number of features in the i-th molecule
        print(f"Model parameters: {model_params}")
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # < 1 increases precision, > 1 recall
        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
        
        # Start training
        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(300): 
            if early_stopping_counter <= 10: # = x * 5 
                # Training
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    
                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model 
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]

# %% Hyperparameter search
print("Running hyperparameter search...")
config = dict()
config["optimizer"] = "Bayesian"
config["num_iteration"] = 100

BEST_PARAMETERS = {
    "batch_size": 128,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "sgd_momentum": 0.8,
    "scheduler_gamma": 0.8,
    "pos_weight": 1.3,
    "model_embedding_size": 64,
    "model_attention_heads": 3,
    "model_layers": 4,
    "model_dropout_rate": 0.2,
    "model_top_k_ratio": 0.5,
    "model_top_k_every_n": 1,
    "model_dense_neurons": 256
}

# Create config using BETS_PARAMETERS
for key in BEST_PARAMETERS.keys():
    config[key] = BEST_PARAMETERS[key]

# Create a dummy model

# Call the train_one_epoch function
epoch = 0
running_loss = run_one_training([config])
