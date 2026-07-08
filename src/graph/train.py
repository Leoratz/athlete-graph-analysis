import torch
import torch.nn as nn
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))

from skeleton_graph import EDGES
from gcn_model import SkeletonGCN

def prepare_data():
    normal = np.load("data/normal_running.npy")
    risky  = np.load("data/risky_running.npy")

    # Normaliser les données entre 0 et 1
    all_data = np.concatenate([normal, risky], axis=0)
    mean = all_data.mean()
    std  = all_data.std()
    normal = (normal - mean) / std
    risky  = (risky  - mean) / std

    X = np.concatenate([normal, risky], axis=0)
    y = np.array([0] * 120 + [1] * 120)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"✅ Données prêtes :")
    print(f"   Train : {len(X_train)} frames")
    print(f"   Test  : {len(X_test)} frames")

    return X_train, X_test, y_train, y_test

def prepare_graph():
    edge_index = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

def train():
    X_train, X_test, y_train, y_test = prepare_data()
    edge_index = prepare_graph()

    model     = SkeletonGCN()
    # Learning rate plus élevé + weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Scheduler : réduit le learning rate si la loss stagne
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"\n🚀 Début de l'entraînement...\n")

    for epoch in range(100):
        model.train()
        total_loss = 0

        for i in range(len(X_train)):
            x     = torch.tensor(X_train[i], dtype=torch.float)
            label = torch.tensor([y_train[i]], dtype=torch.long)

            optimizer.zero_grad()
            output = model(x, edge_index)
            loss   = criterion(output.unsqueeze(0), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0

            with torch.no_grad():
                for i in range(len(X_test)):
                    x     = torch.tensor(X_test[i], dtype=torch.float)
                    label = y_test[i]
                    output = model(x, edge_index)
                    pred   = output.argmax().item()
                    if pred == label:
                        correct += 1

            accuracy  = correct / len(X_test) * 100
            avg_loss  = total_loss / len(X_train)
            lr_actual = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/100 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}% | LR: {lr_actual:.5f}")

    torch.save(model.state_dict(), "data/gcn_model.pth")
    print(f"\n✅ Modèle sauvegardé : data/gcn_model.pth")

if __name__ == "__main__":
    train()