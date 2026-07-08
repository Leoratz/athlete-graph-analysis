import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SkeletonGCN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=2):
        """
        in_channels  : 3 (x, y, z)
        hidden_channels : neurones cachés
        out_channels : 2 classes (normal / à risque)
        """
        super(SkeletonGCN, self).__init__()

        # Couches de convolution sur le graphe
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Couche de classification finale
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # Dropout pour éviter l'overfitting
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        """
        x          : positions des articulations (17, 3)
        edge_index : connexions du graphe (2, nb_aretes)
        """
        # Couche 1 : convolution + activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Couche 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Couche 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Moyenne sur tous les noeuds → représentation globale
        x = x.mean(dim=0)

        # Classification finale
        x = self.classifier(x)

        return x


# Test rapide du modèle
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from skeleton_graph import EDGES

    # Convertir les arêtes en format PyTorch Geometric
    edge_index = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
    # Ajouter les arêtes dans les 2 sens (graphe non orienté)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Simuler une frame : 17 articulations × 3 coordonnées
    x = torch.randn(17, 3)

    # Créer le modèle
    model = SkeletonGCN()
    print(f"✅ Modèle créé :")
    print(model)

    # Passer une frame dans le modèle
    output = model(x, edge_index)
    print(f"\n✅ Test forward pass réussi !")
    print(f"   Input  : {x.shape}  (17 articulations × 3 coordonnées)")
    print(f"   Output : {output.shape} (2 classes : normal / à risque)")
    print(f"   Scores : {output.detach().numpy()}")