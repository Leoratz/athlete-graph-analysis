import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Définition des articulations (nœuds du graphe)

JOINTS = {
    0: "Nez",
    1: "Cou",
    2: "Epaule_G",
    3: "Coude_G",
    4: "Poignet_G",
    5: "Epaule_D",
    6: "Coude_D",
    7: "Poignet_D",
    8: "Hanche_G",
    9: "Genou_G",
    10: "Cheville_G",
    11: "Hanche_D",
    12: "Genou_D",
    13: "Cheville_D",
    14: "Oeil_G",
    15: "Oeil_D",
    16: "Oreille_G",
}


# Connexions anatomiques (arêtes du graphe)

EDGES = [
    (0, 1),   # Nez → Cou
    (1, 2),   # Cou → Epaule G
    (2, 3),   # Epaule G → Coude G
    (3, 4),   # Coude G → Poignet G
    (1, 5),   # Cou → Epaule D
    (5, 6),   # Epaule D → Coude D
    (6, 7),   # Coude D → Poignet D
    (1, 8),   # Cou → Hanche G
    (8, 9),   # Hanche G → Genou G
    (9, 10),  # Genou G → Cheville G
    (1, 11),  # Cou → Hanche D
    (11, 12), # Hanche D → Genou D
    (12, 13), # Genou D → Cheville D
    (0, 14),  # Nez → Oeil G
    (0, 15),  # Nez → Oeil D
    (14, 16), # Oeil G → Oreille G
]


# Articulations à risque (pour la course)

RISK_JOINTS = [9, 12, 8, 11]  # Genoux et Hanches

def build_skeleton_graph():
    """Construit et retourne le graphe de squelette"""
    G = nx.Graph()
    
    for idx, name in JOINTS.items():
        is_risk = idx in RISK_JOINTS
        G.add_node(idx, name=name, risk=is_risk)
    
    G.add_edges_from(EDGES)
    
    return G

def visualize_graph(G):
    """Visualise le graphe avec couleurs par zone"""
    plt.figure(figsize=(10, 12))
    
    color_map = []
    for node in G.nodes():
        if node in [0, 14, 15, 16]:
            color_map.append("#a78bfa")   # Tête - violet
        elif node in [1, 2, 5]:
            color_map.append("#60a5fa")   # Buste - bleu
        elif node in [3, 4, 6, 7]:
            color_map.append("#34d399")   # Bras - vert
        elif node in [8, 11]:
            color_map.append("#f97316")   # Hanches - orange
        elif node in [9, 12]:
            color_map.append("#ef4444")   # Genoux - rouge (risque)
        else:
            color_map.append("#facc15")   # Chevilles - jaune

    pos = {
        0:  (0,    10),
        14: (-0.3, 10.3),
        15: (0.3,  10.3),
        16: (-0.6, 10.1),
        1:  (0,    9),
        2:  (-1.5, 8),
        3:  (-2,   6.5),
        4:  (-2.2, 5),
        5:  (1.5,  8),
        6:  (2,    6.5),
        7:  (2.2,  5),
        8:  (-0.8, 6.5),
        9:  (-0.8, 4),
        10: (-0.8, 2),
        11: (0.8,  6.5),
        12: (0.8,  4),
        13: (0.8,  2),
    }
    
    labels = {idx: name for idx, name in JOINTS.items()}
    
    nx.draw(G, pos=pos, labels=labels, node_color=color_map,
            node_size=800, font_size=7, font_color="white",
            edge_color="#555", width=2)
    
    plt.title("Graphe de squelette humain — BioGraph Sports\n🔴 Genoux (risque)  🟠 Hanches  🔵 Buste  🟢 Bras  🟣 Tête  🟡 Chevilles",
              fontsize=11)
    plt.tight_layout()
    plt.savefig("skeleton_graph.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Graphe sauvegardé : skeleton_graph.png")

if __name__ == "__main__":
    G = build_skeleton_graph()
    print(f"✅ Graphe créé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    visualize_graph(G)