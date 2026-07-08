import numpy as np

# Position de base du squelette "debout"
BASE_POSE = np.array([
    [0,    10,  0],   # 0  Nez
    [0,    9,   0],   # 1  Cou
    [-1.5, 8,   0],   # 2  Epaule_G
    [-2,   6.5, 0],   # 3  Coude_G
    [-2.2, 5,   0],   # 4  Poignet_G
    [1.5,  8,   0],   # 5  Epaule_D
    [2,    6.5, 0],   # 6  Coude_D
    [2.2,  5,   0],   # 7  Poignet_D
    [-0.8, 6.5, 0],   # 8  Hanche_G
    [-0.8, 4,   0],   # 9  Genou_G
    [-0.8, 2,   0],   # 10 Cheville_G
    [0.8,  6.5, 0],   # 11 Hanche_D
    [0.8,  4,   0],   # 12 Genou_D
    [0.8,  2,   0],   # 13 Cheville_D
    [-0.3, 10.3,0],   # 14 Oeil_G
    [0.3,  10.3,0],   # 15 Oeil_D
    [-0.6, 10.1,0],   # 16 Oreille_G
], dtype=float)

def simulate_normal_running(n_frames=120):
    """
    Simule une séquence de course NORMALE
    → mouvements fluides et symétriques
    """
    sequence = []
    
    for t in range(n_frames):
        pose = BASE_POSE.copy()
        
        # Oscillation naturelle du corps
        cycle = 2 * np.pi * t / 30

        # Jambe gauche
        pose[9,  1] += np.sin(cycle) * 0.5       # Genou G monte/descend
        pose[10, 1] += np.sin(cycle + 0.5) * 0.4 # Cheville G

        # Jambe droite (déphasée)
        pose[12, 1] += np.sin(cycle + np.pi) * 0.5       # Genou D
        pose[13, 1] += np.sin(cycle + np.pi + 0.5) * 0.4 # Cheville D

        # Bras (mouvement opposé aux jambes)
        pose[3, 1] += np.sin(cycle + np.pi) * 0.3  # Coude G
        pose[6, 1] += np.sin(cycle) * 0.3           # Coude D

        # Légère oscillation latérale du buste
        pose[1, 0] += np.sin(cycle) * 0.05  # Cou

        sequence.append(pose)
    
    return np.array(sequence)  # shape: (120, 17, 3)

def simulate_risky_running(n_frames=120):
    """
    Simule une séquence de course À RISQUE
    → genou gauche s'effondre vers l'intérieur (risque LCA)
    """
    sequence = []
    
    for t in range(n_frames):
        pose = BASE_POSE.copy()
        
        cycle = 2 * np.pi * t / 30

        # Jambe gauche — mouvement anormal
        pose[9,  1] += np.sin(cycle) * 0.5
        pose[9,  0] += np.sin(cycle) * 0.4   # ⚠️ Genou G dévie latéralement
        pose[10, 1] += np.sin(cycle + 0.5) * 0.4

        # Jambe droite — normale
        pose[12, 1] += np.sin(cycle + np.pi) * 0.5
        pose[13, 1] += np.sin(cycle + np.pi + 0.5) * 0.4

        # Bras
        pose[3, 1] += np.sin(cycle + np.pi) * 0.3
        pose[6, 1] += np.sin(cycle) * 0.3

        sequence.append(pose)
    
    return np.array(sequence)  # shape: (120, 17, 3)

def save_sequences():
    """Sauvegarde les séquences simulées"""
    normal = simulate_normal_running()
    risky  = simulate_risky_running()
    
    np.save("data/normal_running.npy", normal)
    np.save("data/risky_running.npy",  risky)
    
    print(f"✅ Séquence normale sauvegardée  : shape {normal.shape}")
    print(f"✅ Séquence à risque sauvegardée : shape {risky.shape}")
    print(f"   → {normal.shape[0]} frames | {normal.shape[1]} articulations | {normal.shape[2]} coordonnées (x,y,z)")

if __name__ == "__main__":
    save_sequences()