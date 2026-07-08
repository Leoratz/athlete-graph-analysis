import json
import numpy as np

def load_keypoints_2d(json_path):
    """
    Charge les keypoints 2D de ta camarade
    et ajoute Z=0 pour avoir du (x, y, z)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    sequence = []

    for frame_data in data:
        # keypoints shape : (1, 17, 2)
        keypoints = np.array(frame_data["keypoints"][0])  # (17, 2)
        scores    = np.array(frame_data["scores"][0])     # (17,)

        # Ajouter Z=0 pour chaque articulation
        z = np.zeros((keypoints.shape[0], 1))             # (17, 1)
        pose_3d = np.concatenate([keypoints, z], axis=1)  # (17, 3)

        sequence.append(pose_3d)

    sequence = np.array(sequence)  # (nb_frames, 17, 3)

    print(f"✅ Données chargées :")
    print(f"   Frames      : {sequence.shape[0]}")
    print(f"   Articulations: {sequence.shape[1]}")
    print(f"   Coordonnées : {sequence.shape[2]} (x, y, z=0)")

    return sequence

def normalize_sequence(sequence):
    """
    Normalise les coordonnées entre 0 et 1
    Important : les pixels (600-2300) sont très grands
    pour le modèle GCN
    """
    mean = sequence.mean()
    std  = sequence.std()
    return (sequence - mean) / std

if __name__ == "__main__":
    # Charger les données
    sequence = load_keypoints_2d("data/keypoints_2d.json")

    # Normaliser
    sequence_norm = normalize_sequence(sequence)

    # Sauvegarder en .npy pour le modèle
    np.save("data/real_data.npy", sequence_norm)
    print(f"\n✅ Données sauvegardées : data/real_data.npy")

    # Aperçu d'une frame
    print(f"\n📋 Aperçu frame 0 (normalisée) :")
    print(f"   Nez      : {sequence_norm[0][0]}")
    print(f"   Genou G  : {sequence_norm[0][9]}")
    print(f"   Genou D  : {sequence_norm[0][12]}")