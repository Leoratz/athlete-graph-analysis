import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skeleton_graph import EDGES, JOINTS, RISK_JOINTS
import sys
import os
sys.path.append(os.path.dirname(__file__))

EDGES_LIST = EDGES
RISK = RISK_JOINTS

def animate_skeleton(sequence, title="Mouvement"):
    """Anime une séquence de squelette en 3D"""
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        pose = sequence[frame]  # shape (17, 3)

        # Dessiner les arêtes
        for (i, j) in EDGES_LIST:
            ax.plot(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 2], pose[j, 2]],
                [pose[i, 1], pose[j, 1]],
                color="#555", linewidth=2
            )

        # Dessiner les nœuds
        for idx in range(len(pose)):
            color = "#ef4444" if idx in RISK else "#60a5fa"
            ax.scatter(
                pose[idx, 0],
                pose[idx, 2],
                pose[idx, 1],
                color=color, s=80, zorder=5
            )

        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 12)
        ax.set_title(f"{title} — Frame {frame+1}/{len(sequence)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (hauteur)")

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(sequence),
        interval=50,  # 50ms entre chaque frame = 20 FPS
        repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Charger les séquences
    normal = np.load("data/normal_running.npy")
    risky  = np.load("data/risky_running.npy")

    print("▶️  Animation course normale...")
    animate_skeleton(normal, title="Course normale ✅")

    print("▶️  Animation course à risque...")
    animate_skeleton(risky, title="Course à risque ⚠️")