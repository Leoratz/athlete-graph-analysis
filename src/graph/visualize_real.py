import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, os
sys.path.append(os.path.dirname(__file__))
from skeleton_graph import EDGES, RISK_JOINTS

def animate_real_skeleton(sequence, title="Vraies données"):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        pose = sequence[frame]  # (17, 3)

        # Dessiner les arêtes
        for (i, j) in EDGES:
            ax.plot(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 2], pose[j, 2]],
                [-pose[i, 1], -pose[j, 1]],  # Inverser Y (pixels → hauteur)
                color="#555", linewidth=2
            )

        # Dessiner les nœuds
        for idx in range(len(pose)):
            color = "#ef4444" if idx in RISK_JOINTS else "#60a5fa"
            ax.scatter(
                pose[idx, 0],
                pose[idx, 2],
                -pose[idx, 1],
                color=color, s=80, zorder=5
            )

        ax.set_title(f"{title} — Frame {frame+1}/{len(sequence)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Hauteur")

    ani = animation.FuncAnimation(
        fig, update,
        frames=min(100, len(sequence)),  # 100 premières frames
        interval=50,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sequence = np.load("data/real_data.npy")
    print(f"✅ {sequence.shape[0]} frames chargées")
    animate_real_skeleton(sequence, title="Course réelle 🏃")