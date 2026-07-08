## Prérequis

- Python 3.12 (ne pas utiliser Python 3.13, incompatible avec PyTorch)
- GPU Nvidia recommandé (CUDA 12.1+)

## Installation

### 1. Créer l'environnement virtuel
py -3.12 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

### 2. Installer PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Pour CPU uniquement :
pip install torch torchvision torchaudio

### 3. Installer les dépendances
pip install -r requirements.txt

## Vérifier l'installation
python -c "import torch; print(torch.cuda.is_available())"