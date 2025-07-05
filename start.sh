#!/bin/bash

# Activeer Conda (nodig om 'conda activate' te laten werken in scripts)
source ~/miniconda3/etc/profile.d/conda.sh

# Activeer de juiste omgeving
conda activate latentsync

# Ga naar de juiste projectmap
cd ~/projects/LatentSync

# Start de Gradio app
python gradio_app.py

