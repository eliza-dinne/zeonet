#!/bin/bash
source /home/TUE/20183707/miniconda3/etc/profile.d/conda.sh
source activate env1
python schnet_test.py --h 64 --i 4
source deactivate