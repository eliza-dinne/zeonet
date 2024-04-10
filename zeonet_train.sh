#!/bin/bash
source /home/TUE/20183707/miniconda3/etc/profile.d/conda.sh
source activate env2
python alignn_test.py --a 0 --e 64
source deactivate