import os

#hidden channels 32, 64, 128, 256
#interactions 2,4,6

#alignn 2,4,6 layers
#cgcn same nr of primary layers, 0 alignn

for a in [0]:
    for e in [32, 64]:
        with open('zeonet_train.sh', 'w') as file:
            file.write(f'#!/bin/bash\nsource /home/TUE/20183707/miniconda3/etc/profile.d/conda.sh\nsource activate env2\npython alignn_test.py --a {a} --e {e}\nsource deactivate')
        os.system(f'sbatch -p all --gres=gpu:1 zeonet_train.sh')
