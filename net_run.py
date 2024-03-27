import os

#hidden channels 32, 64, 128, 256
#interactions 2,4,6

#alignn 2,4,6 layers
#cgcn same nr of primary layers, 0 alignn

for ch in [32]: #32, 64, 128, 256
    for interactions in [2, 4, 6]: #2, 4, 6
        with open('zeonet_train.sh', 'r') as file:
            lines = file.readlines()

            for j in range(len(lines)):
                if lines[j].startswith('python schnet_test.py'):
                    lines[j] = f'python schnet_test.py --h {ch} --i {interactions}\n'
        os.system(f'sbatch -p all --gres=gpu:1 zeonet_train.sh')
