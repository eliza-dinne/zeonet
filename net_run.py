import os

#hidden channels 32, 64, 128, 256
#interactions 2,4,6

#alignn 2,4,6 layers
#cgcn same nr of primary layers, 0 alignn

for ch in [32]: #32, 64, 128, 256
    for interactions in [2]: #2, 4, 6
        os.system(f'python schnet_test.py --h {ch} --i {interactions}')


