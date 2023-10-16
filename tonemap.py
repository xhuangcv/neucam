import numpy as np
import imageio
import os
import json
import shutil
from matplotlib import pyplot as plt
def tonemapGamma(x):
    return ( (x) ** (1/2.2) )

def tonemapSimple(x):
    return (x / (x + 1)) ** (1 / 2.2)

def tonemapUnity(x):
    # Tonemap
    a          = 0.2
    b          = 0.29
    c          = 0.24
    d          = 0.272
    e          = 0.02
    f          = 0.3
    whiteLevel = 5.3
    whiteClip  = 1.0

    NeutralCurve = lambda w: ((w * (a * w + c * b) + d * e) / (w * (a * w + b) + d * f)) - e / f

    whiteScale = 1.0 / NeutralCurve(whiteLevel)
    x = NeutralCurve(x * whiteScale)
    x = x * whiteScale

    # Post-curve white point adjustment
    x = x / whiteClip

    return x 



if __name__ == '__main__':

    scene_list = ['YellowDog']
    # scene_list = ['BathRoom', 'Bear', 'Desk', 'DiningRoom', 'Dog', 'Fireplace', 'Sponza', 'YellowDog']

    data_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MFME/Blender/'
    out_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MFME_blender_9/'
    # exp_list = [-4,0,4] # DiningRoom
    # exp_list = [-2.,0.,5.] # sponza
    # exp_list = [-2,1,4] # Dog
    # exp_list = [-3,1,5] # BathRoom
    # exp_list = [-3,1,5] # Desk
    exp_list = [-1,1,4] # YellowDog
    # exp_list = [-3,1,4] # Bear
    # exp_list = [-3,1,5] # Fireplace

    for scene in scene_list:
        data_path = os.path.join(data_root, scene)
        out_path = os.path.join(out_root, scene)
        os.makedirs(out_path, exist_ok=True)

        train_exps = {}
        np.random.seed(1)
        print(data_path)
        # training dataset
        dir_list = [f for f in os.listdir(data_path) if f.endswith('hdr')]
        dir_list.sort()
        print(dir_list)
        with open(os.path.join(out_path,'exposure.txt'), 'w') as f:
            for i, d in enumerate(dir_list[:3]):
                img_path = os.path.join(data_path, d)
                for j, exp_value in enumerate(exp_list):
                    save_path = os.path.join(out_path, 'F%dE%d.png'% (i, j))
                    hdrimg = imageio.imread(img_path, 'hdr')
                    hdrimg[:,:,0:3] = np.clip(tonemapSimple(hdrimg[:,:,0:3] * (2 ** exp_value)), 0, 1)
                    ldrimg = (hdrimg * 255).astype(np.uint8)
                    imageio.imwrite(save_path, ldrimg)
                    train_exps['F%dE%d.png'% (i, j)] = exp_value
                    print(save_path)
                    f.write(str(exp_value) + '\n')


        with open(os.path.join(out_path,'exposure.json'), 'w') as f:
            f.write(json.dumps(train_exps, indent=4))

        print('Done with ' + scene)