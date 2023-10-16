import cv2
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

    # scene_list = ['BookShelf', 'FlowerShelf', 'Plants', 'Arch', 'Bike', 'MusicTiger', 'Sculpture', 'Tree', 'BathRoom', 'Sponza', 'Dog', 'YellowDog']
    scene_list = ['MF001', 'MF002', 'MF003', 'MF004', 'MF005', 'MF006', 'MF007', 'MF008']

    # data_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MFME_blender_9/'
    data_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MF_nikon/'
    out_root = '/apdcephfs/private_xanderhuang/hdr_video_data/Baseline_data'

    near_path = os.path.join(out_root, 'near')
    mid_path = os.path.join(out_root, 'mid')
    far_path = os.path.join(out_root, 'far')
    os.makedirs(near_path, exist_ok=True)
    os.makedirs(mid_path, exist_ok=True)
    os.makedirs(far_path, exist_ok=True)

    near=0; mid=0; far=0
    for scene in scene_list:
        data_path = os.path.join(data_root, scene)
        print(data_path)
        # training dataset
        dir_list = [f for f in os.listdir(data_path) if f.endswith('png') or f.endswith('jpg') or f.endswith('JPG')]
        dir_list.sort()
        print(dir_list)

        for i, d in enumerate(dir_list):
            img_path = os.path.join(data_path, d)
            img = imageio.imread(img_path)
            if img.shape[0] != 600:
                img = cv2.resize(img, (900, 600))
            if d[0:2] == 'F0':
                near += 1
                save_path = os.path.join(near_path, '%03d.png'% near)
                imageio.imwrite(save_path, img)
            if d[0:2] == 'F1':
                mid += 1
                save_path = os.path.join(mid_path, '%03d.png'% mid)
                imageio.imwrite(save_path, img)
            if d[0:2] == 'F2':
                far += 1
                save_path = os.path.join(far_path, '%03d.png'% far)
                imageio.imwrite(save_path, img)


        print('Done with ' + scene)