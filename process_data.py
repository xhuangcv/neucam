from pickletools import uint8
from cv2 import imwrite
import imageio
import numpy as np
import os
import cv2
import torch

root_path = '/apdcephfs/private_xanderhuang/hdr_video_data/Denoise'
out_path = '/apdcephfs/private_xanderhuang/hdr_video_data/Denoise'
file_name = [f for f in sorted(os.listdir(root_path)) if f.endswith('PNG')]
H = 400; W = 500
s = 1000
all_img = np.zeros([256,256,3])
for i, f in enumerate(file_name[0:2]):
    file_path = os.path.join(root_path, f)
    img = cv2.imread(file_path)
    img = img[s:s+256, s:s+256, :]
    all_img += img
    # img = cv2.resize(img, [W, H])
    # img = img.transpose(1, 0, 2)  
    # img = np.flip(img, 0)

    cv2.imwrite(os.path.join(out_path,'%04d.png'%i), img)
    print(i)

    # gamma_list = np.linspace(1, 500, 10)
    # for j in gamma_list:
    #     lw = (img/255.)
    #     lw = np.log(lw*j + 1) / np.log(j+1)
    #     cv2.imwrite('%03d.png'%j, (lw*255).astype(np.uint8))
    #     print(j)
    print(all_img)
cv2.imwrite(os.path.join(out_path,'mean.png'), np.uint8(all_img/(i+1)))