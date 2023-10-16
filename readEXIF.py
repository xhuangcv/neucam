import math
import os
import json
import numpy as np
import PIL.ExifTags
from PIL import Image

if __name__ == '__main__':
    # scene_list = ['lens', 'windows', 'computer1', 'computer2']
    scene_list = ['BookShelf', 'FlowerShelf', 'Plants', 'Arch', 'Bike', 'MusicTiger', 'Sculpture', 'Tree']
    source_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MFME_nikon_9'
    out_root = '/apdcephfs/private_xanderhuang/hdr_video_data/MFME_nikon_9'

    # preprocess the source data of the scene in 'scene_list'
    for scene in scene_list:
        out_path = os.path.join(out_root, scene)
        source_path = os.path.join(source_root, scene)

        os.makedirs(out_path, exist_ok=True)
        dir_list = sorted([f for f in os.listdir(source_path) if any([f.endswith(ex) for ex in ['png', 'jpg']])])
        exp_t = np.zeros([3])
        with open(os.path.join(out_path,'exposure.txt'), "w") as f:
            for i, d in enumerate(dir_list):
                img_path = os.path.join(source_path, d)
                img = Image.open(img_path)
                        
                exif = {PIL.ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in PIL.ExifTags.TAGS
                    }
                
                exp_value =  float(exif['ExposureTime'])
                exp_value = math.log(exp_value, 2)
                exp_t[i % 3] = exp_value
                if i == 2 or i == 5 or i == 8:
                    for j in range(3):
                        f.write(str('%.1f'%(exp_t[j]-exp_t[1])) + '\n')

        print('Done with ' + out_path)
    print('Done!') 



