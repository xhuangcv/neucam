# Inverting the Imaging Process by Learning an Implicit Camera Model
### [Project Page](https://xhuangcv.github.io/neucam/) | [Paper](https://arxiv.org/abs/2304.12748) | [Dataset](https://drive.google.com/drive/folders/1dNnsCBasIVW3BxvUHlvwzWlaVmKaczJC) | [Results](https://drive.google.com/drive/folders/1dNnsCBasIVW3BxvUHlvwzWlaVmKaczJC)
<br>

[Xin Huang](https://xhuangcv.github.io/)<sup>1</sup>,
[Qi Zhang](https://qzhang-cv.github.io/)<sup>2</sup>,
[Ying Feng](https://scholar.google.com.tw/citations?user=PhkrqioAAAAJ&hl=zh-TW)<sup>2</sup>,
[Hongdong Li](http://users.cecs.anu.edu.au/~hongdong/)<sup>3</sup>,
[Qing Wang](https://teacher.nwpu.edu.cn/qwang.html)<sup>1</sup><br>
<sup>1</sup>Northwestern Polytechnical University, <sup>2</sup>Tencent AI Lab, <sup>3</sup>Australian National University

This is the official implementation of the paper "Inverting the Imaging Process by Learning an Implicit Camera Model". Our method, Neucam, is able to recover all-in-focus HDR images from images captured with multi-setting images (multi-exposure and multi-focus). Moreover, Neucam has the potential to benefit a wide array of other inverse imaging tasks such as video deblurring and video HDR Imaging.

<p align="left">
    <img src='https://xhuangcv.github.io/neucam/images/overview.png' width="800">
</p>



## Get started
If you want to reproduce all the results shown in the paper, you can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate neucam
```

## Download NeuCam Dataset
We have prepared several sample datasets for NeuCam training. You can download these test samples via the provided [link](https://drive.google.com/drive/folders/1dNnsCBasIVW3BxvUHlvwzWlaVmKaczJC). Once downloaded, please extract the files and place them in the `./dataset` directory of this GitHub repository. For instance, your file path should look something like this: `./dataset/MFME`.

* **MFME** directory contains 4 real-world scenes and 4 synthetic scenes. Each scene contains 9 images of 3 different
focuses and 3 different exposures. The real-world images are captured by a digital camera with a tripod, and the synthetic images are rendered in Blender.
* **MF-static** directory contains 8 real-world scenes. Two images focusing on the foreground and background respectively are captured for each scene.
* **ME-dynamic** directory contains 5 real-world dynamic scenes from the [HDR imaging dataset](https://web.ece.ucsb.edu/~psen/hdrvideo). Three images with different exposures are captured for each scene.
* **Video-deblur** directory contains two videos from the [Deep Video Deblurring dataset](https://github.com/shuochsu/DeepVideoDeblurring), characterized by camera motion blur.
* **Video-hdr** directory contains two videos from the [Deep HDR Video dataset](https://guanyingc.github.io/DeepHDRVideo-Dataset/), notable for their varying exposure levels.

## Reproducing experiments
The directory `experiment_scripts` contains scripts used for the experiments in our paper. The directory `configs` contains parameter settings for all these experiments. To keep track of your progress, the training code generates TensorBoard summaries and stores them in a "summaries" subdirectory located within the `logging_root`.

### Training
* The all-in-focus HDR imaging experiment can be reproduced with
```
python experiment_scripts/train_video.py -c configs/config_mfme_blender.txt # blender dataset
python experiment_scripts/train_video.py -c configs/config_mfme_nikon.txt # nikon dataset
```
* The all-in-focus imaging experiment can be reproduced with
```
python experiment_scripts/train_video.py -c configs/config_mf_static.txt
```
* The HDR imaging experiment can be reproduced with
```
python experiment_scripts/train_video.py -c configs/config_me_dynamic.txt
```
* The video deblurring experiment can be reproduced with
```
python experiment_scripts/train_video.py -c configs/config_video_deblur.txt
```
* The video HDR imaging experiment can be reproduced with
```
python experiment_scripts/train_video.py -c configs/config_video_hdr.txt
```

Upon completion of the training, both the model and the results will be stored in the `./results` directory. Please note that the final all-in-focus and HDR results are produced post-inference.

### Inference
To render the all-in-focus and HDR results, you can run
```
python experiment_scipts/render_video.py -c <path_to_saved_results>/config.txt --checkpoint_path=<path_to_saved_results>
```
The term `<path_to_saved_results>` refers to the specific location where the trained model and results are stored during the training process. An example of this could be `./results/20231018/Dog_145236`.

## Results
We also offer results generated by our method. These can be downloaded using the provided [link](https://drive.google.com/drive/folders/1dNnsCBasIVW3BxvUHlvwzWlaVmKaczJC). Please note that all HDR results in the experiment have been tonemapped using [Photomatix](https://www.hdrsoft.com/). For visualization of HDR results, we recommend installing either [Phototmatix](https://www.hdrsoft.com/) or [Luminance HDR](http://qtpfsgui.sourceforge.net/).

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{huang2023inverting,
  title={Inverting the Imaging Process by Learning an Implicit Camera Model},
  author={Huang, Xin and Zhang, Qi and Feng, Ying and Li, Hongdong and Wang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21456--21465},
  year={2023}
}
```

## Acknowledgments

Our project is benefit from these great resources:

- [Implicit Neural Representations with Periodic Activation Functions.](https://github.com/vsitzmann/siren/tree/master)
- [Layered Neural Atlases for Consistent Video Editing.](https://github.com/ykasten/layered-neural-atlases)

We are grateful for their contribution in sharing their code.
