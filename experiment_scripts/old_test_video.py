'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
import torch
import imageio
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from torchmeta.modules import module

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import configargparse
from functools import partial
import skvideo.datasets

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='/apdcephfs/private_xanderhuang/hdr_video_results/', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, default='dog',
               help='Video dataset; one of (cat, bikes)', choices=['cat', 'bikes', 'bear', 'dog', 'lamp', 'channel'])
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=40e-4,
               help='What fraction of video pixels to sample in each batch (default is all)')
p.add_argument('--checkpoint_path', type=str, default='/apdcephfs/private_xanderhuang/results/logs_hdr/20220222/worker_v3_195112/checkpoints/model_current.pth', help='Checkpoint to trained model.')
p.add_argument('--pre_hdr',type=bool, default=True, help='Whether to reconstruct HDR video.')
p.add_argument('--uv_mapping',type=bool, default=True, help='Whether to map (u,v,t) to (u,v).')
p.add_argument('--learn_exp', type=bool, default=True, help='Whether to learn the exposure of each frame.')
p.add_argument('--split_layer', type=bool, default=False, help='Whether split to static layer and dynamic layer.')
p.add_argument("--chunk", type=int, default=65536, help='number of rays processed in parallel, decrease if running out of memory')
opt = p.parse_args()

if opt.dataset == 'cat':
    video_path = './data/video_512.npy'
elif opt.dataset == 'bikes':
    video_path = skvideo.datasets.bikes()
elif opt.dataset == 'dog':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/Dog-3Exp-2Stop/'
elif opt.dataset == 'bear':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/bear.mp4'
elif opt.dataset == 'lamp':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/10-11-19.14.50_seq_grab_loop_0912_3stop_2exps_3s_12.0g_wb/'
elif opt.dataset == 'channel':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/10-11-16.12.47_seq_grab_loop_0912_2stop_3exps_3s_12.0g_wb/'
elif opt.dataset == 'zhangqi':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/zhangqi/'

vid_dataset = dataio.Video(video_path, load_mask=False)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape)
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
ckp = torch.load(opt.checkpoint_path)
image_resolution = (360, 640)
# Define the model.
if opt.uv_mapping:
    in_features=2
else:
    in_features=3
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=in_features, out_features=3,
                                 mode='mlp', hidden_features=512, num_hidden_layers=3, sidelength=image_resolution)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=in_features, out_features=3, hidden_features=512, 
                                 mode=opt.model_type, sidelength=image_resolution)
else:
    raise NotImplementedError
model.load_state_dict(ckp['model_state_dict'])
model.cuda()

# Define exposure learning model.
if 'model_exp_state_dict' in ckp:
    model_exp = modules.LearnExps(vid_dataset.shape[0])
    model_exp.load_state_dict(ckp['model_exp_state_dict'])
    model_exp.cuda()
else:
    model_exp = None

# Define tone-mapping model.
if 'model_tm_state_dict' in ckp:
    model_tm = modules.TonemapNet()
    model_tm.load_state_dict(ckp['model_tm_state_dict'])
    model_tm.cuda()
else:
    model_tm = None

# Define static layer uvt mapping model.
if 'model_mapping_s_state_dict' in ckp:
    model_mapping_s = modules.MappingNet(hidden_list=[256,256,256,256])
    model_mapping_s.load_state_dict(ckp['model_mapping_s_state_dict'])
    model_mapping_s.cuda()
else:
    model_mapping_s = None

# Define dynamic layer uvt mapping model.
if 'model_mapping_d_state_dict' in ckp:
    model_mapping_d = modules.MappingNet(hidden_list=[256, 256, 256, 256], use_encoding=False)
    model_mapping_d.load_state_dict(ckp['model_mapping_d_state_dict'])
    model_mapping_d.cuda()
else:
    model_mapping_d = None

# Define alpha blending model.
if 'model_alpha_state_dict' in ckp:
    model_alpha = modules.MappingNet(hidden_list=[256, 256, 256, 256], out_dim=1, use_encoding=True, use_sigmoid=True)
    model_alpha.load_state_dict(ckp['model_alpha_state_dict'])
    model_alpha.cuda()
else:
    model_alpha = None

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Get ground truth and input data
model_input, gt = next(iter(dataloader))
model_input = {key: value.cuda() for key, value in model_input.items()}
gt = {key: value.cuda() for key, value in gt.items()}

# Evaluate the trained model
pre_hdr_video = []
pre_ldr_video = []
alpha_video = []

with torch.no_grad():
    # Learned exposures
    if model_exp != None:
        learned_exp = model_exp().view(-1, 1, 1).repeat(1, vid_dataset.shape[1],  vid_dataset.shape[2])

    for i in range(0, gt['img'].shape[1], opt.chunk):
        input_chunk = {
            'coords':model_input['coords'][:,i:i+opt.chunk,:]
        }

        if model_exp != None:
            input_chunk.update({'exps':learned_exp.view(1,-1,1)[:,i:i+opt.chunk,:]})
        else:
            input_chunk.update({'exps':model_input['exps'][:,i:i+opt.chunk,:]})
        
        # Step1: map (u,v,t) to (u,v).
        model_input_s = {}
        model_input_d = {}
        if model_mapping_s != None:
            input_uv_s = model_mapping_s(input_chunk['coords']) \
                                        .view(input_chunk['coords'].shape[0], -1, 2)
            model_input_s['coords'] =  input_chunk['coords'][..., 1:] + input_uv_s
            if model_mapping_d is not None:
                model_input_s['coords'] = model_input_s['coords']*0.25 - 0.5
                input_uv_d = model_mapping_d(input_chunk['coords']) \
                                        .view(input_chunk['coords'].shape[0], -1, 2)
                model_input_d['coords'] =  input_chunk['coords'][..., 1:] + input_uv_d
                model_input_d['coords'] = model_input_d['coords']*0.25 + 0.5
  
        # Step2: map (u,v) to HDR rgb.
        model_output_s = model(model_input_s)
        if model_mapping_d != None:
            model_output_d = model(model_input_d)

        # Blend the static scene and dynamic scene.
        model_output = model_output_s
        if model_alpha != None:
            alpha = model_alpha(input_chunk['coords']).view(1, -1, 1)
            model_output['model_out'] = (1-alpha)*model_output_s['model_out'] + alpha*model_output_d['model_out']
            alpha_video.append(alpha)

        # # Step3: map HDR rgb to LDR rgb
        # if model_tm != None:
        #     hdr_pixels = torch.exp(model_output['model_out'])
        #     ldr_pixels = torch.clamp(model_tm(model_output['model_out'].view(-1, 3), 
        #                                     input_chunk['exps'].view(-1, 1)).view(1, -1, 3), -1, 1)
        #     ldr_pixels = ldr_pixels / 2 + 0.5
        #     pre_hdr_video.append(hdr_pixels)
        #     pre_ldr_video.append(ldr_pixels)
        # else:
        #     pre_ldr_video.append(torch.clamp(model_output['model_out']/2+0.5, 0, 1))

        # Step3: map HDR rgb to LDR rgb
        if model_tm != None:
            
            ldr_pixels_s = torch.clamp(model_tm(model_output_s['model_out'].view(-1, 3), 
                                            input_chunk['exps'].view(-1, 1)).view(1, -1, 3), -1, 1)
            if model_mapping_d is not None:
                ldr_pixels_d = torch.clamp(model_tm(model_output_d['model_out'].view(-1, 3), 
                                            input_chunk['exps'].view(-1, 1)).view(1, -1, 3), -1, 1)
                ldr_pixels = (1-alpha)*model_output_s['model_out'] + alpha*model_output_d['model_out']
            ldr_pixels = ldr_pixels / 2 + 0.5
            hdr_pixels = torch.exp(model_output['model_out'])
            pre_hdr_video.append(hdr_pixels)
            pre_ldr_video.append(ldr_pixels)
        else:
            pre_ldr_video.append(torch.clamp(model_output['model_out']/2+0.5, 0, 1))

# Draw the CRF curve.
if model_tm != None:
    ln_x = torch.linspace(-5, 3, 1000).reshape([-1, 1]).repeat(1, 3).cuda()
    ln_t = torch.zeros([ln_x.shape[0], 1]).cuda()
    with torch.no_grad():
        y = torch.clamp(model_tm(ln_x, ln_t).view(-1, 3), -1, 1)
        y = y / 2 + 0.5

    x = ln_x.cpu().numpy()
    y = y.detach().cpu().numpy()
    plt.figure()
    plt.xlabel("lnx")
    plt.ylabel("pixel value")
    plt.plot(x[:,:1],y[:,0:1], color='r', label='CRF red')
    plt.plot(x[:,:1],y[:,1:2], color='g', label='CRF green')
    plt.plot(x[:,:1],y[:,2:3], color='b', label='CRF blue')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root_path, 'CRF.png'))
    plt.close()

# Save reconstructed videos.
pre_ldr_video = torch.cat(pre_ldr_video, 1).reshape(list(vid_dataset.shape) + [3])
prevideo = torch.squeeze(pre_ldr_video).detach().cpu().numpy()
imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_ldr_video.gif'), utils.to8b(prevideo), fps=30)

if pre_hdr_video != []:
    pre_hdr_video = torch.cat(pre_hdr_video, 1).reshape(list(vid_dataset.shape) + [3])
    prevideo = torch.squeeze(pre_hdr_video).detach().cpu().numpy()
    imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_hdr_video.gif'), utils.tonemap(prevideo / np.max(prevideo)), fps=20)

    # for i in range(prevideo.shape[0]):
    #     imageio.imwrite(os.path.join(root_path, 'frame_%03d.hdr'%i), prevideo[i])

if alpha_video != []:
    alpha_video = torch.cat(alpha_video, 1).reshape(list(vid_dataset.shape) + [1])
    prevideo = torch.squeeze(alpha_video).detach().cpu().numpy()
    imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_alpha.gif'), utils.to8b(prevideo), fps=20)

    for i in range(prevideo.shape[0]):
        imageio.imwrite(os.path.join(root_path, 'alpha_%03d.png'%i), utils.to8b(prevideo[i]))

print('Done with testing.')

