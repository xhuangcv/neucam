# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
import imageio
import numpy as np
import dataio, utils, modules
from configs import opt
from torch.utils.data import DataLoader
import torch.nn.functional as tfunc
from matplotlib import pyplot as plt
plt.switch_backend('agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = utils.load_data_path(opt.dataset)
vid_dataset = dataio.Video(video_path, opt.seq_size, load_gt=opt.save_gt, load_depth=opt.use_depth, not_load_exps=opt.learn_exp)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape)
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=1)
ckp = torch.load(os.path.join(opt.checkpoint_path, 'checkpoints', opt.checkpoint_name))
image_resolution = (opt.seq_size[1], opt.seq_size[2])

# Define the model.
model_atlas_b = modules.SingleBVPNet(type=opt.atlas_b_type,
                                     mode=opt.atlas_b_mode,
                                     hidden_features=opt.atlas_b_hidden_dim,
                                     num_hidden_layers=opt.atlas_b_num_layers,
                                     in_features=opt.atlas_b_in_dim,
                                     out_features=opt.atlas_b_out_dim,
                                     sidelength=image_resolution,
                                     num_frequencies=opt.atlas_b_num_frequencies).to(device)
model_atlas_b.load_state_dict(ckp['model_state_dict'])

model_uv_b = None
if 'model_mapping_s_state_dict' in ckp:
    model_uv_b = modules.SimpleMLP(hidden_dim=opt.uv_b_hidden_dim, 
                                   num_layer=opt.uv_b_num_layers, 
                                   in_dim=opt.uv_b_in_dim,
                                   out_dim=opt.uv_b_out_dim,
                                   use_encoding=opt.uv_b_use_encoding, 
                                   num_frequencies=opt.uv_b_num_frequencies,
                                   skip_layer=opt.uv_b_skip,
                                   nonlinear=opt.uv_b_last_func).to(device)
    model_uv_b.load_state_dict(ckp['model_mapping_s_state_dict'])

model_atlas_f = None
if 'model_f_state_dict' in ckp:
    out_features = 3 if opt.alpha_mlp else 4
    model_atlas_f = modules.SingleBVPNet(type=opt.atlas_f_type,
                                         mode=opt.atlas_f_mode,
                                         hidden_features=opt.atlas_f_hidden_dim,
                                         in_features=opt.atlas_f_in_dim,
                                         out_features=out_features,
                                         sidelength=image_resolution,
                                         num_frequencies=opt.atlas_f_num_frequencies).to(device)
    model_atlas_f.load_state_dict(ckp['model_f_state_dict'])

model_uv_f = None
if 'model_mapping_d_state_dict' in ckp:
    model_uv_f = modules.SimpleMLP(hidden_dim=opt.uv_f_hidden_dim, 
                                   num_layer=opt.uv_f_num_layers, 
                                   in_dim=opt.uv_f_in_dim,
                                   out_dim=opt.uv_f_out_dim,
                                   use_encoding=opt.uv_f_use_encoding, 
                                   num_frequencies=opt.uv_f_num_frequencies,
                                   skip_layer=opt.uv_f_skip,
                                   nonlinear=opt.uv_f_last_func).to(device)
    model_uv_f.load_state_dict(ckp['model_mapping_d_state_dict'])

model_exp = None
if 'model_exp_state_dict' in ckp:
    model_exp = modules.LearnExps(vid_dataset.num_frame).to(device) 
    model_exp.load_state_dict(ckp['model_exp_state_dict'])  

model_focal = None
if 'model_focal_state_dict' in ckp:
    model_focal = modules.LearnFocal(opt.focal_list).to(device) 
    # model_focal = modules.LearnExps(vid_dataset.num_frame).to(device) 
    model_focal.load_state_dict(ckp['model_focal_state_dict'])   

model_depth = None
if 'model_depth_state_dict' in ckp:
    model_depth = modules.LearnDepth(vid_dataset.depth).to(device)
    model_depth.load_state_dict(ckp['model_depth_state_dict'])  

model_tm = None
if 'model_tm_state_dict' in ckp:
    model_tm = modules.TonemapNet().to(device)
    model_tm.load_state_dict(ckp['model_tm_state_dict'])

model_blur = None
if 'model_blur_state_dict' in ckp:
    model_blur = modules.SimpleMLP(hidden_dim=opt.blur_hidden_dim,
                                   num_layer=opt.blur_num_layers,
                                   in_dim=opt.blur_in_dim,
                                   out_dim=opt.patch_size*3,
                                   use_encoding=opt.blur_use_encoding,
                                   num_frequencies=opt.blur_num_frequencies,
                                   skip_layer=opt.blur_skip,
                                   nonlinear=opt.blur_last_func).to(device)
    model_blur.load_state_dict(ckp['model_blur_state_dict'])
    
model_noise = None
if 'model_noise_state_dict' in ckp:
    model_noise = modules.SingleBVPNet(type='relu', mode='mlp',  hidden_features=128,
                                       in_features=2, out_features=18, sidelength=image_resolution,
                                       num_frequencies=7).to(device)
    model_noise.load_state_dict(ckp['model_noise_state_dict'])

model_alpha = None
if 'model_alpha_state_dict' in ckp:
    model_alpha = modules.SimpleMLP(hidden_dim=opt.alpha_hidden_dim, 
                                    num_layer=opt.alpha_num_layers,
                                    in_dim=opt.alpha_in_dim,
                                    out_dim=opt.alpha_out_dim, 
                                    use_encoding=opt.alpha_use_encoding, 
                                    num_frequencies=opt.alpha_num_frequencies, 
                                    skip_layer=opt.alpha_skip,
                                    nonlinear=opt.alpha_last_func).to(device)
    model_alpha.load_state_dict(ckp['model_alpha_state_dict'])


# Start main pipeline
root_path = os.path.join(opt.checkpoint_path, opt.experiment_name)
utils.cond_mkdir(root_path)

# Get ground truth and input data
total_input, gt = next(iter(dataloader))
total_input = {key: value.cuda() for key, value in total_input.items()}
total_input['coords'] = total_input['coords'].view(vid_dataset.shape[0], -1, 3)
total_input['coord_idx'] = total_input['coord_idx'].view(vid_dataset.shape[0], -1)
gt = {key: value.cuda() for key, value in gt.items()}
gt['img'] = gt['img'].view(vid_dataset.shape[0], -1, 3)

if vid_dataset.depth is not None:
    depth = ((torch.from_numpy(vid_dataset.depth)- 0.5) / 0.5).to(device)
    if depth.shape[0] != vid_dataset.shape[0]:
        depth = depth.permute(3, 0, 1, 2).unsqueeze(0) # [1, C, D, H, W]
        depth = tfunc.interpolate(depth, size=(vid_dataset.shape[0], depth.shape[-2], depth.shape[-1]), mode='trilinear', align_corners=False)
        depth = depth.squeeze(0).permute(1, 2, 3, 0)
        total_input['depth'] = depth
    total_input['depth'] = total_input['depth'].view(vid_dataset.shape[0], -1, 1)

# Evaluate the trained model
pre_hdr_video = []; pre_ldr_video = []
pre_bg_video = []; pre_fg_video = []
bg_uv = []; fg_uv = []
alpha_video = []
kernel_video = []

N, H, W = vid_dataset.shape

print('Start rendering.')
with torch.no_grad():

    # Learned exposures
    if model_exp is not None:
        learned_exp = model_exp().view(1, 1, -1)
        learned_exp = tfunc.interpolate(learned_exp, size=vid_dataset.shape[0], mode='linear', align_corners=True)
        learned_exp = learned_exp.view(-1, 1, 1).repeat(1, vid_dataset.shape[1],  vid_dataset.shape[2])
    
    if model_focal is not None:
        learned_focal = model_focal.get_minmax().view(1, 1, -1)
        # learned_focal = model_focal().view(1, 1, -1)
        learned_focal = tfunc.interpolate(learned_focal, size=vid_dataset.shape[0], mode='linear', align_corners=True)
        learned_focal = learned_focal.view(-1, 1, 1).repeat(1, vid_dataset.shape[1],  vid_dataset.shape[2])

    for i in range(0, gt['img'].shape[1], opt.chunk):
        model_input = {
            'coords':total_input['coords'][:,i:i+opt.chunk,:]
        }
        
        if vid_dataset.depth is not None:
            model_input.update({
                'depth':total_input['depth'][:,i:i+opt.chunk,:]
            })

        if model_exp is not None:
            model_input.update({'exps':learned_exp.view(vid_dataset.shape[0],-1,1)[:,i:i+opt.chunk,:]})
        else:
            model_input.update({'exps':total_input['exps'].view(vid_dataset.shape[0],-1,1)[:,i:i+opt.chunk,:]})

        if model_focal is not None:
            model_input.update({'focal':learned_focal.view(vid_dataset.shape[0],-1,1)[:,i:i+opt.chunk,:]})
        
        # Step1: map (u,v,t) to (u,v).
        model_input_s = {}
        model_input_d = {}
        model_input_alpha = {}
        coords_kernel = model_input['coords'].clone()
        # Blend kernel patch
        if model_blur is not None and opt.render_focal:
            model_input_patch = utils.get_input_patch(model_input, vid_dataset.shape, patch_size=opt.patch_size)
            patch_size = opt.patch_size
            max_shift = opt.offset_size
            if opt.use_depth:
                kernel_out = model_blur(torch.cat((model_input['focal'], model_input['depth']), -1)).view(N, -1, patch_size*3)
            else:
                kernel_out = model_blur(torch.cat((model_input['focal'], model_input['coords'][..., 1:3]), -1)).view(N, -1, patch_size*3)
    
            model_input_patch['coords'][:, :, 1] += kernel_out[..., patch_size:patch_size*2].permute(
                2, 0, 1).reshape(patch_size * model_input['coords'].shape[0], -1) * (2 / (vid_dataset.shape[1]-1)) * max_shift
            model_input_patch['coords'][:, :, 2] += kernel_out[..., patch_size*2:patch_size*3].permute(
                2, 0, 1).reshape(patch_size * model_input['coords'].shape[0], -1) * (2 / (vid_dataset.shape[2]-1)) * max_shift
            model_input_patch['coords'] = model_input_patch['coords'].view(patch_size, vid_dataset.shape[0], -1, 3)
            model_input_patch['coords'][patch_size // 2] = model_input['coords']
            kernel_weights = kernel_out[:, :, 0:patch_size].permute(2, 0, 1).unsqueeze(-1)
            kernel_video.append(kernel_weights)
        else:
            model_input_patch = model_input
            patch_size = 1
        ref_idx = patch_size // 2

        if model_uv_b is not None:
            input_uv_s = model_uv_b(model_input_patch['coords']) \
                                        .view(model_input['coords'].shape[0]*patch_size, -1, 2)
            model_input_s['coords'] =  (model_input_patch['coords'][..., 1:] + input_uv_s) / 2
            model_output_s = model_atlas_b(model_input_s)
            model_output = model_output_s

            bg_uv.append(model_input_s['coords'].view(patch_size, vid_dataset.shape[0], -1, 2)[ref_idx])
            pre_bg_video.append(torch.exp(model_output_s['model_out'].view(patch_size, vid_dataset.shape[0], -1, 2)[ref_idx]))


        if model_uv_f is not None:
            input_uv_d = model_uv_f(model_input_patch['coords']) \
                                    .view(model_input['coords'].shape[0]*patch_size, -1, 2)
            model_input_d['coords'] =  (model_input_patch['coords'][..., 1:] + input_uv_d) / 2
            model_output_d = model_atlas_f(model_input_d)
            if model_alpha is not None:
                # alpha = torch.sigmoid(model_alpha(model_input_alpha)['model_out'])
                alpha = model_alpha(model_input_patch['coords']).view(model_input['coords'].shape[0]*patch_size, -1, 1)
            else:
                alpha = torch.sigmoid(model_output_d['model_out'][..., -1:])
            model_output['model_out'] = (1-alpha)*model_output_s['model_out'] + alpha*model_output_d['model_out'][..., 0:3]

            fg_uv.append(model_input_d['coords'])
            alpha_video.append(alpha)
            pre_fg_video.append(torch.exp(alpha*model_output_d['model_out'][..., 0:3]))

        if model_uv_b is None and model_uv_f is None:
            model_input_patch['coords'] = model_input_patch['coords'][..., 1:]
            model_output = model_atlas_b(model_input_patch)
        
        if model_blur is not None and opt.render_focal:
            model_output['model_out'] = torch.sum(kernel_weights * model_output['model_out'].view(patch_size, vid_dataset.shape[0], -1, 3), 0)

        # Step3: map HDR rgb to LDR rgb
        if model_tm != None:
            ldr_pixels = torch.clamp(model_tm(model_output['model_out'].view(-1, 3), 
                                            model_input['exps'].reshape(-1, 1)).view(vid_dataset.shape[0], -1, 3), -1, 1)
            ldr_pixels = ldr_pixels / 2 + 0.5
            pre_ldr_video.append(ldr_pixels)
            pre_hdr_video.append(torch.exp(model_output['model_out']))
        else:
            pre_ldr_video.append(torch.clamp(model_output['model_out']/2+0.5, 0, 1))

    # Save background and foreground atlases
    if opt.save_atlas:
        print('Saving atlases...')
        utils.save_atlas(model_b=model_atlas_b, model_f=model_atlas_f, atlase_h=opt.seq_size[1], atlase_w=opt.seq_size[2], root_path=root_path, exp_name=opt.experiment_name)
        print('Atlases saved.')

    # Draw the CRF curve.
    if model_tm != None:
        print('Saving CRF...')
        utils.save_CRF(model_tm=model_tm, root_path=root_path, exp_name=opt.experiment_name)
        print('CRF saved.')

    # Render video from atlases.
    if opt.edited_video:
        print('Render video form atalses...')
        utils.render_video(bg_uv=torch.cat(bg_uv, 1), fg_uv=torch.cat(fg_uv, 1), alpha=torch.cat(alpha_video, 1), shape=vid_dataset.shape, 
                        root_path=root_path, exp_name=opt.experiment_name, atlas_name=opt.atlas_name)
        print('Rendered.')

    # Save reconstructed videos.
    print('Save all reconstructed videos...')
    if opt.save_gt:
        gt = gt['img'].reshape(list(vid_dataset.shape) + [3])
        gt = gt / 2 + 0.5
        prevideo = gt.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_gt.mp4'), utils.to8b(prevideo), fps=20)
        if opt.save_frame:
            print('Save gt frames...')
            for i in range(prevideo.shape[0]):
                imageio.imwrite(os.path.join(root_path, 'gt_%03d.png'%i), utils.to8b(prevideo[i]))

    if pre_ldr_video != []:
        pre_ldr_video = torch.cat(pre_ldr_video, 1).reshape(list(vid_dataset.shape) + [3])
        prevideo = pre_ldr_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_ldr_video.mp4'), utils.to8b(prevideo), fps=10, quality=9)
        if opt.save_frame:
            print('Save ldr frames...')
            for i in range(prevideo.shape[0]):
                imageio.imwrite(os.path.join(root_path, 'ldr_%03d.png'%i), utils.to8b(prevideo[i]))

    if pre_hdr_video != []:
        pre_hdr_video = torch.cat(pre_hdr_video, 1).reshape(list(vid_dataset.shape) + [3])
        prevideo = pre_hdr_video.detach().cpu().numpy()
        max_radiance = np.max(prevideo) 
        print(max_radiance)
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_hdr_video.gif'), utils.tonemap(prevideo / max_radiance), fps=20)
        
        if opt.evaluate_hdr:
            utils.evaluate_hdr(prevideo, video_path, root_path)

        if opt.save_frame:
            print('Save hdr frames...')
            for i in range(prevideo.shape[0]):
                imageio.imwrite(os.path.join(root_path, 'hdr_%03d.hdr'%i), prevideo[i])
    
    # if kernel_video != []:
    #     pre_kernel_video = torch.cat(kernel_video, -2).reshape([opt.patch_size, H, W, 1])
    #     prevideo = pre_kernel_video.detach().cpu().numpy()
    #     prevideo /= np.max(prevideo)
    #     imageio.mimsave(os.path.join(root_path, opt.experiment_name+'kernel.mp4'), utils.to8b(prevideo), fps=1)
    
    if pre_bg_video != []:
        pre_bg_video = torch.cat(pre_bg_video, 1).reshape(list(vid_dataset.shape) + [3])
        prevideo = pre_bg_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_bg_video.gif'), utils.tonemap(prevideo / max_radiance), fps=20)

    if pre_fg_video != []:
        pre_fg_video = torch.cat(pre_fg_video, 1).reshape(list(vid_dataset.shape) + [3])
        prevideo = pre_fg_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_fg_video.gif'), utils.tonemap(prevideo / max_radiance), fps=20)

    if alpha_video != []:
        alpha_video = torch.cat(alpha_video, 1).reshape(list(vid_dataset.shape) + [1])
        prevideo = alpha_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_alpha.gif'), utils.to8b(prevideo), fps=20)
        if opt.save_frame:
            print('Save alpha frames...')
            for i in range(prevideo.shape[0]):
                imageio.imwrite(os.path.join(root_path, 'alpha_%03d.png'%i), utils.to8b(prevideo[i]))

    if fg_uv != []:
        fg_uv_video = torch.cat(fg_uv, 1).reshape(list(vid_dataset.shape) + [2])
        max_uv = torch.max(fg_uv_video)
        min_uv = torch.min(fg_uv_video)
        fg_uv_video = (fg_uv_video-min_uv) / (max_uv-min_uv) # uv in [0, 1]
        fg_uv_video = torch.cat([fg_uv_video, torch.zeros(list(vid_dataset.shape) + [1]).cuda()], -1) # (u v) -> (u, v, 0)
        fg_uv_video = fg_uv_video * alpha_video
        prevideo = fg_uv_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_fg_uv.mp4'), utils.to8b(prevideo), fps=20)

    if bg_uv != []:
        bg_uv_video = torch.cat(bg_uv, 1).reshape(list(vid_dataset.shape) + [2])
        max_uv = torch.max(bg_uv_video)
        min_uv = torch.min(bg_uv_video)
        bg_uv_video = (bg_uv_video-min_uv) / (max_uv-min_uv) # uv in [0, 1]
        bg_uv_video = torch.cat([bg_uv_video, torch.zeros(list(vid_dataset.shape) + [1]).cuda()], -1) # (u v) -> (u, v, 0)
        prevideo = bg_uv_video.detach().cpu().numpy()
        imageio.mimsave(os.path.join(root_path, opt.experiment_name+'_bg_uv.mp4'), utils.to8b(prevideo), fps=20)

print('Done with testing.')

