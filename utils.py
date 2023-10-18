from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import os
import csv
import diff_operators
from torchvision.utils import make_grid, save_image
import skimage.measure
import cv2
import meta_modules
import scipy.io.wavfile as wavfile
import cmapy
import imageio
import torch.optim as optim
import torch.distributions as tdist
import torch.nn.functional as F

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
tonemap = lambda x : (np.log(x * 5000 + 1 ) / np.log(5000 + 1) * 255).astype(np.uint8)

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)


def densely_sample_activations(model, num_dim=1, num_steps=int(1e6)):
    input = torch.linspace(-1., 1., steps=num_steps).float()

    if num_dim == 1:
        input = input[...,None]
    else:
        input = torch.stack(torch.meshgrid(*(input for _ in num_dim)), dim=-1).view(-1, num_dim)

    input = {'coords':input[None,:].cuda()}
    with torch.no_grad():
        activations = model.forward_with_activations(input)['activations']
    return activations


def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_video_summary(vid_dataset, results_dir, models, gt, model_output, writer, total_steps, opt, prefix='train_'):

    N, H, W = vid_dataset.shape
    frames = list(range(N))
    Nslice = N*5

    with torch.no_grad():
        coords = [dataio.get_mgrid((1, H, W), dim=3).view(-1, 3)[None,...].cuda() for f in frames]
        exps = torch.zeros([len(frames),H, W, 1])
        depth = torch.zeros([len(frames),H, W, 1])
        if models['exp'] is not None:
            learned_exp = models['exp']().view(-1, 1, 1, 1).repeat(1, H, W, 1)
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (N - 1 + 1e-16) - 0.5) * 2
            if models['exp'] is not None:
                exps[idx] = learned_exp[f,:,:,:]
            else:
                exps[idx] = torch.from_numpy(vid_dataset.exps[f,:,:,:])
        coords = torch.cat(coords, dim=0)
        exps = exps.view(len(frames), -1, 1).cuda()
        if vid_dataset.depth is not None:
            if models['depth'] is not None:
                coord_idx = torch.arange(N*H*W)
                depth = models['depth'](coord_idx).view(N,-1, 1)
            else:
                depth = ((torch.from_numpy(vid_dataset.depth)- 0.5) / 0.5).view(len(frames), -1, 1).cuda()

        output = torch.zeros(coords.shape)
        output_hdr = torch.zeros(coords.shape)
        alpha_d = torch.zeros(list(coords.shape[:-1])+[1])
        uv_b = torch.zeros(list(coords.shape[:-1])+[2])
        uv_f = torch.zeros(list(coords.shape[:-1])+[2])
        alpha_d = torch.zeros(list(coords.shape[:-1])+[1])
        split = int(coords.shape[1] / Nslice)
        model_input = {}

        for i in range(Nslice):
            
            model_input['coords'] = coords[:, i*split:(i+1)*split, :]
            if models['blur'] is not None and opt.render_focal:
                model_input['depth'] = depth[:, i*split:(i+1)*split, :]
                model_input_patch = get_input_patch(model_input, vid_dataset.shape, patch_size=opt.patch_size)
                patch_size = opt.patch_size
                max_shift = opt.offset_size
                learned_focal = models['focal']().view(-1, 1, 1).repeat(1, H,  W)
                input_focal = learned_focal.view(N, -1, 1)[:, i*split:(i+1)*split, :]
                if opt.use_depth:
                    kernel_out = models['blur'](torch.cat((input_focal, model_input['depth']), -1)).view(coords.shape[0], -1, patch_size*3)
                else:
                    kernel_out = models['blur'](torch.cat((input_focal, model_input['coords'][..., 1:3]), -1)).view(coords.shape[0], -1, patch_size*3)
                model_input_patch['coords'][:,:,1] += kernel_out[:,:, patch_size:patch_size*2].permute(2, 0, 1).reshape(patch_size* coords.shape[0], -1)* (2 / (H-1)) * max_shift
                model_input_patch['coords'][:,:,2] += kernel_out[:,:, patch_size*2:patch_size*3].permute(2, 0, 1).reshape(patch_size* coords.shape[0], -1) * (2 / (W-1)) * max_shift
                kernel_weights = kernel_out[:, :, 0:patch_size].permute(2, 0, 1).unsqueeze(-1)
            else:
                model_input_patch = model_input
                patch_size = 1
            ref_idx = patch_size // 2

            # Map (u,v,t) to (u,v).
            source_uv = model_input_patch['coords'][:, :, 1:]
            if models['uv_b'] is not None:
                delta_uv = models['uv_b'](model_input_patch['coords']).view(coords.shape[0]*patch_size, -1, 2)
                input_coords_s = (source_uv + delta_uv) / 2
                uv_b[:, i*split:(i+1)*split, :] = input_coords_s.view(patch_size, coords.shape[0], -1, 2)[ref_idx].cpu()

            else:
                input_coords_s= source_uv

            if models['uv_f'] is not None:
                delta_uv = models['uv_f'](model_input_patch['coords']).view(coords.shape[0]*patch_size, -1, 2)
                input_coords_d = (source_uv + delta_uv) / 2
                uv_f[:, i*split:(i+1)*split, :] = input_coords_d.view(patch_size, coords.shape[0], -1, 2)[ref_idx].cpu()

            # Map (u,v) to HDR rgb.
            pred_s = models['atlas_b']({'coords': input_coords_s})['model_out']
            if models['uv_f'] is not None:
                pred_d = models['atlas_f']({'coords': input_coords_d})['model_out']
            
            # Blend static layer and dynamic layer.
            pred = pred_s
            if models['uv_f'] is not None:
                if models['alpha'] is not None:
                    alpha = models['alpha'](model_input_patch['coords']).view(coords.shape[0]*patch_size, -1, 1)
                else:
                    alpha = torch.sigmoid(pred_d[..., -1:])
                pred = (1-alpha)*pred_s + alpha*pred_d[..., 0:3]
                alpha = alpha.view(patch_size, coords.shape[0], -1, 1)[ref_idx] # get the alpha of center point
                alpha_d[:, i*split:(i+1)*split, :] = alpha.cpu()
            
            # Blend the kernel patch
            output_hdr[:, i*split:(i+1)*split, :] = torch.exp(pred.view(patch_size, coords.shape[0], -1, 3)[ref_idx]).cpu()
            if models['blur'] is not None and opt.render_focal:
                deblur_pred =  pred.view(patch_size, coords.shape[0], -1, 3)[ref_idx]
                pred = torch.sum(kernel_weights * pred.view(patch_size, coords.shape[0], -1, 3), 0)
                output_hdr[:, i*split:(i+1)*split, :] = torch.exp(deblur_pred).cpu()

            noise = None
            # if models['noise'] is not None:
            #     noise = models['noise'](model_input)['model_out'].view(-1, 3)

            # Map HDR rgb to LDR rgb.
            if models['tm'] is not None:
                pred = models['tm'](pred.view(-1, 3), exps[:,i*split:(i+1)*split, :].reshape(-1, 1), noise).view(len(frames), -1, 3)
            output[:, i*split:(i+1)*split, :] =  pred.cpu()
            
    pred_vid = output.view(len(frames), H, W, 3) / 2 + 0.5
    # pred_vid = torch.clamp(pred_vid, 0, 1)
    pre_hdr = output_hdr.view(len(frames), H, W, 3)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :3])
    alpha_vid = alpha_d.view(len(frames), H, W, 1)

    uv_f = uv_f.view(len(frames), H, W, 2)
    max_uv = torch.max(uv_f)
    min_uv = torch.min(uv_f)
    uv_f = (uv_f-min_uv) / (max_uv-min_uv + 1e-16) # uv in [0, 1]
    uv_f = torch.cat([uv_f, torch.zeros_like(uv_f[...,0:1])], -1) * alpha_vid
    
    uv_b = uv_b.view(len(frames), H, W, 2)
    min_max_summary(prefix + 'uv_coords', uv_b, writer, total_steps)
    max_uv = torch.max(uv_b)
    min_uv = torch.min(uv_b)
    uv_b = (uv_b-min_uv) / (max_uv-min_uv + 1e-16) # uv in [0, 1]
    uv_b = torch.cat([uv_b, torch.zeros_like(uv_b[...,0:1])], -1)

    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))

    # Save results
    imageio.mimsave(os.path.join(results_dir, 'ldr_gt_video.mp4'), to8b(gt_vid.numpy()), fps=1)
    imageio.mimsave(os.path.join(results_dir, 'ldr_video.mp4'), to8b(pred_vid.numpy()), fps=1)
    for i in range(pred_vid.numpy().shape[0]):
        imageio.imwrite(os.path.join(results_dir, 'ldr_%03d.png'%i), to8b(pred_vid.numpy()[i]))
    imageio.mimsave(os.path.join(results_dir, 'alpha_video.mp4'), to8b(alpha_vid.numpy()), fps=1)
    imageio.mimsave(os.path.join(results_dir, 'uv_f.mp4'), to8b(uv_f.numpy()), fps=1)
    imageio.mimsave(os.path.join(results_dir, 'uv_b.mp4'), to8b(uv_b.numpy()), fps=1)
    imageio.mimsave(os.path.join(results_dir, 'hdr_video.mp4'), tonemap(pre_hdr.numpy() / np.max(pre_hdr.numpy())), fps=1)
    with torch.no_grad():
        if models['uv_b'] is not None:
            save_atlas(model_b=models['atlas_b'], model_f=models['atlas_f'], atlase_h=H, atlase_w=W, root_path=results_dir, exp_name='')
        if models['tm'] is not None:
            save_CRF(model_tm=models['tm'], root_path=results_dir, exp_name='')

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)
    alpha_vid = alpha_vid.permute(0, 3, 1, 2)[:4]
    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)[:4]

    # writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
    #                  global_step=total_steps)
    # writer.add_image(prefix + 'alpha', make_grid(alpha_vid), global_step=total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)
    with open(os.path.join(results_dir, 'psnr.csv'),"a") as csvfile: 
        writer_cvs = csv.writer(csvfile)
        writer_cvs.writerows([[psnr.numpy(), total_steps]])
    
    min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)


def tonemapSimple(x):
    return (torch.exp(x) / (torch.exp(x) + 1)) ** (1 / 2.2)

def warm_tm(model, pretrain_iters=1000, device='cuda'):
    optimizer_crf = torch.optim.Adam(model.parameters(), lr=5e-4)

    for i in range(pretrain_iters):
        optimizer_crf.zero_grad()
        rand_radiance = torch.rand(1024, 3, requires_grad=True).to(device) * 3 # radiance in [0, 3]
        rand_exposure = torch.rand(1024, 1, requires_grad=True).to(device) * 6 - 3.0 # radiance in [-3, 3]

        gt_rgb = tonemapSimple(rand_radiance + rand_exposure)

        rgb_l = model(rand_radiance, rand_exposure)
        rgb_l = rgb_l / 2 + 0.5

        loss = (rgb_l - gt_rgb).norm(dim=1).mean()
        print(f"step {i} pre-train loss: {loss.item()}")
        loss.backward()
        optimizer_crf.step()
    return model

def warm_mapping(model_mapping, shape, uv_mapping_scale=0.8, pretrain_iters=100, device='cuda'):
    frames_num, img_h, img_w = shape
    optimizer_mapping = optim.Adam(model_mapping.parameters(), lr=1e-4)
    for i in range(pretrain_iters):
        for f in range(frames_num):
            i_s_int = torch.randint(img_h, (np.int64(10000), 1))
            j_s_int = torch.randint(img_w, (np.int64(10000), 1))

            i_s = i_s_int / (img_h-1) * 2 - 1
            j_s = j_s_int / (img_w-1) * 2 - 1
            t_s = (f / (frames_num-1 + 1e-16) * 2 - 1) * torch.ones_like(i_s)

            coords = torch.cat((t_s, i_s, j_s),dim=1).to(device)
            uv_temp = model_mapping(coords)

            model_mapping.zero_grad()

            loss = (coords[:, 1:] * uv_mapping_scale - uv_temp).norm(dim=1).mean()
            print(f"pre-train loss: {loss.item()}")
            loss.backward()
            optimizer_mapping.step()
    return model_mapping

def warm_alpha(model_alpha, dataset, pretrain_iters=1000, device='cuda'):
    frames_num, img_h, img_w = dataset.shape
    optimizer_mapping = optim.Adam(model_alpha.parameters(), lr=1e-4)
    mask_gt = torch.from_numpy(dataset.vid[..., 3:4]).to(device)

    y = torch.linspace(-1, 1, img_h)
    x = torch.linspace(-1, 1, img_w)
    y, x = torch.meshgrid(y, x)
    xy_grid = torch.stack([y, x], -1).reshape(-1, 2).to(device)

    for i in range(pretrain_iters):
        for f in range(frames_num):
            idx = torch.randint(xy_grid.shape[0], (10000,))

            i_s = xy_grid[idx, 0:1]
            j_s = xy_grid[idx, 1:2]
            t_s = (f / (frames_num-1) * 2 - 1) * torch.ones_like(i_s)

            coords = torch.cat((t_s, i_s, j_s),dim=1).to('cuda')
            alpha_temp = model_alpha(coords)

            model_alpha.zero_grad()

            loss = (mask_gt[f].view(-1, 1)[idx] - alpha_temp).norm(dim=1).mean()
            print(f"pre-train loss: {loss.item()}, max alpha: {torch.max(alpha_temp).item()}")
            loss.backward()
            optimizer_mapping.step()
    return model_alpha

def save_atlas(model_b, model_f, atlase_h=360, atlase_w=640, batch_size=360*640, root_path=None, exp_name=None):
    x = torch.linspace(-1, 1, atlase_h)
    y = torch.linspace(-1, 1, atlase_w)
    x, y = torch.meshgrid(x, y)
    idx_grid = torch.stack([x, y], -1).reshape(1, -1, 2).cuda()

    bg_atlas = []
    fg_atlas = []
    for i in range(0,atlase_h*atlase_w, batch_size):
        bg_atlas.append(torch.exp(model_b({'coords':idx_grid[:,i:i+batch_size,:]})['model_out']))
        if model_f is not None:
            fg_atlas.append(torch.exp(model_f({'coords':idx_grid[:,i:i+batch_size,:]})['model_out'][..., 0:3]))
    bg_atlas = torch.cat(bg_atlas, 1).view(atlase_h, atlase_w, 3).detach().cpu().numpy()
    if len(fg_atlas) != 0:
        fg_atlas = torch.cat(fg_atlas, 1).view(atlase_h, atlase_w, 3).detach().cpu().numpy()

    max_radiance = np.max(bg_atlas)
    print(max_radiance)
    imageio.imwrite(os.path.join(root_path, exp_name + '_bg_atlas.png'), tonemap(bg_atlas/max_radiance))
    imageio.imwrite(os.path.join(root_path, exp_name + '_bg_atlas.hdr'), bg_atlas)
    if len(fg_atlas) != 0:
        imageio.imwrite(os.path.join(root_path, exp_name + '_fg_atlas.png'), tonemap(fg_atlas/max_radiance))
        imageio.imwrite(os.path.join(root_path, exp_name + '_fg_atlas.hdr'), fg_atlas)
    print('Done with atlases.')


def save_CRF(model_tm, root_path, exp_name):
    ln_x = torch.linspace(-10, 10, 1000).reshape([-1, 1]).repeat(1, 3).cuda()
    ln_t = torch.zeros([ln_x.shape[0], 1]).cuda()
    with torch.no_grad():
        y = torch.clamp(model_tm(ln_x, ln_t).view(-1, 3), -1, 1)
        y = y / 2 + 0.5

    x = ln_x.cpu().numpy()
    y = y.detach().cpu().numpy()
    np.save(os.path.join(root_path, 'CRF_x.npy'), x)
    np.save(os.path.join(root_path, 'CRF_y.npy'), y)
    plt.figure()
    plt.xlabel("lnx")
    plt.ylabel("pixel value")
    plt.plot(x[:,:1],y[:,0:1], color='r', label='CRF red')
    plt.plot(x[:,:1],y[:,1:2], color='g', label='CRF green')
    plt.plot(x[:,:1],y[:,2:3], color='b', label='CRF blue')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root_path, exp_name + '_CRF.png'))
    plt.close()
    return


def save_noise(model_noise, root_path, exp_name):
    ln_x = torch.linspace(-5, 3, 1000).reshape([-1, 1]).cuda()
    with torch.no_grad():
        y = model_noise(ln_x).view(-1, 1)
    x = ln_x.cpu().numpy()
    y = y.detach().cpu().numpy()
    plt.figure()
    plt.xlabel("lnx")
    plt.ylabel("pixel value")
    plt.plot(x,y, color='r')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root_path, exp_name + '_noise.png'))
    plt.close()
    return

def get_rigidity_loss(src_coords, src_uv, model_mapping, shape, derivative_amount=1, uv_mapping_scale=1, mask=1):

    step_y = 2 / (shape[1] - 1) * derivative_amount
    step_x = 2 / (shape[2] - 1) * derivative_amount
    # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_p:
    t_path = torch.cat((src_coords[:,:,0:1], src_coords[:,:,0:1]), 1)
    y_patch = torch.cat((src_coords[:,:,1:2]-step_y, src_coords[:,:,1:2]), 1)
    x_patch = torch.cat((src_coords[:,:,2:3], src_coords[:,:,2:3]-step_x), 1)
    tyx_p = torch.cat((t_path, y_patch, x_patch), dim=-1).squeeze()

    uv_p = (model_mapping(tyx_p) + tyx_p[:, 1:3]) / 2
    u_p = uv_p[:, 0].view(2, -1)  # u_p[0,:]= u(t,y-derivative_amount,x).  u_p[1,:]= u(t, y, x-derivative_amount)
    v_p = uv_p[:, 1].view(2, -1)  # v_p[0,:]= u(t,y-derivative_amount,x).  v_p[1,:]= v(t, y, x-derivative_amount)

    src_uv = src_uv.squeeze()
    u_p_d_ = src_uv[:, 0].unsqueeze(
        0) - u_p  # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    v_p_d_ = src_uv[:, 1].unsqueeze(
        0) - v_p  # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).

    # to match units: 1 in uv coordinates is resx/2 in image space.
    du_dx = u_p_d_[1, :] * (shape[1]-1) / 2
    du_dy = u_p_d_[0, :] * (shape[1]-1) / 2
    dv_dy = v_p_d_[0, :] * (shape[2]-1) / 2
    dv_dx = v_p_d_[1, :] * (shape[2]-1) / 2

    jacobians = torch.cat((torch.cat((du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)), dim=2),
                           torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2)),
                          dim=1)
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to('cuda')
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    # See Equation (9) in the paper:
    rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + (JTJinv ** 2).sum(1).sum(1).sqrt()
    rigidity_loss = rigidity_loss * mask

    return rigidity_loss.mean()

def flow_loss(model_mapping, coords, flow, shape, uv, alpha=None, use_alpha=False):
    step_t = 2 / (shape[0] -1)

    if uv.shape[0] != 1:
        uv = (model_mapping(coords) + coords[..., 1:3]) / 2

    # Foreword flow
    step_y = 2 / (shape[1] - 1) * flow[..., 1:2]
    step_x = 2 / (shape[2] - 1) * flow[..., 0:1]
    fore_coords = torch.cat([coords[..., 0:1]+step_t, coords[..., 1:2] + step_y, coords[..., 2:3] + step_x], -1)
    fore_uv = (model_mapping(fore_coords) + fore_coords[..., 1:3]) / 2
    mask = flow[..., 2:3]
    if use_alpha:
        fore_loss = (torch.norm(uv-fore_uv, dim=-1) ** 2 * alpha * mask).mean()
    else:
        fore_loss = (torch.norm(uv-fore_uv, dim=-1) ** 2 * mask).mean()

    # Backword flow
    step_y = 2 / (shape[1] - 1) * flow[..., 4:5]
    step_x = 2 / (shape[2] - 1) * flow[..., 3:4]
    back_coords = torch.cat([coords[..., 0:1]-step_t, coords[..., 1:2] + step_y, coords[..., 2:3] + step_x], -1)
    back_uv = (model_mapping(back_coords) + back_coords[..., 1:3]) / 2
    mask = flow[..., 5:6]
    if use_alpha:
        back_loss = (torch.norm(uv-back_uv, dim=-1) ** 2 * alpha * mask).mean()
    else:
        back_loss = (torch.norm(uv-back_uv, dim=-1) ** 2 * mask).mean()
    return (fore_loss + back_loss) / 2.


def flow_loss_zeros(model_mapping, coords, shape, uv, alpha=0, use_alpha=True):
    step_t = 2 / (shape[0] -1 + 1e-16)

    # Foreword flow
    fore_coords = torch.cat([coords[..., 0:1]+step_t, coords[..., 1:2], coords[..., 2:3]], -1)
    fore_uv = (model_mapping(fore_coords) + fore_coords[..., 1:3]) / 2
    if use_alpha:
        fore_loss = (torch.norm(uv-fore_uv, dim=-1) ** 2 * alpha).mean()
    else:
        fore_loss = (torch.norm(uv-fore_uv, dim=-1) ** 2).mean()

    # Backword flow
    back_coords = torch.cat([coords[..., 0:1]-step_t, coords[..., 1:2], coords[..., 2:3]], -1)
    back_uv = (model_mapping(back_coords) + back_coords[..., 1:3]) / 2
    if use_alpha:
        back_loss = (torch.norm(uv-back_uv, dim=-1) ** 2 * alpha).mean()
    else:    
        back_loss = (torch.norm(uv-back_uv, dim=-1) ** 2).mean()

    return (fore_loss + back_loss) / 2.

def flow_alpha_loss(model_alpha, coords, flow, shape, alpha):
    step_t = 2 / (shape[0] -1)

    # Foreword flow
    step_y = 2 / (shape[1] - 1) * flow[..., 1:2]
    step_x = 2 / (shape[2] - 1) * flow[..., 0:1]
    fore_coords = torch.cat([coords[..., 0:1]+step_t, coords[..., 1:2] + step_y, coords[..., 2:3] + step_x], -1)
    fore_alpha = model_alpha(fore_coords)
    mask = flow[..., 2:3]
    fore_loss = (torch.norm(alpha-fore_alpha, dim=-1) ** 2 * mask).mean()

    # Backword flow
    step_y = 2 / (shape[1] - 1) * flow[..., 4:5]
    step_x = 2 / (shape[2] - 1) * flow[..., 3:4]
    back_coords = torch.cat([coords[..., 0:1]-step_t, coords[..., 1:2] + step_y, coords[..., 2:3] + step_x], -1)
    back_alpha = model_alpha(back_coords)
    mask = flow[..., 5:6]
    back_loss = (torch.norm(alpha-back_alpha, dim=-1) ** 2 * mask).mean()

    return (fore_loss + back_loss) / 2.


def smooth_loss(model, coords, rgb, exps, shape, sigma=2, alpha=None, use_alpha=True):

    step_y = 2 / (shape[1] - 1)
    step_x = 2 / (shape[2] - 1)
    exps = torch.Tensor([2]).cuda()*exps
    rgb = torch.exp(rgb)*exps
    # y direction
    coords_dy = torch.cat([coords[..., 0:1] + step_y, coords[..., 1:2]], -1)
    rgb_dy = torch.exp(model({'coords':coords_dy})['model_out'])*exps

    dvalue = torch.abs(rgb-rgb_dy)
    weights = torch.exp(-dvalue/(sigma**2))
    dy_grad = torch.mean(dvalue * weights )

    # x direction
    coords_dx = torch.cat([coords[..., 0:1], coords[..., 1:2]+step_x], -1)
    rgb_dx = torch.exp(model({'coords':coords_dx})['model_out'])*exps

    dvalue = torch.abs(rgb-rgb_dx)
    weights = torch.exp(-dvalue/(sigma**2))
    dx_grad = torch.mean(dvalue * weights )

    return (dy_grad + dx_grad) / 2.


def render_video(bg_uv, fg_uv, alpha, shape, root_path, exp_name, atlas_name=None):
    if atlas_name is None:
        atlas_path = os.path.join(root_path, exp_name)
        bg_atlas = imageio.imread(atlas_path + '_bg_atlas.png') # [H, W, 3]
        fg_atlas = imageio.imread(atlas_path + '_fg_atlas.png') # [H, W, 3]
    else:
        atlas_path = os.path.join(root_path, exp_name)
        bg_atlas = imageio.imread(atlas_path + '_bg_atlas_' + atlas_name + '.jpg') # [H, W, 3]
        fg_atlas = imageio.imread(atlas_path + '_fg_atlas_' + atlas_name + '.jpg') # [H, W, 3]

    bg_atlas = (bg_atlas / 255.).astype(np.float32)
    fg_atlas = (fg_atlas / 255.).astype(np.float32)

    bg_atlas = torch.from_numpy(bg_atlas).cuda().permute(2,0,1).unsqueeze(0) # [1, 3, H, W]
    fg_atlas = torch.from_numpy(fg_atlas).cuda().permute(2,0,1).unsqueeze(0) # [1, 3, H, W]

    bg_atlas = bg_atlas.expand(shape[0], 3, shape[1], shape[2])
    fg_atlas = fg_atlas.expand(shape[0], 3, shape[1], shape[2])

    bg_uv = bg_uv.reshape(list(shape)+[2]) # [N, H, W, 2] 
    fg_uv = fg_uv.reshape(list(shape)+[2]) # [N, H, W, 2] 
    alpha = alpha.reshape(list(shape)+[1]) # [N, H, W, 2] 

    bg_video = F.grid_sample(bg_atlas, bg_uv.flip(-1), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # [N, 3, H, W]
    fg_video = F.grid_sample(fg_atlas, fg_uv.flip(-1), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # [N, 3, H, W]

    rendered_video = (1-alpha) * bg_video + alpha * fg_video
    rendered_video = rendered_video.detach().cpu().numpy()
    imageio.mimsave(os.path.join(root_path, exp_name + '_edited.gif'), to8b(rendered_video), fps=20)
    
    return


def get_input_patch(model_input, shape, patch_size=9):
    step_x = 2 / (shape[2] - 1)
    step_y = 2 / (shape[1] - 1)
    coords_patch = []
    idx_path = []
    model_input_patch = model_input.copy()
    if patch_size == 9:
        x_list = [-1, 0, 1]
        y_list = [-1, 0, 1]
    elif patch_size == 25:
        x_list = [-2, -1, 0, 1, 2]
        y_list = [-2, -1, 0, 1, 2]
    for i in y_list:
        for j in x_list:
            coords_temp = torch.cat((model_input['coords'][..., 0:1],
                                        model_input['coords'][..., 1:2] + i*step_y,  
                                        model_input['coords'][..., 2:3] + j*step_x), -1) 
            idx_temp = torch.cat((torch.ones_like(model_input['coords'][..., 1:2])*i,  
                                    torch.ones_like(model_input['coords'][..., 2:3])*j), -1)
            coords_patch.append(coords_temp)
            idx_path.append(idx_temp)
    
    model_input_patch['coords']  = torch.cat(coords_patch, 0) # [N, B, 3]
    model_input_patch['idx']  = torch.cat(idx_path, 0).cuda() # [N, B, 3]

    return model_input_patch


def gamma_map(x, gamma):
    x = x/2. + 0.5
    x = torch.log(x*gamma + 1 ) / torch.log(gamma + 1)
    x = x*2. - 1.
    return x


def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[0:2]
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= neww / oldw
    flow[:, :, 1] *= newh / oldh
    return flow


def compute_consistency(flow12, flow21):
    wflow21 = warp_flow(flow21, flow12)
    diff = flow12 + wflow21
    diff = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) ** .5
    return diff


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def save_model(models, path):

    save_parameters = {
            'model_state_dict':models['atlas_b'].state_dict()
        }
    if models['tm'] is not None:
        save_parameters.update({'model_tm_state_dict':models['tm'].state_dict()})
    if models['uv_b'] is not None:
        save_parameters.update({'model_mapping_s_state_dict':models['uv_b'].state_dict()})
    if models['uv_f'] is not None:
        save_parameters.update({'model_mapping_d_state_dict':models['uv_f'].state_dict()})
    if models['blur'] is not None:
        save_parameters.update({'model_blur_state_dict':models['blur'].state_dict()})
    if models['atlas_f'] is not None:
        save_parameters.update({'model_f_state_dict':models['atlas_f'].state_dict()})
    if models['alpha'] is not None:
        save_parameters.update({'model_alpha_state_dict':models['alpha'].state_dict()})
    if models['exp'] is not None:
        save_parameters.update({'model_exp_state_dict':models['exp'].state_dict()})
    if models['focal'] is not None:
        save_parameters.update({'model_focal_state_dict':models['focal'].state_dict()})
    if models['noise'] is not None:
        save_parameters.update({'model_noise_state_dict':models['noise'].state_dict()})
    if models['depth'] is not None:
        save_parameters.update({'model_depth_state_dict':models['depth'].state_dict()})
        
    torch.save(save_parameters, path)

    return


def get_dist_depth(depth, center_depth, patch_coords, center_coords, patch_size, max_shift):
    patch_coords = patch_coords.clone().detach().requires_grad_(False)
    center_coords = center_coords.clone().detach().requires_grad_(False)
    center_depth = center_depth.clone().detach().requires_grad_(False)

    patch_coords = patch_coords.view(patch_size, -1, patch_coords.shape[1], patch_coords.shape[2]) # [9, N, B, 3]
    center_coords = center_coords.view(1, -1, center_coords.shape[1], center_coords.shape[2]) # [1, N, B, 3]

    patch_coords[..., 0] = (patch_coords[..., 0] /2 +0.5) * (depth.shape[0] - 1)
    patch_coords[..., 1] = (patch_coords[..., 1] /2 +0.5) * (depth.shape[1] - 1)
    patch_coords[..., 2] = (patch_coords[..., 2] /2 +0.5) * (depth.shape[2] - 1)

    center_coords[..., 0] = (center_coords[..., 0] /2 +0.5) * (depth.shape[0] - 1)
    center_coords[..., 1] = (center_coords[..., 1] /2 +0.5) * (depth.shape[1] - 1)
    center_coords[..., 2] = (center_coords[..., 2] /2 +0.5) * (depth.shape[2] - 1)

    delta_coords = patch_coords[..., 1:3]-center_coords[..., 1:3]
    max_dist = np.sqrt( (max_shift+1)**2 + (max_shift+1)**2)
    dist_to_center = torch.sqrt(torch.sum(delta_coords**2, -1, keepdim=True)) / max_dist
    # patch_coords = torch.round(patch_coords)
    # t_idx = list(patch_coords[..., 0].view(-1).int().clip(0, depth.shape[0] - 1))
    # y_idx = list(patch_coords[..., 1].view(-1).int().clip(0, depth.shape[1] - 1))
    # x_idx = list(patch_coords[..., 2].view(-1).int().clip(0, depth.shape[2] - 1))

    # center_depth = center_depth.view(1, -1, center_depth.shape[1], center_depth.shape[2])
    # depth_patch = depth[t_idx, y_idx, x_idx].reshape(patch_size, -1, center_depth.shape[2], center_depth.shape[3])
    # delta_depth = torch.abs(depth_patch - center_depth) / 2

    # output = torch.cat([dist_to_center, delta_depth], -1)
    # output = output.view(-1, output.shape[2], output.shape[3])

    return dist_to_center


def get_delta_depth(depth, center_depth, patch_coords, center_coords, patch_size, max_shift):
    patch_coords = patch_coords.clone().detach().requires_grad_(False)
    center_coords = center_coords.clone().detach().requires_grad_(False)
    center_depth = center_depth.clone().detach().requires_grad_(False)

    patch_coords = patch_coords.view(1, patch_size, -1, patch_coords.shape[1], patch_coords.shape[2]) # [1, 9, N, B, 3]
    center_depth = center_depth.view(1, -1, center_depth.shape[1], center_depth.shape[2]) # [1, N, B, 1]
    depth = depth.unsqueeze(0).permute(0, 4, 1, 2, 3)

    depth_patch = F.grid_sample(depth, patch_coords.flip(-1), mode='nearest', align_corners=False).squeeze(0).permute(1,2,3,0)
    delta_depth = torch.abs(depth_patch - center_depth) / 2

    return delta_depth


def evaluate_hdr(all_pred_hdr, gt_path, results_path):
    gt_files = [f for f in os.listdir(gt_path) if f.endswith('hdr')]
    gt_hdr = imageio.imread(os.path.join(gt_path, gt_files[0]), 'hdr')
    
    if gt_hdr.shape[0] != all_pred_hdr.shape[1] or gt_hdr.shape[1] != all_pred_hdr.shape[2]:
        gt_hdr = cv2.resize(gt_hdr, (all_pred_hdr.shape[2], all_pred_hdr.shape[1]), interpolation=cv2.INTER_AREA)

    for i in range(len(all_pred_hdr)):
        pred_hdr = all_pred_hdr[i]
        pred_hdr /= np.max(pred_hdr)
        pred_hdr = np.clip(np.mean(gt_hdr)/np.mean(pred_hdr) * pred_hdr, 0, 1)
        psnr = 10*np.log10(1 / np.mean((pred_hdr - gt_hdr)**2))
        print([i, psnr])
        with open(os.path.join(results_path, 'results.csv'),"a") as csvfile: 
            writer_cvs = csv.writer(csvfile)
            writer_cvs.writerows([[i, psnr]])
    
    return
        


def load_data_path(scene):
    video_path = None

    # MFME data (Blender, static)
    if scene == 'BathRoom':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_blender/BathRoom/'
    elif scene == 'Dog':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_blender/Dog/'
    elif scene == 'Sponza':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_blender/Sponza/'
    elif scene == 'YellowDog':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_blender/YellowDog/'

    # MFME data (Nikon, static)
    elif scene == 'BookShelf':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_nikon/BookShelf/'
    elif scene == 'FlowerShelf':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_nikon/FlowerShelf/'
    elif scene == 'MusicTiger':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_nikon/MusicTiger/'
    elif scene == 'Sculpture':
        video_path = '../hdr_video_data_release/multi-focus_multi-exposure/MFME_nikon/Sculpture/'

    # Multi-exposure data (dynamic)
    elif scene == 'LadyEating':
        video_path = '../hdr_video_data_release/multi-exposure/LadyEating/'
    elif scene == 'BabyAtWindow':
        video_path = '../hdr_video_data_release/multi-exposure/BabyAtWindow/'
    elif scene == 'ChristmasRider':
        video_path = '../hdr_video_data_release/multi-exposure/ChristmasRider/'
    elif scene == 'PianoMan':
        video_path = '../hdr_video_data_release/multi-exposure/PianoMan/'
    elif scene == 'SantasLittleHelper':
        video_path = '../hdr_video_data_release/multi-exposure/SantasLittleHelper/'

    # Multi-focus Data (static)
    elif scene == 'MF001':
        video_path = '../hdr_video_data_release/multi-focus/MF001/'
    elif scene == 'MF002':
        video_path = '../hdr_video_data_release/multi-focus/MF002/'
    elif scene == 'MF003':
        video_path = '../hdr_video_data_release/multi-focus/MF003/'
    elif scene == 'MF004':
        video_path = '../hdr_video_data_release/multi-focus/MF004/'
    elif scene == 'MF005':
        video_path = '../hdr_video_data_release/multi-focus/MF005/'
    elif scene == 'MF006':
        video_path = '../hdr_video_data_release/multi-focus/MF006/'
    elif scene == 'MF007':
        video_path = '../hdr_video_data_release/multi-focus/MF007/'
    elif scene == 'MF008':
        video_path = '../hdr_video_data_release/multi-focus/MF008/'

    # Video deblur data
    elif scene == 'Kitchen':
        video_path = '../hdr_video_data_release/video_deblur/Kitchen/'
    elif scene == 'Road':
        video_path = '../hdr_video_data_release/video_deblur/Road/'

    # Video HDR imaging data
    elif scene == 'Night':
        video_path = '../hdr_video_data_release/video_hdr/Night/'
    elif scene == 'Worker':
        video_path = '../hdr_video_data_release/video_hdr/Worker/'

    return video_path