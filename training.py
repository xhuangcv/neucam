'''Implements a generic training loop.
'''

from ast import Not
from xml.parsers.expat import model
# from turtle import distance
import torch
import utils
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
import torch.distributions as tdist
import time
import numpy as np
import os
import shutil


def train(models, train_dataloader, model_dir, loss_fn, summary_fn, val_dataloader=None, loss_schedules=None, video_shape=None, opt=None):
    
    epochs = opt.num_epochs
    lr = opt.lr
    steps_til_summary = opt.steps_til_summary
    epochs_til_checkpoint=opt.epochs_til_ckpt
    flow_weights = 1000
    start_weight = flow_weights
    use_mask = False
    

    opt_parameters = [{'params':models['atlas_b'].parameters()}]
    if models['tm'] != None:
        opt_parameters.append({'params':models['tm'].parameters()})
    if models['uv_b'] != None:
        opt_parameters.append({'params':models['uv_b'].parameters()})
    if models['uv_f'] != None:
        opt_parameters.append({'params':models['uv_f'].parameters()})
    if models['blur'] != None:
        opt_parameters.append({'params':models['blur'].parameters()})
    if models['atlas_f'] != None:
        opt_parameters.append({'params':models['atlas_f'].parameters()})
    if models['alpha'] != None:
        opt_parameters.append({'params':models['alpha'].parameters()})
    if models['exp'] != None:
        opt_parameters.append({'params':models['exp'].parameters()})
    if models['focal'] != None:
        opt_parameters.append({'params':models['focal'].parameters()})
    if models['noise'] != None:
        opt_parameters.append({'params':models['noise'].parameters()})
    if models['depth'] != None:
        opt_parameters.append({'params':models['depth'].parameters()})

    optim = torch.optim.Adam(lr=lr, params=opt_parameters)

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    results_dir = os.path.join(model_dir, 'results')
    utils.cond_mkdir(results_dir)

    writer = SummaryWriter(summaries_dir)

    # Create log dir and copy the config file
    f = os.path.join(model_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(opt):
            attr = getattr(opt, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if opt.config_filepath is not None:
        f = os.path.join(model_dir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(opt.config_filepath, 'r').read())
            file.write('checkpoint_path = ' + model_dir)

    total_steps = 0
    # total = len(train_dataloader) * epochs
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                utils.save_model(models, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                ######################### Main Pipeline #######################
                # Generate neighbor coords
                if models['depth'] is not None:
                    model_input['depth'] = models['depth'](model_input['coord_idx'])
                if models['blur'] is not None:
                    
                    model_input_patch = utils.get_input_patch(model_input, video_shape, patch_size=opt.patch_size)
                    patch_size = opt.patch_size
                    max_shift = opt.offset_size
                    learned_focal = models['focal']().view(-1, 1, 1).repeat(1, video_shape[1],  video_shape[2])
                    if opt.patch_sample:
                        input_focal = learned_focal[:, model_input['coord_idx'][0][0], model_input['coord_idx'][0][1]].view(-1, 1).unsqueeze(0)
                    else:
                        input_focal = learned_focal.view(-1, 1)[model_input['coord_idx'][0]].unsqueeze(0)

                    if opt.use_depth:
                        kernel_out = models['blur'](torch.cat((input_focal, model_input['depth']), -1))
                    else:
                        kernel_out = models['blur'](torch.cat((input_focal, model_input['coords'][..., 1:3]), -1))
                    
                    y_offset = kernel_out[:, patch_size:patch_size*2].permute(1, 0) 
                    x_offset = kernel_out[:, patch_size*2:patch_size*3].permute(1, 0) 
                    
                    model_input_patch['coords'][:,:,1] += y_offset * (2 / (video_shape[1]-1)) * max_shift
                    model_input_patch['coords'][:,:,2] += x_offset * (2 / (video_shape[2]-1)) * max_shift
                    model_input_patch['coords'][patch_size // 2] = model_input['coords']
                    kernel_weights = kernel_out[:, 0:patch_size].permute(1, 0).unsqueeze(-1)        
                else:
                    model_input_patch = model_input
                    patch_size = 1
                ref_idx = patch_size // 2
                    
                # Map (x,y,t) to (u,v).
                uv_b = {}
                uv_f = {}
                source_uv = model_input_patch['coords'][..., 1:]
                uv_b['coords'] = source_uv # when don't use uv mapping model
                if models['uv_b'] is not None:
                    delta_uv  = models['uv_b'](model_input_patch['coords']).view(patch_size, -1, 2)  
                    uv_b['coords']  = (delta_uv + source_uv) / 2  
                
                if models['uv_f'] is not None:
                    delta_uv = models['uv_f'](model_input_patch['coords']).view(patch_size, -1, 2)
                    uv_f['coords'] = (delta_uv + source_uv) / 2

                # Map (u,v) or (u,v,t) to HDR rgb values.
                hdr_rgb_b = models['atlas_b'](uv_b)
                if models['uv_f'] is not None:
                    hdr_rgb_f = models['atlas_f'](uv_f)

                # Blend the static scene and dynamic scene.
                model_output = hdr_rgb_b
                if models['atlas_f'] is not None:
                    if models['alpha'] is not None:
                        alpha = models['alpha'](model_input_patch['coords']).view(patch_size, -1, 1)
                    else:
                        alpha = torch.sigmoid(hdr_rgb_f['model_out'][..., -1:])
                    model_output['model_out'] = (1-alpha)*hdr_rgb_b['model_out'] + alpha*hdr_rgb_f['model_out'][..., 0:3]
                    alpha = alpha[ref_idx].unsqueeze(0)

                # Generate blur.
                if models['blur'] is not None:
                    model_output['model_out'] = torch.sum(kernel_weights * model_output['model_out'], 0, keepdim=True)
                
                # Tone-mapping function: map HDR rgb values to LDR rgb values.
                noise = None
                if models['exp'] is not None:
                    learned_exp = models['exp']().view(-1, 1, 1).repeat(1, video_shape[1],  video_shape[2])
                    if opt.patch_sample:
                        input_exp = learned_exp[:, model_input['coord_idx'][0][0], model_input['coord_idx'][0][1]].view(-1, 1)
                    else:
                        input_exp = learned_exp.view(-1, 1)[model_input['coord_idx'][0]]
                else:
                    input_exp = model_input['exps'].view(-1, 1)

                if models['tm'] is not None:
                    model_output['model_out'] = models['tm'](model_output['model_out'].view(-1, 3), input_exp, noise)

                ######################### Loss Function #######################

                # Reconstruction loss.
                fake = {}; real = {}
                fake['model_out'] = model_output['model_out']
                real['img'] = gt['img']

                if opt.use_random_gamma:
                    random_gamma = (torch.rand(1)*100 + 1.).cuda() # range in [1, 200]
                    fake['model_out'] = fake['model_out'] + utils.gamma_map(model_output['model_out'], random_gamma) * model_input['gamma_weights'].view(-1, 1)
                    real['img'] = real['img'] + utils.gamma_map(gt['img'], random_gamma) * model_input['gamma_weights'].view(-1, 1)
                
                refine_mask = None
                if use_mask:
                    refine_mask = model_input['weights']
                losses = loss_fn(use_mask, refine_mask, fake, real)

                # Flow loss.
                if opt.flow_loss and total_steps < opt.flow_loss_steps:
                    # flow_loss_f = utils.flow_loss(models['uv_f'], model_input['coords'], model_input['flow'], video_shape, uv_f['coords'], model_input['mask'])
                    # losses.update({'flow_loss_f':flow_loss_f * 100000})

                    # flow_loss_b = utils.flow_loss(models['uv_b'], model_input['coords'], model_input['flow'], video_shape, uv_b['coords'])
                    # losses.update({'flow_loss_b':flow_loss_b * flow_weights})

                    # For the aligned image stack, the flow is set to zero
                    flow_loss_b = utils.flow_loss_zeros(models['uv_b'], model_input['coords'], video_shape, uv_b['coords'], use_alpha=False)
                    losses.update({'flow_loss_b':flow_loss_b * flow_weights})

                # Rigidity loss.
                if opt.rigidity_loss and total_steps < opt.rigidity_loss_steps:

                    refine_mask = 1
                    if use_mask:
                        refine_mask = model_input['weights']

                    rigidity_loss = utils.get_rigidity_loss(
                        model_input['coords'], uv_b['coords'][ref_idx], models['uv_b'], video_shape, uv_mapping_scale=opt.uv_mapping_scale, mask=refine_mask)

                    if models['uv_f'] is not None:
                        rigidity_loss += utils.get_rigidity_loss(
                            model_input['coords'], uv_f['coords'][ref_idx], models['uv_f'], video_shape, uv_mapping_scale=opt.uv_mapping_scale, mask=refine_mask)

                    rigidity_loss = rigidity_loss * opt.rigidity_loss_w
                    losses.update({'rigidity_loss':rigidity_loss})


                if models['uv_f'] is not None:
                    # Sparsity loss.
                    if opt.sparsity_loss:
                        radiance_dynamic_not = (1-alpha)*torch.exp(hdr_rgb_f['model_out'])
                        sparsity_loss1 = (torch.norm(radiance_dynamic_not, dim=-1) ** 2).mean() * opt.sparsity_loss_w
                        losses.update({'sparsity_loss1':sparsity_loss1})

                        sparsity_loss2 = torch.mean(-1 / (torch.log(alpha + 1e-24) + torch.log(1-alpha + 1e-24) )) * opt.sparsity_loss
                        losses.update({'sparsity_loss2':sparsity_loss2})

                    # Alpha loss
                    if opt.alpha_loss:
                        # alpha_loss = torch.mean(-1 / (torch.log(alpha + 1e-24) + torch.log(1-alpha + 1e-24) )) * opt.alpha_loss_w
                        alpha_loss = torch.mean(alpha) * opt.alpha_loss_w
                        losses.update({'alpha_loss':alpha_loss})

                    # Mask loss
                    if opt.mask_loss and total_steps < opt.mask_loss_steps:
                        mask_loss = ((alpha-model_input['mask']) ** 2).mean() * opt.mask_loss_w
                        losses.update({'mask_loss':mask_loss})

                # White balance loss and gradient loss.
                if models['tm'] is not None:
                    # White balance loss 
                    zero_loss = torch.mean(torch.abs(models['tm'](torch.zeros([1,3]).cuda(), torch.zeros([1,1]).cuda()) - opt.fixed_value))
                    losses.update({'ue_loss':zero_loss})

                    # Gradient loss
                    rand_radiance = torch.rand(10000, 3, requires_grad=True) * 5 # radiance in [0, 5]
                    rand_exposure = torch.rand(10000, 1, requires_grad=True) * 6 - 3.0 # radiance in [-3, 3]
                    _, grads_tm = models['tm'](rand_radiance.cuda(), rand_exposure.cuda(), output_grads=True)
                    grad_loss = torch.nn.functional.relu(-grads_tm) * 1e6
                    losses.update({'grad_loss':grad_loss})           

                # Total training loss.
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar('loss/'+loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("loss/total_train_loss", train_loss, total_steps)

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    utils.save_model(models, os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(results_dir, models, gt, model_output, writer, total_steps, opt)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        models['atlas_b'].eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = models['atlas_b'](model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        models['atlas_b'].train()
                

                if flow_weights > 0.001:
                    decay_rate = 0.1
                    decay_steps = 10000
                    flow_weights = start_weight * (decay_rate ** (total_steps / decay_steps))

                if total_steps == opt.start_freeze_tm: # freeze tone-mapping model and learned exposure
                    opt_parameters = [{'params':models['atlas_b'].parameters()}]
                    if models['uv_b'] != None:
                        opt_parameters.append({'params':models['uv_b'].parameters()})
                    if models['uv_f'] != None:
                        opt_parameters.append({'params':models['uv_f'].parameters()})
                    if models['atlas_f'] != None:
                        opt_parameters.append({'params':models['atlas_f'].parameters()})
                    if models['alpha'] != None:
                        opt_parameters.append({'params':models['alpha'].parameters()})
                    if models['blur'] != None:
                        opt_parameters.append({'params':models['blur'].parameters(), 'lr':1e-6})

                    optim = torch.optim.Adam(lr=1e-5, params=opt_parameters)
                    use_mask = True

                total_steps += 1
        utils.save_model(models, os.path.join(checkpoints_dir, 'model_final.pth'))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
