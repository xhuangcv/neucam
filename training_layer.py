'''Implements a generic training loop.
'''

from ast import Not
import torch
import utils
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, model_tm, model_mapping_s, model_mapping_d, model_alpha, model_exp, train_dataloader, 
          epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn, summary_fn, val_dataloader=None,
          double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, video_shape=None):

    opt_parameters = [{'params':model.parameters()}]
    if model_tm != None:
        opt_parameters.append({'params':model_tm.parameters(), 'lr':1e-5}) # pretrain model with a small learning rate
    if model_mapping_s != None:
        opt_parameters.append({'params':model_mapping_s.parameters()})
    if model_mapping_d != None:
        opt_parameters.append({'params':model_mapping_d.parameters()})
    if model_alpha != None:
        opt_parameters.append({'params':model_alpha.parameters()})
    if model_exp != None:
        opt_parameters.append({'params':model_exp.parameters()})

    optim = torch.optim.Adam(lr=lr, params=opt_parameters)

    if os.path.exists(model_dir):
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                save_parameters = {
                        'model_state_dict':model.state_dict()
                    }
                if model_tm != None:
                    save_parameters.update({'model_tm_state_dict':model_tm.state_dict()})
                if model_mapping_s != None:
                    save_parameters.update({'model_mapping_s_state_dict':model_mapping_s.state_dict()})
                if model_mapping_d != None:
                    save_parameters.update({'model_mapping_d_state_dict':model_mapping_d.state_dict()})
                if model_alpha != None:
                    save_parameters.update({'model_alpha_state_dict':model_alpha.state_dict()})
                if model_exp != None:
                    save_parameters.update({'model_exp_state_dict':model_exp.state_dict()})
                torch.save(save_parameters,
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                #            np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                # Map (u,v,t) to (u,v).
                model_input_s = {}
                model_input_d = {}
                if model_mapping_s != None:
                    uv_source = model_input['coords'][..., 1:]
                    model_input_s['coords']  = model_mapping_s(model_input['coords']).view(1, -1, 2) + uv_source
                    if model_mapping_d != None:
                        model_input_s['coords'] = model_input_s['coords']*0.25 - 0.5 # uv of static in [-1,0]
                        model_input_d['coords'] = model_mapping_d(model_input['coords']).view(1, -1, 2) + uv_source
                        model_input_d['coords'] = model_input_d['coords']*0.25 + 0.5 # uv of dynamic in [0,1]

                # Map (u,v) or (u,v,t) to HDR rgb values.
                model_output_s = model(model_input_s)
                if model_mapping_d != None:
                    model_output_d = model(model_input_d)

                # # Blend the static scene and dynamic scene.
                # model_output = model_output_s
                # if model_alpha != None:
                #     alpha = model_alpha(model_input['coords']).view(1, -1, 1)
                #     model_output['model_out'] = (1-alpha)*model_output_s['model_out'] + alpha*model_output_d['model_out']

                # Tone-mapping function: map HDR rgb values to LDR rgb values.
                if model_tm != None:
                    if model_exp != None:
                        input_exp = model_exp().view(-1, 1, 1).repeat(1, video_shape[1],  video_shape[2])
                        input_exp = input_exp.view(-1, 1)[model_input['coord_idx'][0]]
                    else:
                        input_exp = model_input['exps'].view(-1, 1)

                    # model_output['model_out'] = model_tm(model_output['model_out'].view(-1, 3), input_exp)
                    # model_output['model_out'] = model_output['model_out'].view(-1, 3)

                    model_output_s['model_out'] = model_tm(model_output_s['model_out'].view(-1, 3), input_exp)
                    if model_mapping_d != None:
                        model_output_d['model_out'] = model_tm(model_output_d['model_out'].view(-1, 3), input_exp)
                
                # Blend the static scene and dynamic scene.
                model_output = model_output_s
                if model_alpha != None:
                    alpha = model_alpha(model_input['coords']).view(1, -1, 1)
                    model_output['model_out'] = (1-alpha)*model_output_s['model_out'] + alpha*model_output_d['model_out']
                model_output['model_out'] = model_output['model_out'].view(-1, 3)


                # Reconstruction loss.
                losses = loss_fn(model_output, gt)

                # Sparsity loss.
                # radiance_dynamic_not = (1-alpha)*torch.exp(model_output_d['model_out'])
                # sparsity_loss = (torch.norm(radiance_dynamic_not, dim=1) ** 2).mean() * 1e-5
                # losses.update({'sparsity_loss':sparsity_loss})

                # Alpha loss
                alpha_loss = torch.mean(-1 / (torch.log(alpha + 1e-16) + torch.log(1-alpha + 1e-16) )) * 5e-4
                losses.update({'alpha_loss':alpha_loss})

                # Unit exposure loss and gradient loss.
                if model_tm != None:
                    # Unit exposure loss
                    zero_loss = torch.mean(torch.abs(model_tm(torch.zeros([1,3]).cuda(), torch.zeros([1,1]).cuda())))
                    losses.update({'zero_loss':zero_loss})
                    # Gradient loss
                    rand_radiance = torch.rand(10000, 3, requires_grad=True) * 5 # radiance in [0, 5]
                    rand_exposure = torch.rand(10000, 1, requires_grad=True) * 6 - 3.0 # radiance in [-3, 3]
                    _, grads_tm = model_tm(rand_radiance.cuda(), rand_exposure.cuda(), output_grads=True)
                    grad_loss = torch.nn.functional.relu(-grads_tm) * 1e6
                    losses.update({'grad_loss':grad_loss})

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    save_parameters = {
                        'model_state_dict':model.state_dict()
                    }
                    if model_tm != None:
                        save_parameters.update({'model_tm_state_dict':model_tm.state_dict()})
                    if model_mapping_s != None:
                        save_parameters.update({'model_mapping_s_state_dict':model_mapping_s.state_dict()})
                    if model_mapping_d != None:
                        save_parameters.update({'model_mapping_d_state_dict':model_mapping_d.state_dict()})
                    if model_alpha != None:
                        save_parameters.update({'model_alpha_state_dict':model_alpha.state_dict()})
                    if model_exp != None:
                        save_parameters.update({'model_exp_state_dict':model_exp.state_dict()})
                    torch.save(save_parameters,
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_tm, model_mapping_s, model_mapping_d, model_alpha, model_exp, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1
        save_parameters = {
            'model_state_dict':model.state_dict()
        }
        if model_tm != None:
            save_parameters.update({'model_tm_state_dict':model_tm.state_dict()})
        if model_mapping_s != None:
            save_parameters.update({'model_mapping_s_state_dict':model_mapping_s.state_dict()})
        if model_mapping_d != None:
            save_parameters.update({'model_mapping_d_state_dict':model_mapping_d.state_dict()})
        if model_alpha != None:
            save_parameters.update({'model_alpha_state_dict':model_alpha.state_dict()})
        if model_exp != None:
            save_parameters.update({'model_exp_state_dict':model_exp.state_dict()})
        torch.save(save_parameters,
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
