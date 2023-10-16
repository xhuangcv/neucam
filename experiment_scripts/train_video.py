import sys
import os
import time
# Enable import from parent package
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import dataio, utils, loss_functions, modules
import training
import torch
from torch.utils.data import DataLoader
from functools import partial
from configs import opt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_path = utils.load_data_path(opt.dataset)
vid_dataset = dataio.Video(video_path, opt.seq_size, load_gt=True, load_mask=opt.mask_loss,  load_flow=opt.flow_loss,
                           load_weights=opt.stage1, load_depth=opt.use_depth, not_load_exps=opt.learn_exp, load_gamma=opt.use_random_gamma)
if opt.patch_sample:
    coord_dataset = dataio.Implicit3DWrapperPatch(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac)
else:
    coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
image_resolution = (opt.seq_size[1], opt.seq_size[2])

# Define model of first layer (background).
model_atlas_b = modules.SingleBVPNet(type=opt.atlas_b_type,
                                     mode=opt.atlas_b_mode,
                                     hidden_features=opt.atlas_b_hidden_dim,
                                     num_hidden_layers=opt.atlas_b_num_layers,
                                     in_features=opt.atlas_b_in_dim,
                                     out_features=opt.atlas_b_out_dim,
                                     sidelength=image_resolution,
                                     num_frequencies=opt.atlas_b_num_frequencies).to(device)

model_uv_b = None
if opt.uv_mapping:
    model_uv_b = modules.SimpleMLP(hidden_dim=opt.uv_b_hidden_dim, 
                                   num_layer=opt.uv_b_num_layers, 
                                   in_dim=opt.uv_b_in_dim,
                                   out_dim=opt.uv_b_out_dim,
                                   use_encoding=opt.uv_b_use_encoding, 
                                   num_frequencies=opt.uv_b_num_frequencies,
                                   skip_layer=opt.uv_b_skip,
                                   nonlinear=opt.uv_b_last_func).to(device)

# Define model of second layer (foreground).
model_atlas_f = None
if opt.split_layer and opt.uv_mapping:
    out_features = 3 if opt.alpha_mlp else 4
    model_atlas_f = modules.SingleBVPNet(type=opt.atlas_f_type,
                                         mode=opt.atlas_f_mode,
                                         hidden_features=opt.atlas_f_hidden_dim,
                                         in_features=opt.atlas_f_in_dim,
                                         out_features=out_features,
                                         sidelength=image_resolution,
                                         num_frequencies=opt.atlas_f_num_frequencies).to(device)

model_uv_f = None
if opt.split_layer:
    model_uv_f = modules.SimpleMLP(hidden_dim=opt.uv_f_hidden_dim, 
                                   num_layer=opt.uv_f_num_layers, 
                                   in_dim=opt.uv_f_in_dim,
                                   out_dim=opt.uv_f_out_dim,
                                   use_encoding=opt.uv_f_use_encoding, 
                                   num_frequencies=opt.uv_f_num_frequencies,
                                   skip_layer=opt.uv_f_skip,
                                   nonlinear=opt.uv_f_last_func).to(device)

# Define alpha model.
model_alpha = None
if opt.alpha_mlp:
    model_alpha = modules.SimpleMLP(hidden_dim=opt.alpha_hidden_dim, 
                                    num_layer=opt.alpha_num_layers,
                                    in_dim=opt.alpha_in_dim,
                                    out_dim=opt.alpha_out_dim, 
                                    use_encoding=opt.alpha_use_encoding, 
                                    num_frequencies=opt.alpha_num_frequencies, 
                                    skip_layer=opt.alpha_skip,
                                    nonlinear=opt.alpha_last_func).to(device)

# Define exposure model.
model_exp = None
if opt.learn_exp:
    model_exp = modules.LearnExps(vid_dataset.shape[0]).to(device)
    # model_exp = modules.LearnFocal([-1., 0., 1., -1., 0.]).to(device)

# Define focus model.
model_focal = None
if opt.learn_focal:
    # model_focal = modules.LearnFocal([-1., -1., 0., 1., 1.], False).to(device)
    model_focal = modules.LearnFocal(opt.focal_list, False).to(device)

# Define exposure model.
model_depth = None
if opt.learn_depth:
    model_depth= modules.LearnDepth((vid_dataset.depth- 0.5) / 0.5).to(device)

# Define noise model.
model_noise = None
if opt.denoise:
    model_noise = modules.SingleBVPNet(type='relu', mode='mlp',  hidden_features=128,
                                       in_features=2, out_features=18, sidelength=image_resolution,
                                       num_frequencies=7).to(device)

# Define deblur model.
model_blur = None
if opt.deblur:
    model_blur = modules.SimpleMLP(hidden_dim=opt.blur_hidden_dim,
                                   num_layer=opt.blur_num_layers,
                                   in_dim=opt.blur_in_dim,
                                   out_dim=opt.patch_size*3,
                                   use_encoding=opt.blur_use_encoding,
                                   num_frequencies=opt.blur_num_frequencies,
                                   skip_layer=opt.blur_skip,
                                   nonlinear=opt.blur_last_func).to(device)

# Define tone-mapping model.
model_tm = None
if opt.pre_hdr:
    model_tm = modules.TonemapNet()
    if opt.load_tm:
        ckp = torch.load(opt.tm_model_path)
        model_tm.load_state_dict(ckp['model_tm_state_dict'])
    model_tm.to(device)

# Pretrain uv model.
if opt.uv_pretrain:
    if model_uv_b is not None:
        model_uv_b = utils.pre_train_mapping(model_mapping=model_uv_b, shape=vid_dataset.shape, 
                                             uv_mapping_scale=opt.uv_mapping_scale, 
                                             pretrain_iters=opt.pretrain_iters, device=device)
    if model_uv_f is not None:
        model_uv_f = utils.pre_train_mapping(model_mapping=model_uv_f, shape=vid_dataset.shape, 
                                             uv_mapping_scale=opt.uv_mapping_scale,
                                             pretrain_iters=opt.pretrain_iters, device=device)

# Pretrain alpha model.
if opt.alpha_pretrain and model_alpha is not None and opt.mask_loss:
    model_alpha = utils.pre_train_alpha(model_alpha=model_alpha, dataset=vid_dataset, device=device)

# Root path for results. 
if opt.experiment_name == 'debug':
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
else:
    root_path = os.path.join(opt.logging_root,
                             time.strftime('%Y%m%d', time.localtime()),
                             opt.experiment_name + time.strftime('_%H%M%S', time.localtime()))

# Define the loss
loss_fn = partial(loss_functions.image_mse)
summary_fn = partial(utils.write_video_summary, vid_dataset)

# All models
models = {
    'atlas_b': model_atlas_b,
    'uv_b': model_uv_b,
    'atlas_f': model_atlas_f,
    'uv_f': model_uv_f,
    'alpha': model_alpha,
    'exp': model_exp,
    'focal': model_focal,
    'noise': model_noise,
    'blur': model_blur,
    'tm': model_tm,
    'depth': model_depth
}

training.train(models=models, train_dataloader=dataloader, model_dir=root_path, loss_fn=loss_fn, 
               summary_fn=summary_fn, video_shape=vid_dataset.shape, opt=opt)
