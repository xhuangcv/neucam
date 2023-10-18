import configargparse


def config_parser():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--logging_root', type=str, default='./results', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--batch_size', type=int, default=1, help='Number of pixels sampled during each iteration')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate. default=1e-4')
    p.add_argument('--num_epochs', type=int, default=3000,
                help='Number of epochs for training')

    p.add_argument('--epochs_til_ckpt', type=int, default=1000,
                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--train_steps', type=int, default=100000,
                help='Total steps to train.')
    p.add_argument('--steps_til_summary', type=int, default=1000,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--dataset', type=str, default='dog',
                help='Dataset name')
    p.add_argument('--model_type', type=str, default='nerf',
                help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                        '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
    p.add_argument('--sample_frac', type=float, default=1e-5,
                help='What fraction of video pixels to sample in each batch (default is all)')
    p.add_argument('--lr_decay', type=int, default=50,
                help='exponential learning rate decay (in 1000 steps)')
    p.add_argument('--max_frame', type=int, default=1000,
                help='maximum number of frame to train.')
    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    p.add_argument('--pre_hdr', action="store_true", help='Reconstruct HDR image.')
    p.add_argument('--deblur', action="store_true", help='Reconstruct all-in-focus image.')
    p.add_argument('--denoise', action="store_true", help='Learn the noise of the image.')
    p.add_argument('--uv_mapping', action="store_true", help='Map (u,v,t) to (u,v).')
    p.add_argument('--split_layer',  action="store_true", help='Split to static layer and dynamic layer.')
    p.add_argument('--learn_exp', action="store_true", help='Learn the exposure of each frame.')
    p.add_argument('--learn_focal', action="store_true", help='Learn the focal length of each frame.')
    p.add_argument('--learn_depth', action="store_true", help='Learn the depth of each frame. (Not implemented)')
    p.add_argument('--use_depth', action="store_true", help='Use the depth of each frame.')
    p.add_argument('--alpha_mlp', action="store_true", help='Learn the alpha blending between background and foreground using an mlp.')
    p.add_argument('--seq_size', type=int, action='append', help='[length, height, width] of training video.')
    p.add_argument('--focal_list', type=float, action='append', help='Focal length of each frame, we can easily set to [-1, 0, 1].')
    p.add_argument('--patch_sample', action="store_true", help='Use patch-based sampling.')

    p.add_argument('--use_random_gamma', action="store_true", help='Apply gamma correction to both the predicted image and the ground truth image to enhance performance.')
    p.add_argument('--alpha_pretrain', action="store_true", help='Warm up of alpha model.')
    p.add_argument('--uv_pretrain', action="store_true", help='Warm up of uv mapping model.')
    p.add_argument('--tm_pretrain', action="store_true", help='Warm up of tone mapping model.')
    p.add_argument('--uv_mapping_scale', default=1.0, type=float, help='UV mapping scale for [-1, 1], e.g. 0.8 means UV in [-0.8, 0.8].')

    # Weights of loss
    p.add_argument('--rigidity_loss', action="store_true", help='Ensure rigid mapping.')
    p.add_argument('--sparsity_loss', action="store_true", help='Ensure the alpha value is 0 or 1')
    p.add_argument('--alpha_loss', action="store_true", help='Ensure the foreground is small.')
    p.add_argument('--mask_loss', action="store_true", help='Use input mask to split foreground and background.')
    p.add_argument('--flow_loss', action="store_true", help='Use input flow to constraint the UV mapping.')

    p.add_argument('--smooth_sigma', type=float, default=2,
                help='Sigma of bilateralfilter.')
    p.add_argument('--rigidity_loss_w', type=float, default=None,
                help='Weight of rigidity loss.')
    p.add_argument('--sparsity_loss_w', type=float, default=None,
                help='Weight of sparsity loss.')
    p.add_argument('--alpha_loss_w', type=float, default=None,
                help='Weight of alpha loss.')
    p.add_argument('--mask_loss_w', type=float, default=None,
                help='Weight of mask loss.')
                
    p.add_argument('--mask_loss_steps', type=int, default=5000,
                help='Total steps for mask loss.')
    p.add_argument('--flow_loss_steps', type=int, default=5000,
                help='Total steps for flow loss.')
    p.add_argument('--rigidity_loss_steps', type=int, default=5000,
                help='Total steps for rigidity loss.')
    
    p.add_argument('--start_freeze_tm', type=int, default=-1,
                help='Start to freeze the tone mapping model.')
    
    p.add_argument('--offset_size', type=int, default=5,
                help='Maximum offset to control the receptive field of sampling patch.')
    p.add_argument('--fixed_value', type=float, default=0,
                help='Fixed value for white balance loss.')
    p.add_argument('--patch_size', type=int, default=9,
                help='points at each sampling patch (blur kernel). e.g. 3x3, 4x4, 5x5.')


    # Setting of model_atlas_b
    p.add_argument('--atlas_b_in_dim', type=int, default=2, help='Dimension of the input feature of atlas_b.')
    p.add_argument('--atlas_b_out_dim', type=int, default=3, help='Dimension of the output feature of atlas_b.')
    p.add_argument('--atlas_b_hidden_dim', type=int, default=512, help='Dimension of the hidden layer of atlas_b.')
    p.add_argument('--atlas_b_num_layers', type=int, default=3, help='Number of hidden layer of atlas_b.')
    p.add_argument('--atlas_b_type', type=str, default='sine', help='Nonlinear function.')
    p.add_argument('--atlas_b_mode', type=str, default='mlp', help='[mlp, nerf, rbf]')
    p.add_argument('--atlas_b_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # Setting of model_atlas_f
    p.add_argument('--atlas_f_in_dim', type=int, default=2, help='Dimension of the input feature of atlas_f.')
    p.add_argument('--atlas_f_out_dim', type=int, default=3, help='Dimension of the output feature of atlas_f.')
    p.add_argument('--atlas_f_hidden_dim', type=int, default=512, help='Dimension of the in feature of atlas_f.')
    p.add_argument('--atlas_f_num_layers', type=int, default=3, help='Number of hidden layer of atlas_f.')
    p.add_argument('--atlas_f_type', type=str, default='sine', help='Nonlinear function.')
    p.add_argument('--atlas_f_mode', type=str, default='mlp', help='[mlp, nerf, rbf]')
    p.add_argument('--atlas_f_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # Setting of model_uv_b
    p.add_argument('--uv_b_hidden_dim', type=int, default=256, help='Dimension of hidden layers.')
    p.add_argument('--uv_b_num_layers', type=int, default=4, help='Number of all layers.')
    p.add_argument('--uv_b_in_dim', type=int, default=3, help='Dimension of the input feature.')
    p.add_argument('--uv_b_out_dim', type=int, default=2, help='Dimension of the output feature.')
    p.add_argument('--uv_b_skip', type=int, action='append', help='Layers for skip connection.')
    p.add_argument('--uv_b_last_func', type=str, default='tanh', help='The nonlinear function for final output tensor.')
    p.add_argument('--uv_b_use_encoding', default=False, action="store_true", help='Use positional enconding.')
    p.add_argument('--uv_b_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # Setting of model_uv_f
    p.add_argument('--uv_f_hidden_dim', type=int, default=256, help='Dimension of hidden layers.')
    p.add_argument('--uv_f_num_layers', type=int, default=4, help='Number of all layers.')
    p.add_argument('--uv_f_in_dim', type=int, default=3, help='Dimension of the input feature.')
    p.add_argument('--uv_f_out_dim', type=int, default=2, help='Dimension of the output feature.')
    p.add_argument('--uv_f_skip', type=int, action='append', help='Layers for skip connection.')
    p.add_argument('--uv_f_last_func', type=str, default='tanh', help='The nonlinear function for final output tensor.')
    p.add_argument('--uv_f_use_encoding', default=False, action="store_true", help='Use positional enconding.')
    p.add_argument('--uv_f_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # Setting of model_alpha
    p.add_argument('--alpha_hidden_dim', type=int, default=256, help='Dimension of hidden layers.')
    p.add_argument('--alpha_num_layers', type=int, default=4, help='Number of all layers.')
    p.add_argument('--alpha_in_dim', type=int, default=3, help='Dimension of the input feature.')
    p.add_argument('--alpha_out_dim', type=int, default=2, help='Dimension of the output feature.')
    p.add_argument('--alpha_skip', type=int, action='append', help='Layers for skip connection.')
    p.add_argument('--alpha_last_func', type=str, default='tanh', help='The nonlinear function for final output tensor.')
    p.add_argument('--alpha_use_encoding', default=False, action="store_true", help='Use positional enconding.')
    p.add_argument('--alpha_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # Setting of model_blur
    p.add_argument('--blur_hidden_dim', type=int, default=256, help='Dimension of hidden layers.')
    p.add_argument('--blur_num_layers', type=int, default=4, help='Number of all layers.')
    p.add_argument('--blur_in_dim', type=int, default=3, help='Dimension of the input feature.')
    p.add_argument('--blur_out_dim', type=int, default=2, help='Dimension of the output feature.')
    p.add_argument('--blur_skip', type=int, action='append', help='Layers for skip connection.')
    p.add_argument('--blur_last_func', type=str, default='tanh', help='The nonlinear function for final output tensor.')
    p.add_argument('--blur_use_encoding', default=False, action="store_true", help='Use positional enconding.')
    p.add_argument('--blur_num_frequencies', type=int, default=7, help='Frequencies of positional encoding.')

    # For testing
    p.add_argument('--save_frame', default=False, action="store_true", help='Save the predicted images.')
    p.add_argument('--save_atlas', default=False, action="store_true", help='Save the atlas in atlas model.')
    p.add_argument('--save_gt', default=False, action="store_true", help='Save the GT (training) images.')
    p.add_argument('--edited_video', default=False, action="store_true", help='Dimension of the in feature of model_f.')
    p.add_argument('--render_focal', default=False, action="store_true", help='Render images with blur.')
    p.add_argument('--evaluate_hdr', default=False, action="store_true", help='If GT HDR images are available, we can evaluate the rendered HDR images.')
    p.add_argument('--atlas_name', type=str, default=None, help='Name of saved atlas.')
    p.add_argument("--chunk", type=int, default=65535, help='Number of rays processed in parallel, decrease if running out of memory')
    return p

parser = config_parser()
opt = parser.parse_args()