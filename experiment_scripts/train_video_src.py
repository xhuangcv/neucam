'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import skvideo.datasets

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='/apdcephfs/private_xanderhuang/results/logs_hdr', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, default='dog',
               help='Video dataset; one of (cat, bikes)', choices=['cat', 'bikes'])
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=38e-3,
               help='What fraction of video pixels to sample in each batch (default is all)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--pre_hdr',type=bool, default=True, help='Whether to reconstruct HDR video.')
opt = p.parse_args()

if opt.dataset == 'cat':
    video_path = './data/video_512.npy'
elif opt.dataset == 'bikes':
    video_path = skvideo.datasets.bikes()
elif opt.dataset == 'dog':
    video_path = '/apdcephfs/private_xanderhuang/hdr_video_data/Dog-3Exp-2Stop/'

vid_dataset = dataio.Video(video_path)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=3,
                                 mode='nerf', hidden_features=512, num_hidden_layers=3)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=3, mode=opt.model_type)
else:
    raise NotImplementedError

model.cuda()

# Define tone-mapping model.
if opt.pre_hdr:
    model_tm = modules.TonemapNet()
    model_tm.cuda()
else:
    model_tm = None

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_video_summary, vid_dataset)

training.train(model=model, model_tm=model_tm, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
