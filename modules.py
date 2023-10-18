from shelve import Shelf
import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output

class IMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=256,
            use_positional=True,
            positional_dim=10,
            skip_layers=[4, 6],
            num_layers=8,  # includes the output layer
            verbose=True,use_tanh=True,apply_softmax=False):
        super(IMLP, self).__init__()
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax= nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)],requires_grad = False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f'Model has {count_parameters(self)} params')

    def forward(self, x):
        x = x.reshape(-1, 3)
        if self.use_positional:
            if self.b.device!=x.device:
                self.b=self.b.to(x.device)
            pos = positionalEncoding_vec(x,self.b)
            x = pos

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x

class TonemapNet(nn.Module):
    '''A network for HDR pixel tone-mapping.'''

    def __init__(self, W=128):
        super().__init__()

        self.layers_r = nn.Sequential(*[
            nn.Linear(1, W),
            nn.ReLU(),
            nn.Linear(W, 1)
        ])
        self.layers_g = nn.Sequential(*[
            nn.Linear(1, W),
            nn.ReLU(),
            nn.Linear(W, 1)
        ])
        self.layers_b = nn.Sequential(*[
            nn.Linear(1, W),
            nn.ReLU(),
            nn.Linear(W, 1)
        ])

    def forward(self, x, input_exps, noise=None, output_grads=False):

        # Split RGB channels 
        if noise == None:
            log_exps = torch.log(torch.Tensor([2])).cuda()*input_exps
            h_r_s = x[:,0:1] + log_exps
            h_g_s = x[:,1:2] + log_exps
            h_b_s = x[:,2:3] + log_exps

            lnx = torch.cat([h_r_s, h_g_s, h_b_s], -1)
        else:
            exps = torch.Tensor([2]).cuda()**input_exps
            x = torch.exp(x)
            h_r_s = torch.clip(x[:,0:1]*exps + noise[:, 0:1], 1e-16)
            h_g_s = torch.clip(x[:,1:2]*exps + noise[:, 1:2], 1e-16)
            h_b_s = torch.clip(x[:,2:3]*exps + noise[:, 2:3], 1e-16)

            lnx = torch.log(torch.cat([h_r_s, h_g_s, h_b_s], -1))

        hdr_r = lnx[:,0:1]
        hdr_g = lnx[:,1:2]
        hdr_b = lnx[:,2:3]

        ldr_r = self.layers_r(hdr_r)
        ldr_g = self.layers_g(hdr_g)
        ldr_b = self.layers_b(hdr_b)

        ldr_out = torch.cat([ldr_r, ldr_g, ldr_b], -1)
        ldr_out = torch.tanh(ldr_out)

        if output_grads:
            grad = torch.autograd.grad(ldr_out.mean(), lnx, retain_graph=True)[0].min()
            return ldr_out, grad
        else:
            return ldr_out


class SimpleMLP(nn.Module):
    '''A simple MLP network'''
    def __init__(self, hidden_dim=256, num_layer=8, in_dim=3, out_dim=2, use_encoding=False, num_frequencies=None, skip_layer=[4], nonlinear='tanh') -> None:
        super().__init__()
        self.use_encoding = use_encoding
        self.skip_layer = skip_layer
        self.last_nonlinear = nonlinear
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        self.in_dim=in_dim

        if self.use_encoding:
            self.positional_encoding = PosEncodingNeRF(in_features=in_dim, num_frequencies=num_frequencies)
            in_dim = self.positional_encoding.out_dim

        input_ch = in_dim
        for i in range(num_layer):
            if i in skip_layer:
                input_ch += in_dim
            if i == num_layer-1:
                output_ch = out_dim
            else:
                output_ch = hidden_dim
            self.layers.append(nn.Linear(input_ch, output_ch))
            input_ch = output_ch
    
    def forward(self, x):
        if self.use_encoding:
            x = self.positional_encoding(x).view(-1, self.positional_encoding.out_dim)
        else:
            x = x.reshape(-1, self.in_dim)

        input = x.clone().detach().requires_grad_(True)

        for i,layer in enumerate(self.layers):
            if i in self.skip_layer:
                x = torch.cat((x, input), -1)
            x = layer(x)
            if i != (self.num_layer-1):
                x = F.relu(x)

        if self.last_nonlinear == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.last_nonlinear == 'softmax':
            x = torch.softmax(x, -1)
        elif self.last_nonlinear == 'tanh':
            x = torch.tanh(x)
        elif self.last_nonlinear == 'both':
            x1 = torch.softmax(x[..., 0:x.shape[-1]//3], -1)
            x2 = torch.tanh(x[..., x.shape[-1]//3:])
            x = torch.cat([x1, x2],-1)
        elif self.last_nonlinear == 'none':
            x = x
        else:
            raise Exception("Unknown activation functions.", self.last_nonlinear)

        return x

class FocalMLP(nn.Module):
    '''A simple MLP network'''
    def __init__(self, hidden_dim=256, num_layer=8, in_dim=3, out_dim=2, use_encoding=False, num_frequencies=None, skip_layer=[4], nonlinear='tanh') -> None:
        super().__init__()
        self.use_encoding = use_encoding
        self.skip_layer = skip_layer
        self.last_nonlinear = nonlinear
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        self.in_dim=in_dim

        if self.use_encoding:
            self.positional_encoding = PosEncodingNeRF(in_features=in_dim, num_frequencies=num_frequencies)
            in_dim = self.positional_encoding.out_dim

        input_ch = in_dim
        for i in range(num_layer):
            if i in skip_layer:
                input_ch += in_dim
            # if i == num_layer-2:
            #     output_ch = 16
            if i == num_layer-1:
                input_ch += 2
                output_ch = out_dim
            else:
                output_ch = hidden_dim
            self.layers.append(nn.Linear(input_ch, output_ch))
            input_ch = output_ch
    
    def forward(self, x, d):
        if self.use_encoding:
            x = self.positional_encoding(x).view(-1, self.positional_encoding.out_dim)
        else:
            x = x.reshape(-1, self.in_dim)
            d = d.reshape(-1, d.shape[-1])

        input = x.clone().detach().requires_grad_(True)

        for i,layer in enumerate(self.layers):
            if i in self.skip_layer:
                x = torch.cat((x, input), -1)
            if i == (self.num_layer - 1):
                x = torch.cat((x, d), -1)
            x = layer(x)
            if i != (self.num_layer-1):
                x = F.relu(x)

        if self.last_nonlinear == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.last_nonlinear == 'softmax':
            x = torch.softmax(x, -1)
        elif self.last_nonlinear == 'tanh':
            x = torch.tanh(x)
        elif self.last_nonlinear == 'both':
            x1 = torch.softmax(x[..., 0:x.shape[-1]//3], -1)
            x2 = torch.tanh(x[..., x.shape[-1]//3:])
            x = torch.cat([x1, x2],-1)
        elif self.last_nonlinear == 'none':
            x = x
        else:
            raise Exception("Unknown activation functions.", self.last_nonlinear)

        return x

class LearnFocal(nn.Module):
    '''Learn the focus of each frame.'''
    def __init__(self, focal_list=[-1., -1., 0., 1., 1.], requires_grad=False):
        super().__init__()
        if isinstance(focal_list, list):
            focus = torch.tensor(focal_list, requires_grad=requires_grad)
        else:
            focus = torch.zeros(focal_list, requires_grad=True)
        self.focus = nn.Parameter(focus)

    def forward(self):
        # return torch.clip(self.focus, -1, 1) # ensure the exposure in [-3, 3]
        return self.focus # ensure the exposure in [-3, 3]

    def get_minmax(self):
        return torch.cat([torch.min(self.focus).unsqueeze(0), torch.max(self.focus).unsqueeze(0)], 0)

    def get_mid(self):
        value, _ = torch.sort(self.focus)
        return value[len(self.focus) // 2]
    

class LearnExps(nn.Module):
    '''Learn the exposure of each frame.'''
    def __init__(self, num=18):
        super().__init__()
        exps = torch.zeros(num, requires_grad=True)
        self.exps = nn.Parameter(exps)

    def forward(self):
        return torch.tanh(self.exps)*3 # ensure the exposure in [-3, 3]

    def get_minmax(self):
        return torch.cat([torch.min(self.exps).unsqueeze(0), torch.max(self.exps).unsqueeze(0)], 0)

    def get_mid(self):
        value, _ = torch.sort(self.exps)
        return value[len(self.exps) // 2]

    

class LearnDepth(nn.Module):
    '''Learn the parameter of noise.'''
    def __init__(self, depth):
        super().__init__()
        depth = torch.tensor(depth, requires_grad=True)
        self.depth = nn.Parameter(depth.view(-1, 1))

    def forward(self, idx):
        return torch.clip(self.depth[idx], -1, 1)
        

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True),
                                                       num_frequencies=kwargs.get('num_frequencies', None))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        if model_input['coords'].shape[-1] == 3:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        else:
            coords_org = model_input['coords']
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


class PINNet(nn.Module):
    '''Architecture used by Raissi et al. 2019.'''

    def __init__(self, out_features=1, type='tanh', in_features=2, mode='mlp'):
        super().__init__()
        self.mode = mode

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=8,
                           hidden_features=20, outermost_linear=True, nonlinearity=type,
                           weight_init=init_weights_trunc_normal)
        print(self)

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. input
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        output = self.net(coords)
        return {'model_in': coords, 'model_out': output}


class ImageDownsampling(nn.Module):
    '''Generate samples in u,v plane according to downsampling blur kernel'''

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        if num_frequencies is not None:
            self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


########################
# Encoder modules
class SetEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features, nonlinearity='relu'):
        super().__init__()

        assert nonlinearity in ['relu', 'sine'], 'Unknown nonlinearity type'

        if nonlinearity == 'relu':
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal
        elif nonlinearity == 'sine':
            nl = Sine()
            weight_init = sine_init

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                         for _ in range(num_hidden_layers)])
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        if ctxt_mask is not None:
            embeddings = embeddings * ctxt_mask
            embedding = embeddings.mean(dim=-2) * (embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2))
            return embedding
        return embeddings.mean(dim=-2)


class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 1)

        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o


class PartialConvImgEncoder(nn.Module):
    '''Adapted from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
    '''
    def __init__(self, channel, image_resolution):
        super().__init__()

        self.conv1 = PartialConv2d(channel, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(256, 256)
        self.layer2 = BasicBlock(256, 256)
        self.layer3 = BasicBlock(256, 256)
        self.layer4 = BasicBlock(256, 256)

        self.image_resolution = image_resolution
        self.channel = channel

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, I):
        M_c = I.clone().detach()
        M_c = M_c > 0.
        M_c = M_c[:,0,...]
        M_c = M_c.unsqueeze(1)
        M_c = M_c.float()

        x = self.conv1(I, M_c)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        o = self.fc(x.view(x.shape[0], 256, -1)).squeeze(-1)

        return o


class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output


def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out
