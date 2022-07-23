import torch
import torch.nn as nn
import pdb

def get_activation(params):
    if type(params) == dict:
        name = params['type']
        kwargs = params['kwargs']
    else:
        name = str(params)
        kwargs = {}
    return lambda: getattr(nn, name)(**kwargs)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, activation='GELU', output_activation='Identity', name='mlp'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.name = name
        activation = get_activation(activation)
        output_activation = get_activation(output_activation)

        layers = []
        current = input_dim
        for dim in hidden_dims:
            linear = nn.Linear(current, dim)
            layers.append(linear)
            layers.append(activation())
            current = dim

        layers.append(nn.Linear(current, output_dim))
        layers.append(output_activation())

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)

    @property
    def num_parameters(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([p.numel() for p in parameters])

    def __repr__(self):
        return  '[ {} : {} parameters ] {}'.format(
            self.name, self.num_parameters,
            super().__repr__())

class FlattenMLP(MLP):

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return super().forward(x)

class TimeConditionedMLP(MLP):
    def __init__(self, time_dim, *args, **kwargs):
        ## @TODO obviously streamline this
        from diffusion.denoising_diffusion_pytorch import SinusoidalPosEmb, Mish

        super().__init__(*args, **kwargs)
        self.time_model = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            Mish(),
            nn.Linear(time_dim * 4, time_dim)
        )

        ## get small model to map from time embedding
        ## to input dimensionality of each linear layer
        self.augment_models = nn.ModuleDict({
            str(i): nn.Sequential(
                Mish(),
                nn.Linear(time_dim, layer.weight.shape[1])
            )
            for i, layer in enumerate(self._layers)
            if type(layer) == nn.Linear
        })

    def forward(self, x, t):
        ## [ B x time_dim ]
        t_embed = self.time_model(t).squeeze(dim=1)

        for i, layer in enumerate(self._layers):
            if str(i) in self.augment_models:
                augment_model = self.augment_models[str(i)]
                augment = augment_model(t_embed)
                # print(i, t_embed.shape, augment.shape, x.shape)
                x = x + augment

            x = layer(x)

        return x
