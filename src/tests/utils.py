import numpy as np
import torch

from classes.classes import ZeroShotModelConfig
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel


def message_passing(g, model_class=PostgresZeroShotModel):
    # define a message passing model
    fc_out_kwargs = dict(p_dropout=0.0, activation_class_name='LeakyReLU', activation_class_kwargs={},
                         norm_class_name='Identity', norm_class_kwargs={}, residual=False, dropout=True,
                         activation=True, inplace=True)
    final_mlp_kwargs = dict(width_factor=1, n_layers=2)
    tree_layer_kwargs = dict(width_factor=1, n_layers=2, test=True)
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    config = ZeroShotModelConfig(
        device='cpu',
        hidden_dim=6,
        final_mlp_kwargs=final_mlp_kwargs,
        tree_layer_name='MscnConv',
        tree_layer_kwargs=tree_layer_kwargs,
        featurization=None,
    )
    m = model_class(model_config=config, feature_statistics={})
    m.test = True
    # initialize hidden states with one hot encodings
    no_nodes_per_type = []
    hidden_dict = dict()
    for i, ntype in enumerate(g.ntypes):
        hidden = np.zeros((g.number_of_nodes(ntype=ntype), len(g.ntypes)), dtype=np.float32)
        hidden[:, i] = 1
        no_nodes_per_type.append(g.number_of_nodes(ntype=ntype))
        hidden_dict[ntype] = torch.from_numpy(hidden)
    model_out = m.message_passing(g, hidden_dict).numpy()
    return no_nodes_per_type, model_out
