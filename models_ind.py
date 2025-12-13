import torch
from torch import nn

from nbfnet import NBFNet
from lmke_ind import LMKE
class Model(nn.Module):

    def __init__(self, belman_model_cfg, lm_model): #, embedding_path
        # kept that because super Ultra sounds cool
        super(Model, self).__init__()
        # rel_model_cfg = {'input_dim': 64,'hidden_dims': [64, 64, 64, 64, 64, 64],'message_func': 'distmult', 'aggregate_func': 'sum',
		# 			  'short_cut': True, 'layer_norm': True}
        # self.relation_model = RelNBFNet(**rel_model_cfg)
        # self.node_feature, self.query = self.load_embedding(embedding_path)
        self.belman_model = NBFNet(**belman_model_cfg)
        self.lmke_model = LMKE(lm_model)
        # self.lmke_model = LMKE(lm_model)
        input_dim = belman_model_cfg['input_dim']
        hidden_dims = belman_model_cfg['hidden_dims']
        language_dim = lm_model.config.hidden_size
        # feature_dim = (sum(hidden_dims) if self.belman_model.concat_hidden else hidden_dims[-1]) + input_dim + language_dim * 4 
        feature_dim = (sum(hidden_dims) if self.belman_model.concat_hidden else hidden_dims[-1]) + input_dim + language_dim     
        self.mlp_merge = nn.Sequential()
        mlp = []
        mlp.append(nn.Linear(feature_dim, language_dim))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(language_dim, 1))
        self.mlp_merge = nn.Sequential(*mlp)

        
    def forward(self, graph, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode, training=True):
        with torch.no_grad():
            nbf_feature = self.belman_model.forward_no_score(graph, batch, training)  
        language_feature = self.lmke_model.forward_no_score(graph, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode)
        final_feature = torch.cat([language_feature, nbf_feature ], dim=-1)           
        score = self.mlp_merge(final_feature).squeeze(-1)

        
        return score
    
    def forward_test(self, graph, batch, language_feature, training=False):
        nbf_feature = self.belman_model.forward_no_score(graph, batch, training)  
        # language_feature = self.lmke_model.forward_no_score(graph, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode)
        final_feature = torch.cat([language_feature, nbf_feature], dim=-1)           
        score = self.mlp_merge(final_feature).squeeze(-1)

        return score