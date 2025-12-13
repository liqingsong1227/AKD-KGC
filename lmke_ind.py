import math
import os
#import pdb
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import pickle
import random
import copy
import pdb
import tasks

num_deg_features = 2
class LMKE(nn.Module):
	def __init__(self, lm_model):
		super().__init__()

		self.lm_model_given = lm_model
		self.lm_model_target = copy.deepcopy(lm_model)

		self.hidden_size = lm_model.config.hidden_size

		self.mask_embeddings = torch.nn.Embedding(3, self.hidden_size)
		self.sim_classifier = nn.Sequential(nn.Linear(self.hidden_size , self.hidden_size),
		                              nn.ReLU(),
		                              nn.Linear(self.hidden_size, 1))
		
		self.hrt_merge = nn.Sequential(nn.Linear(self.hidden_size *2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size))
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, graph, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode):
		h_index, t_index, r_index = batch.unbind(-1)
		h_index, t_index, r_index = tasks.negative_sample_to_tail(graph, h_index, t_index, r_index)  # batch * (1+neg)

		hr_preds = self.forward_hr(hr_inputs, hr_positions, mode).unsqueeze(1).expand(-1, h_index.shape[1], -1)
		t_preds = self.encode_target(t_inputs, t_positions, mode).unsqueeze(0).expand(h_index.shape[0], -1 ,-1)

		total_feature = self.hrt_merge(torch.cat([hr_preds, t_preds], dim=-1))
		# total_feature = torch.cat([hr_preds, t_preds, hr_preds-t_preds, hr_preds*t_preds], dim=-1)  
		preds = self.sim_classifier(total_feature).squeeze(-1)

		return preds
	
	def forward_no_score(self, graph, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode):
		h_index, t_index, r_index = batch.unbind(-1)

		h_index, t_index, r_index = tasks.negative_sample_to_tail(graph, h_index, t_index, r_index)  # batch * (1+neg)
		hr_preds = self.forward_hr(hr_inputs, hr_positions, mode).unsqueeze(1).expand(-1, h_index.shape[1], -1)
		t_preds = self.encode_target(t_inputs, t_positions, mode).unsqueeze(0).expand(h_index.shape[0],-1,-1)


		merge_feature = self.hrt_merge(torch.cat([hr_preds, t_preds], dim=-1))
		total_feature = merge_feature
		# total_feature = torch.cat([hr_preds, t_preds, hr_preds-t_preds, hr_preds*t_preds], dim=-1) 
		
		return total_feature
	
	def forward_test(self, graph, batch, hr_inputs, hr_positions, t_preds, mode):
		h_index, t_index, r_index = batch.unbind(-1)
		h_index, t_index, r_index = tasks.negative_sample_to_tail(graph, h_index, t_index, r_index)  # batch * (1+neg)

		hr_preds = self.forward_hr(hr_inputs, hr_positions, mode).unsqueeze(1).expand(-1, h_index.shape[1], -1)
		t_preds = t_preds.unsqueeze(0).expand(h_index.shape[0], -1, -1)
	
		# print(hr_preds.shape, t_preds.shape, deg_feature.shape)
		merge_feature = self.hrt_merge(torch.cat([hr_preds, t_preds], dim=-1))
		total_feature = merge_feature
		# total_feature = torch.cat([hr_preds, t_preds, hr_preds-t_preds, hr_preds*t_preds], dim=-1)  
		return total_feature


	def forward_hr(self, inputs, positions, mode):

		batch_size = len(positions)


		lm_model = self.lm_model_given

		device = lm_model.device
		h_idx = [positions[i]['head'][0] for i in range(batch_size)]
		# print(h_idx)
		h_idx = torch.LongTensor(h_idx)
		h_idx = h_idx.to(device)
		# print(h_idx.device, device)
		h_pos = torch.LongTensor([positions[i]['head'][1] for i in range(batch_size)]).to(device)
		r_idx = torch.LongTensor([positions[i]['rel'][0]  for i in range(batch_size)]).to(device)
		r_pos = torch.LongTensor([positions[i]['rel'][1]  for i in range(batch_size)]).to(device)
		t_idx = torch.LongTensor([positions[i]['tail'][0] for i in range(batch_size)]).to(device)
		t_pos = torch.LongTensor([positions[i]['tail'][1] for i in range(batch_size)]).to(device)


		input_ids = inputs.pop('input_ids')
		input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)
		

		if mode == 'link_prediction_h':
			mask_emb = self.mask_embeddings(torch.LongTensor([0]).cuda())
		elif mode == 'link_prediction_r':
			mask_emb = self.mask_embeddings(torch.LongTensor([1]).cuda())
		elif mode == 'link_prediction_t':
			mask_emb = self.mask_embeddings(torch.LongTensor([2]).cuda())

		for i in range(batch_size):
			if mode != 'link_prediction_h':
				pass
			else:
				input_embeds[i, h_pos[i], :] = mask_emb
			if mode != 'link_prediction_r':
				pass
			else:
				input_embeds[i, r_pos[i], :] = mask_emb
			if mode != 'link_prediction_t':
				pass
			else:
				input_embeds[i, t_pos[i], :] = mask_emb

		inputs['inputs_embeds'] = input_embeds

		logits = lm_model(**inputs) 
		

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []
		#pdb.set_trace()
		'''try:
			triple_embs = logits[1]
		except:
			triple_embs = logits[0][:, 0, :]'''

		for i in range(batch_size):
			h_emb_list.append(logits[0][i, h_pos[i], :].unsqueeze(0))
			r_emb_list.append(logits[0][i, r_pos[i], :].unsqueeze(0))
			t_emb_list.append(logits[0][i, t_pos[i], :].unsqueeze(0))

		h_embs = torch.cat(h_emb_list, dim=0)
		r_embs = torch.cat(r_emb_list, dim=0)
		t_embs = torch.cat(t_emb_list, dim=0)


		# Triple classification 

		if mode == 'link_prediction_h':
			return h_embs
		elif mode == 'link_prediction_r':
			return r_embs
		elif mode == 'link_prediction_t':
			return t_embs

	def encode_target(self, inputs, positions, mode):

		batch_size = len(positions)
		device = self.lm_model_target.device
		
		target_idx = torch.LongTensor([positions[i][0] for i in range(batch_size)]).to(device)
		target_pos = torch.LongTensor([positions[i][1] for i in range(batch_size)]).to(device)


		input_ids = inputs.pop('input_ids')
		input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)

		'''for i in range(batch_size):
			if mode != 'link_prediction_r':
				input_embeds[i, target_pos[i], :] = self.ent_embeddings(target_idx[i])
			else:
				input_embeds[i, target_pos[i], :] = self.rel_embeddings(target_idx[i])'''

		inputs['inputs_embeds'] = input_embeds

		logits = self.lm_model_target(**inputs) 

		target_embs = logits[0][:, 1, :]

		return target_embs