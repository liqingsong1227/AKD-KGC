import os
#import pdb
import random
import math
import pickle
import torch
import time
from tqdm import tqdm
import copy
from transformers import BatchEncoding
from torchdrug import data, core
from torch.utils import data as torch_data  # dataset
from torch_geometric.data import Data  # graph
from tasks import build_relation_graph
count_sampler = 0

class DataProcess(object):
	"""
	Class for reading triplet from origin file, bulid Graph, and read description
	"""
	def __init__(self, in_paths, tokenizer, lm_model, max_desc_length = 512, add_tokens=False, p_tuning=False, model='bert', description_sentence_num=1, use_description=True, device='cuda:0'):
		self.datasetName = in_paths['dataset']
		# read data from file and build Graph
		self.use_description = use_description
		self.lm_model = lm_model
		self.gpu_device = device
		self.train_triplets = self.load_triplet(in_paths['train'])
		if self.datasetName not in ['fb13']:
			self.valid_triplets = self.load_triplet(in_paths['valid'])
			self.test_triplets = self.load_triplet(in_paths['test'])
			self.valid_triplets_with_neg = None
			self.test_triplets_with_neg = None
		else:
			self.valid_triplets, self.valid_triplets_with_neg = self.load_triplet_with_neg(in_paths['valid'])
			self.test_triplets, self.test_triplets_with_neg = self.load_triplet_with_neg(in_paths['test'])

		print(len(self.train_triplets), len(self.valid_triplets), len(self.test_triplets))
		self.whole_triplets = self.train_triplets + self.valid_triplets + self.test_triplets
		self.train_entity_set = set([t[0] for t in self.train_triplets] + [t[1] for t in self.train_triplets])
		self.entity_set = set([t[0] for t in self.whole_triplets] + [t[1] for t in self.whole_triplets])
		self.train_relation_set = set([t[-1] for t in self.train_triplets])
		self.relation_set = set([t[-1] for t in self.whole_triplets])

		self.buld_id_map(inpaths=[in_paths['entitydict'], in_paths['relationdict']])
		print(len(self.ent2id), len(self.rel2id))

		self.relation_num = int(len(self.rel2id))
		self.entity_num = int(len(self.ent2id))
		if self.valid_triplets_with_neg != None:
			self.valid_idx_triplets_with_neg = [((self.ent2id[sample[0][0]], self.ent2id[sample[0][1]], self.rel2id[sample[0][2]]), sample[1]) for sample in self.valid_triplets_with_neg]
			self.test_idx_triplets_with_neg = [((self.ent2id[sample[0][0]], self.ent2id[sample[0][1]], self.rel2id[sample[0][2]]), sample[1]) for sample in self.test_triplets_with_neg]
		else:
			self.valid_idx_triplets_with_neg = None
			self.test_idx_triplets_with_neg = None

		self.create_graph()

		# read description get tokens and bulid map from ent/rel idx to its description/tokens
		self.uid2description = {}
		self.uid2description_tokens = {}
		self.uid2name = {}
		self.uid2name_tokens = {}
		self.tokenizer = tokenizer
		for p in in_paths['name']:
			self.load_name(p)
		self.idx2name()
		if self.use_description:
			self.load_description(in_paths['description'], description_sentence_num=description_sentence_num)
			self.idx2description()

		self.max_desc_length = max_desc_length
		self.groundtruth, self.possible_entities= self.count_groundtruth()   #(h,r):[t1, t2,...]

		self.add_tokens = add_tokens
		self.p_tuning = p_tuning
		self.model = model
		self.orig_vocab_size = len(tokenizer)
		self.count_degrees()
		# self.entity_embedding, self.relation_embedding = self.caculate_relation_embedding()
		# self.relation_embedding = self.caculate_relation_embedding()
		print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
		print('----------data processing ended----------')

	
	def load_triplet(self, in_path):
		'''
		load triplets without labels by dataset path
		return list made up of triplets (h,t,r)
		'''
		triplet_lists = []
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				if in_path[-3:] == 'txt':
					h, t, r = line.strip('\n').split('\t')
				else:
					h, r, t = line.strip('\n').split('\t')
				triplet_lists.append((h, t, r))
		return triplet_lists
	
	def buld_id_map(self, inpaths):
		entity_path = inpaths[0]
		relation_path = inpaths[1]
		if entity_path != None:
			with open(entity_path, 'r') as f:
				lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
				ent2id = {key: int(value) for value, key in lines}
				id2ent = {int(value): key for value, key in lines}
		else:
			ent2id = {name: id for id, name in enumerate(list(self.entity_set))}
			id2ent = {id: name for id, name in enumerate(list(self.entity_set))}
		if relation_path != None:
			with open(relation_path, 'r') as f:
				lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
				rel2id = {key: int(value) for value, key in lines}
				id2rel = {int(value): key for value, key in lines}
		else:
			rel2id = {name: id for id, name in enumerate(list(self.relation_set))}
			id2rel = {id: name for id, name in enumerate(list(self.relation_set))}
		
		self.ent2id=ent2id
		self.id2ent=id2ent
		self.rel2id=rel2id
		self.id2rel=id2rel
		# print(rel2id, id2rel)
	
	def load_triplet_with_neg(self, in_path):
		'''
		load triplets with labels by dataset path
		return list made up of triplets (h,t,r,l)
		'''
		triplet_lsits = []
		triplet_lsits_with_neg = []
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				h, r, t, l = line.strip('\n').split('\t')
				if l == '-1':
					l = 0
				else:
					l = 1
				triplet_lsits.append((h, t, r))
				triplet_lsits_with_neg.append(((h, t, r), l))
		return triplet_lsits, triplet_lsits_with_neg
	
	def create_graph(self):
		ent2id = copy.deepcopy(self.ent2id)
		rel2id = copy.deepcopy(self.rel2id)
		idx_triplets = []

		for triplet in self.whole_triplets:
			h, t, r = triplet[0], triplet[1], triplet[2]
			h_idx = ent2id[h]
			r_idx = rel2id[r]
			t_idx = ent2id[t]
			idx_triplets.append((h_idx, t_idx, r_idx))
		num_node = len(ent2id)
		num_relation = len(rel2id)
		self.idx_triplets = idx_triplets
		# self.graph = data.Graph(idx_triplets, num_node=num_node, num_relation=num_relation)
		self.train_idx_triplet = idx_triplets[ :len(self.train_triplets)]
		self.valid_idx_triplet = idx_triplets[len(self.train_triplets) : len(self.train_triplets) + len(self.valid_triplets)]
		self.test_idx_triplet = idx_triplets[len(self.train_triplets) + len(self.valid_triplets) : ]
		# self.train_graph = data.Graph(self.train_idx_triplet, num_node=num_node, num_relation=num_relation)
		train_target_edges = torch.tensor([[t[0], t[1]] for t in self.train_idx_triplet], dtype=torch.long).t()
		train_target_etypes = torch.tensor([t[2] for t in self.train_idx_triplet])

		valid_edges = torch.tensor([[t[0], t[1]] for t in self.valid_idx_triplet], dtype=torch.long).t()
		valid_etypes = torch.tensor([t[2] for t in self.valid_idx_triplet])

		test_edges = torch.tensor([[t[0], t[1]] for t in self.test_idx_triplet], dtype=torch.long).t()
		test_etypes = torch.tensor([t[2] for t in self.test_idx_triplet])

		train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
		train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relation])

		self.train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
						  target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relation*2)
		self.valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
						  target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relation*2)
		self.test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
						 target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relation*2)
		# self.train_data = build_relation_graph(self.train_data)
		# self.valid_data = build_relation_graph(self.valid_data)
		# self.test_data = build_relation_graph(self.test_data)

	def split(self):
		offset = 0
		splits = []
		num_samples = [len(self.train_triplets), len(self.valid_triplets), len(self.test_triplets)]
		for num_sample in num_samples:
			split = torch_data.Subset(self, range(offset, offset + num_sample))
			splits.append(split)
			offset += num_sample
		return splits
	
	def load_name(self, in_path):

		uid2name = self.uid2name
		uid2name_tokens = self.uid2name_tokens
		tokenizer = self.tokenizer

		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				uid, name = line.strip('\n').split('\t',1)
				name = name.replace('@en', '').strip('"')
				if uid not in uid2name.keys():
					uid2name[uid] = name
				tokens = tokenizer.tokenize(name)
				if uid not in uid2name_tokens.keys():
					uid2name_tokens[uid] = tokens
		self.uid2name = uid2name
		self.uid2name_tokens = uid2name_tokens

	def idx2name(self):
		'''
		convert entity idx and relation idx into their corresponding name and tokens
		'''
		uid2name = self.uid2name
		uid2name_tokens = self.uid2name_tokens
		id2ent = self.id2ent
		id2rel = self.id2rel
		ent_idx2name = dict()
		ent_idx2name_tokens = dict()
		rel_idx2name = dict()
		rel_idx2name_tokens = dict()
		for idx in list(id2ent.keys()):
			ent_idx2name[idx] = uid2name[id2ent[idx]]
			ent_idx2name_tokens[idx] = uid2name_tokens[id2ent[idx]]
		for idx in list(id2rel.keys()):
			rel_idx2name[idx] = uid2name[id2rel[idx]]
			rel_idx2name_tokens[idx] = uid2name_tokens[id2rel[idx]]
			reverse_name = 'reverse of ' + uid2name[id2rel[idx]]
			reverse_tokens = self.tokenizer.tokenize(reverse_name)
			rel_idx2name[idx + len(id2rel)] = reverse_name
			rel_idx2name_tokens[idx + len(id2rel)] = reverse_tokens

		self.ent_idx2name = ent_idx2name
		self.ent_idx2name_tokens = ent_idx2name_tokens
		self.rel_idx2name = rel_idx2name
		self.rel_idx2name_tokens = rel_idx2name_tokens
	
	def load_description(self, in_path, description_sentence_num=1):
		uid2description = self.uid2description
		uid2description_tokens = self.uid2description_tokens
		tokenizer = self.tokenizer

		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				uid, text = line.strip('\n').split('\t', 1)
				text = text.replace('@en', '').strip('"')
				if 'relation' not in in_path:
					text = text.split('.')
					if len(text) >= description_sentence_num:
						text = text[:description_sentence_num]
					text = '.'.join(text)
				if uid not in uid2description.keys():
					uid2description[uid] = text 
				tokens = tokenizer.tokenize(text)
				if uid not in uid2description_tokens.keys():
					uid2description_tokens[uid] = tokens
		self.uid2description = uid2description
		self.uid2description_tokens = uid2description_tokens

	def idx2description(self):
		'''
		convert entity idx into their corresponding description and tokens
		'''
		uid2description = self.uid2description
		uid2description_tokens = self.uid2description_tokens
		id2ent = self.id2ent
		ent_idx2description = dict()
		ent_idx2description_tokens = dict()
		for idx in list(id2ent.keys()):
			ent_idx2description[idx] = uid2description[id2ent[idx]]
			ent_idx2description_tokens[idx] = uid2description_tokens[id2ent[idx]]

		self.ent_idx2description = ent_idx2description
		self.ent_idx2description_tokens = ent_idx2description_tokens


	def adding_tokens(self):
		'''
		adding tokens of entity, relation into tokenizer
		'''

		n_ent = len(self.ent2id)
		n_rel = len(self.rel2id)

		new_tokens = ["[ent_{}]".format(i) for i in range(n_ent)] + ["[rel_{}]".format(i) for i in range(n_rel)]
		
		if self.p_tuning:
			new_tokens += ["[head_b1]", "[head_b2]", "[head_a1]", "[head_a2]", 
							"[rel_b1]", "[rel_b2]", "[rel_a1]", "[rel_a2]", 
							"[tail_b1]", "[tail_b2]", "[tail_a1]", "[tail_a2]"] # continuous prompt 
	
		self.tokenizer.add_tokens(new_tokens)


	def count_groundtruth(self):
		'''
		get groundtrouth labels t/r/h for each query (h,r)/(h,t)/(r,t)
		'''
		groundtruth = { split: {'head': {}, 'rel': {}, 'tail': {}} for split in ['all', 'train', 'valid', 'test']}
		possible_entities = { split: {'head': {}, 'tail': {}} for split in ['train']}

		for triple in self.train_idx_triplet:
			h_idx, t_idx, r_idx = triple
			groundtruth['all']['head'].setdefault((r_idx, t_idx), [])   #如果没有，则按默认值添加
			groundtruth['all']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['all']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['all']['tail'][(r_idx, h_idx)].append(t_idx)
			groundtruth['all']['rel'].setdefault((h_idx, t_idx), [])
			groundtruth['all']['rel'][(h_idx, t_idx)].append(r_idx)  
			groundtruth['train']['head'].setdefault((r_idx, t_idx), [])
			groundtruth['train']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['train']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['train']['tail'][(r_idx, h_idx)].append(t_idx) 
			groundtruth['train']['rel'].setdefault((h_idx, t_idx), [])
			groundtruth['train']['rel'][(h_idx, t_idx)].append(r_idx) 
			possible_entities['train']['head'].setdefault(r_idx, set())
			possible_entities['train']['head'][r_idx].add(h_idx)
			possible_entities['train']['tail'].setdefault(r_idx, set())
			possible_entities['train']['tail'][r_idx].add(t_idx)

		for triple in self.valid_idx_triplet:
			h_idx, t_idx, r_idx = triple
			groundtruth['all']['head'].setdefault((r_idx, t_idx), [])
			groundtruth['all']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['all']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['all']['tail'][(r_idx, h_idx)].append(t_idx)
			groundtruth['all']['rel'].setdefault((h_idx, t_idx), [])
			groundtruth['all']['rel'][(h_idx, t_idx)].append(r_idx)   
			groundtruth['valid']['head'].setdefault((r_idx, t_idx), [])
			groundtruth['valid']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['valid']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['valid']['tail'][(r_idx, h_idx)].append(t_idx)

		for triple in self.test_idx_triplet:
			h_idx, t_idx, r_idx = triple
			groundtruth['all']['head'].setdefault((r_idx, t_idx), [])
			groundtruth['all']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['all']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['all']['tail'][(r_idx, h_idx)].append(t_idx)
			groundtruth['all']['rel'].setdefault((h_idx, t_idx), [])
			groundtruth['all']['rel'][(h_idx, t_idx)].append(r_idx)   
			groundtruth['test']['head'].setdefault((r_idx, t_idx), [])
			groundtruth['test']['head'][(r_idx, t_idx)].append(h_idx)
			groundtruth['test']['tail'].setdefault((r_idx, h_idx), [])
			groundtruth['test']['tail'][(r_idx, h_idx)].append(t_idx) 

		return groundtruth, possible_entities

	def get_groundtruth(self):
		return self.groundtruth
	
	def get_dataset(self, split):
		assert (split in ['train', 'valid', 'test'])
		
		if split == 'train':
			return self.train_triplets
		elif split == 'valid':
			return self.valid_triplets
		elif split == 'test':
			return self.test_triplets

	def load_name_wiki(self, in_path):
		'''
		load map from entity uid to entity name
		return map dict
		'''
		uid2name = {}
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				_ = line.strip('\n').split('\t')
				uid = _[0]
				name = _[1]
				uid2name[uid] = name
		return uid2name
	
	def count_degrees(self):
		'''
		count the count of each head, relation and tail; and count the degree distribution
		'''
		train_set = self.train_idx_triplet #+ self.valid_idx_set + self.test_idx_set
		degrees = {}

		
		for triple in train_set:
			h_idx, t_idx, r_idx=triple
			degrees[h_idx] = degrees.get(h_idx, 0) + 1
			degrees[t_idx] = degrees.get(t_idx, 0) + 1
			# degrees[r_idx] = degrees.get(r_idx, 0) + 1

		raw_degrees = copy.deepcopy(degrees)

		max_degree = 0
		for k, v in degrees.items():
			max_degree = max(max_degree, v)
		max_degree = math.floor(math.log(max_degree) / math.log(2))  # floor 向下取整
		count_degree_group = { i:0 for i in range(0, max_degree+1)}

		for k, v in degrees.items():
			degrees[k] = math.floor(math.log(v) / math.log(2)) + 1
			count_degree_group[degrees[k]] = count_degree_group.get(degrees[k], 0) + 1

		self.statistics = {
			'degrees': raw_degrees,
			'degree_group': degrees,
			'count_degree_group': count_degree_group,
			'max_degree': max_degree
		}

	def triple_to_text(self, triple_idx, with_text):
		'''
		convert triplet into their description, using with_text to decide which conponent to save
		return h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens 
		'''
		tokenizer = self.tokenizer
		ent2id = self.ent2id
		rel2id = self.rel2id

		if True:
			# 512 tokens, 1 CLS, 1 SEP, 1 head, 1 rel, 1 tail, so 507 remaining.
			h_n_tokens = min(241, self.max_desc_length)
			t_n_tokens = min(241, self.max_desc_length)
			r_n_tokens = min(16, self.max_desc_length)

		h_idx, t_idx, r_idx = triple_idx
		if self.use_description:
			h_text_tokens =  self.ent_idx2description_tokens.get(h_idx, [])[:h_n_tokens] if with_text['h'] else []		
			# r_text_tokens =  self.rel_idx2tokens.get(r_idx, [])[:r_n_tokens] if with_text['r'] else []
			t_text_tokens =  self.ent_idx2description_tokens.get(t_idx, [])[:t_n_tokens] if with_text['t'] else []

		
		if self.add_tokens:
			if self.p_tuning:
				h_token = ["[head_b1]", "[head_b2]"] + (['[ent_{}]'.format(ent2id[h_idx])] if with_text['h'] else [tokenizer.mask_token]) + ["[head_a1]", "[head_a2]"]
				r_token = ["[rel_b1]", "[rel_b2]"] + (['[rel_{}]'.format(rel2id[r_idx])] if with_text['r'] else [tokenizer.mask_token]) + ["[rel_a1]", "[rel_a2]"]
				t_token = ["[tail_b1]", "[tail_b2]"] + (['[ent_{}]'.format(ent2id[t_idx])] if with_text['t'] else [tokenizer.mask_token]) + ["[tail_a1]", "[tail_a2]"]
			else:
				h_token = ['[ent_{}]'.format(h_idx)] if with_text['h'] else [tokenizer.mask_token]
				r_token = ['[rel_{}]'.format(r_idx)] if with_text['r'] else [tokenizer.mask_token]
				t_token = ['[ent_{}]'.format(t_idx)] if with_text['t'] else [tokenizer.mask_token]
		else:
			h_token = [tokenizer.cls_token] + self.ent_idx2name_tokens.get(h_idx, []) if with_text['h'] else [tokenizer.mask_token]
			r_token = [tokenizer.cls_token] + self.rel_idx2name_tokens.get(r_idx, []) if with_text['r'] else [tokenizer.mask_token]
			t_token = [tokenizer.cls_token] + self.ent_idx2name_tokens.get(t_idx, []) if with_text['t'] else [tokenizer.mask_token]


		if self.use_description:
			tokens = h_token + h_text_tokens + r_token + t_token + t_text_tokens 
		else:
			tokens = h_token + r_token + t_token 
		
		text = tokenizer.convert_tokens_to_string(tokens)
		'''print(h_idx, t_idx, r_idx)
		print(self.id2ent[h_idx], self.id2ent[t_idx], self.id2rel[r_idx])
		print(text)
		print(tokens)'''

		return text, tokens
	
	def element_to_text(self, target_idx, type='entity'):
		'''
		convert one target(entity/relation) to its tokens and text
		'''
		tokenizer = self.tokenizer

		n_tokens = min(508, self.max_desc_length)
		if self.use_description:
			if type == 'entity':
				text_tokens = self.ent_idx2description_tokens.get(target_idx, [])[:n_tokens]
			else:
				text_tokens = []
		
		if self.add_tokens:
			if type == 'entity':
				token = ['[ent_{}]'.format(target_idx)]
			else:
				token = ['[rel_{}]'.format(target_idx)]
		else:
			if type == 'entity':
				token = [self.tokenizer.cls_token]  + self.ent_idx2name_tokens.get(target_idx, [])
			else:
				token = [self.tokenizer.cls_token]  + self.rel_idx2name_tokens.get(target_idx, [])
		if self.use_description:
			tokens = token + text_tokens 
		else:
			tokens = token
		
		text = tokenizer.convert_tokens_to_string(tokens)

		return text, tokens
	
	def batch_tokenize(self, batch_triples, mode):
		'''
		create a batch of tokens of train data, return as a dictionary; and return the position of h, r, t
		'''
		batch_texts = []
		batch_tokens = []
		batch_positions = []

		ent2id = self.ent2id
		rel2id = self.rel2id

		if mode in ['triple_classification']:
			with_text = {'h': True, 'r': True, 't': True}
		elif mode == "link_prediction_h":
			with_text = {'h': False, 'r': True, 't': True}
		elif mode == "link_prediction_r":
			with_text = {'h': True, 'r': False, 't': True}
		elif mode == "link_prediction_t":
			with_text = {'h': True, 'r': True, 't': False}


		for triple in batch_triples:
			text, tokens = self.triple_to_text(triple, with_text)  #h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens
			batch_texts.append(text)
			batch_tokens.append(tokens)

			# h_idx, t_idx, r_idx = triple
		# print('text read finished!!!')

		#batch_tokens_ = self.tokenizer(batch_texts, truncation = True, max_length = 512, return_tensors='pt', padding=True )
		batch_tokens = self.my_tokenize(batch_tokens, max_length=512, padding=True, model=self.model)  # from token to token_idx
		# print('toknes converted finished!!!')

		orig_vocab_size = self.orig_vocab_size
		num_ent_rel_tokens = len(ent2id) + len(rel2id)

		mask_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
		cls_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

		
		for i, _ in enumerate(batch_tokens['input_ids']):
			triple = batch_triples[i]
			h_idx, t_idx, r_idx = triple

			#import pdb
			#pdb.set_trace()
			if not self.add_tokens:
				cls_pos, h_pos, r_pos, t_pos = torch.where((_==mask_idx) + (_==cls_idx))[0]
			else:
				h_pos, r_pos, t_pos = torch.where( (_ >= orig_vocab_size) * (_ < orig_vocab_size + num_ent_rel_tokens) + (_ == mask_idx) )[0]


			batch_positions.append({'head': (h_idx, h_pos.item()), 'rel': (r_idx, r_pos.item()), 'tail': (t_idx, t_pos.item())})

		return batch_tokens, batch_positions

	def batch_tokenize_target(self, batch_triples=None, mode=None, targets = None):
		'''
		create a batch of target tokens, only used for contrasitive learning
		'''
		batch_texts = []
		batch_tokens = []
		batch_positions = []

		if targets == None:
			if mode == "link_prediction_h":
				targets = [ triple[0] for triple in batch_triples]  # idx
				type_ = 'entity'
			elif mode == "link_prediction_r":
				targets = [ triple[2] for triple in batch_triples]
				type_ = 'relation'
			elif mode == "link_prediction_t":
				targets = [ triple[1] for triple in batch_triples]
				type_ = 'entity'
		else:
			type_ = 'entity'

		for target_idx in targets:
			text, tokens = self.element_to_text(target_idx=target_idx, type=type_)
			batch_texts.append(text)
			batch_tokens.append(tokens)
			# print(target_idx, self.id2ent[target_idx],text, tokens)
		
		batch_tokens = self.my_tokenize(batch_tokens, max_length=512, padding=True, model=self.model)

		for i, _ in enumerate(batch_tokens['input_ids']):
			target_pos = 1
			batch_positions.append( (target_idx, target_pos) )

		return batch_tokens, batch_positions
	
	def batch_tokenize_element(self, targets, type='relation'):
		'''
		create a batch of target tokens, only used for contrasitive learning
		'''
		batch_texts = []
		batch_tokens = []
		batch_positions = []

		for target_idx in targets:
			text, tokens = self.element_to_text(target_idx=target_idx, type=type)
			batch_texts.append(text)
			batch_tokens.append(tokens)
		
		batch_tokens = self.my_tokenize(batch_tokens, max_length=512, padding=True, model=self.model)

		for i, _ in enumerate(batch_tokens['input_ids']):
			target_pos = 1
			batch_positions.append( (target_idx, target_pos) )

		return batch_tokens, batch_positions


	def my_tokenize(self, batch_tokens, max_length=512, padding=True, model='roberta'):
		'''
		convert tokens into index
		if model == 'roberta':
			start_tokens = ['<s>']
			end_tokens = ['</s>']
			pad_token = '<pad>'
		elif model == 'bert':
			start_tokens = ['[CLS]']
			end_tokens = ['[SEP]']
			pad_token = '[PAD]'
		'''

		start_tokens = [self.tokenizer.cls_token]
		end_tokens = [self.tokenizer.sep_token]


		batch_tokens = [ start_tokens + i + end_tokens for i in batch_tokens] 

		batch_size = len(batch_tokens)
		longest = min(max([len(i) for i in batch_tokens]), 512)

		if model == 'bert':
			input_ids = torch.zeros((batch_size, longest)).long()
		elif model == 'roberta':
			input_ids = torch.ones((batch_size, longest)).long()

		token_type_ids = torch.zeros((batch_size, longest)).long()
		attention_mask = torch.zeros((batch_size, longest)).long()


		for i in range(batch_size):
			tokens = self.tokenizer.convert_tokens_to_ids(batch_tokens[i])
			input_ids[i, :len(tokens)] = torch.tensor(tokens).long() 
			attention_mask[i, :len(tokens)] = 1

		if model == 'roberta':
			data = {'input_ids': input_ids, 'attention_mask': attention_mask}
			return BatchEncoding(data)
		elif model == 'bert':
			data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
			return BatchEncoding(data)
		


	def link_prediction_dataset(self, mode):
		if not os.path.exists('./sampler'):
			os.makedirs('./sampler')

		dataset = []
		dataset_path = 'sampler/{}-{}.pkl'.format(self.datasetName, mode)
		if os.path.exists(dataset_path):
			with open(dataset_path, 'rb') as fil:
				dataset = pickle.load(fil)
			print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
			print('dataset is loaded from {}'.format(dataset_path))
		else:
			if mode == 'train':
				pos_triplet = self.train_idx_triplet
			elif mode == 'valid':
				pos_triplet = self.valid_idx_triplet
			elif mode == 'test':
				pos_triplet = self.test_idx_triplet
			

			for i in range(len(pos_triplet)):
				triplet = pos_triplet[i]
				triplet_label = 1
				'''hr_token, hr_position = self.batch_tokenize(triplet, mode='link_prediction_t')[0], self.batch_tokenize(triplet, mode='link_prediction_t')[1][0]
				t_token, t_position = self.batch_tokenize_target(triplet, mode='link_prediction_t')[0], self.batch_tokenize_target(triplet, mode='link_prediction_t')[1][0]
				rt_token, rt_positon = self.batch_tokenize(triplet, mode='link_prediction_h')[0], self.batch_tokenize(triplet, mode='link_prediction_h')[1][0]
				h_token, h_positon = self.batch_tokenize_target(triplet, mode='link_prediction_h')[0], self.batch_tokenize_target(triplet, mode='link_prediction_h')[1][0]'''
				dataset.append((triplet, 1))
				# dataset.append((triplet[0], triplet_label, hr_token, hr_position, t_token, t_position, rt_token, rt_positon, h_token, h_positon))
			with open(dataset_path, 'wb') as fil:
				pickle.dump(dataset, fil)

		return dataset
	

	def triplet_classification_dataset(self, mode):
		if mode =='valid':
			pos_neg_dataset = self.valid_idx_triplets_with_neg 
		elif mode == 'test':
			pos_neg_dataset = self.test_idx_triplets_with_neg


		if self.datasetName in ['fb13'] and mode != 'train':
			triplet_with_neg = [ ((i[0][0], i[0][1], i[0][2]), i[1]) for i in pos_neg_dataset]
		else:
			if self.mode == 'train':
				pos_triplet = self.train_idx_triplet
			elif self.mode == 'valid':
				pos_triplet = self.valid_idx_triplet
			elif self.mode == 'test':
				pos_triplet == self.test_idx_triplet
			triplet_with_neg = self.negtive_sampling(pos_triplet)
		
		if not os.path.exists('./sampler'):
			os.makedirs('./sampler')
		dataset = []
		dataset_path = 'sampler/{}-{}.pkl'.format(self.datasetName, mode)
		if os.path.exists(dataset_path):
			with open(dataset_path, 'rb') as fil:
				dataset = pickle.load(fil)
				print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
				print('dataset is loaded from {}'.format(dataset_path))
		else:
			for i in range(len(triplet_with_neg)):
				triplet = triplet_with_neg[i][0]
				triplet_label = triplet_with_neg[i][1]
				# token, position = self.batch_tokenize(triplet, mode='triplet_classification')[0], self.batch_tokenize(triplet, mode='triplet_classification')[1][0]
				dataset.append((triplet, triplet_label))
			with open(dataset_path, 'wb') as fil:
				pickle.dump(dataset, fil)
		return dataset


	def negtive_sampleing(self, pos_dataset):
		dataset = [] 
		random.shuffle(pos_dataset)
		pos_dataset_set = set(pos_dataset)
		whole_dataset_set = set(self.whole_dataset)

		if self.mode == 'train':
			random_ratio = 1#1/3
			constrain_ratio = 0#1/3
			reverse_ratio = 0#1/3
			viewable = 'train'
			viewable_set = pos_dataset_set
		else:
			random_ratio = 1
			constrain_ratio = 0
			reverse_ratio = 0
			viewable = 'all'
			viewable_set = whole_dataset_set

		for triple in pos_dataset:
			dataset.append((triple, 1))
			choice = random_choose(random_ratio, constrain_ratio, reverse_ratio)
			h_idx, t_idx, r_idx = triple
			for _ in range(self.neg_rate):
				count = 0
				while (True):
					if (random.sample(range(2), 1)[0] == 1):
						# replace head
						if choice == 'random':
							candidate_ents = self.entity_set - set(self.groundtruth[viewable]['head'][(r_idx, t_idx)])
							replace_ent_idx = random.sample(candidate_ents, 1)[0]
							neg_triple_idx = (replace_ent_idx, t_idx, r_idx)
						elif choice == 'constrain':
							candidate_ents = self.possible_entities['train']['head'][r_idx] - set(self.groundtruth[viewable]['head'][(r_idx, t_idx)])
							# head that are head of rel, but not head of (rel, tail) 
							if len(candidate_ents) == 0:
								candidate_ents = self.entity_set - set(self.groundtruth[viewable]['head'][(r_idx, t_idx)])
							replace_ent_idx = random.sample(candidate_ents, 1)[0]
							neg_triple_idx = (replace_ent_idx, t_idx, r_idx)
						else: # choice == 'reverse'
							neg_triple_idx = (t_idx, h_idx, r_idx)
					else:
						# replace tail
						if choice == 'random':
							candidate_ents = self.entity_set - set(self.groundtruth[viewable]['tail'][(r_idx, h_idx)])
							replace_ent_idx = random.sample(candidate_ents, 1)[0]
							neg_triple_idx = (h_idx, replace_ent_idx, r_idx)
						elif choice == 'constrain':
							candidate_ents = self.possible_entities['train']['tail'][r_idx] - set(self.groundtruth[viewable]['tail'][(r_idx, h_idx)])
							# tail that are tail of rel, but not tail of (rel, head) 
							if len(candidate_ents) == 0:
								candidate_ents = self.entity_set - set(self.groundtruth[viewable]['tail'][(r_idx, h_idx)])
							replace_ent_idx = random.sample(candidate_ents, 1)[0]
							neg_triple_idx = (h_idx, replace_ent_idx, r_idx)
						else: # choice == 'reverse':
							neg_triple_idx = (t_idx, h_idx, r_idx)

					if neg_triple_idx not in viewable_set:
						dataset.append((neg_triple_idx, 0))
						break 
					elif choice == 'reverse':
						dataset.append((neg_triple_idx, 1))


		return dataset
	
	def caculate_relation_embedding(self):
		lm_model = self.lm_model
		device = self.gpu_device
		lm_model.to(device)

		# n_ent =len(self.ent2id)
		n_rel = len(self.rel2id) * 2
		# entity_list = list(range(n_ent))
		rel_list = list(range(n_rel))
		batch_size_target = 128
		# ent_target_encoded = torch.zeros((n_ent, lm_model.config.hidden_size)).to(device)	
		rel_target_encoded = torch.zeros((n_rel, lm_model.config.hidden_size)).to(device)	
		
		with torch.no_grad():

			'''random_map = [ i for i in range(n_ent)]
			batch_list = [ random_map[i:i+batch_size_target] for i in range(0, n_ent, batch_size_target)] 
		
			for batch in batch_list:
				batch_targets = [ entity_list[_] for _ in batch]
				target_inputs, target_positions = self.batch_tokenize_element(batch_targets, type='entity')
				target_inputs.to(device)
				logits = lm_model(**target_inputs) 

				target_embs = logits[0][:, 1, :]
				ent_target_encoded[batch] = target_embs'''

			random_map_relation = [ i for i in range(n_rel)]
			batch_list_relation = [ random_map_relation[i:i+batch_size_target] for i in range(0, n_rel, batch_size_target)] 
			for batch in batch_list_relation:
				batch_targets = [ rel_list[_] for _ in batch]
				inputs, positions = self.batch_tokenize_element(batch_targets, type='relation')
				inputs.to(device)
				logits = lm_model(**inputs) 

				target_embs = logits[0][:, 1, :]

				rel_target_encoded[batch] = target_embs
		
		return  rel_target_encoded


			
	@property
	def num_entity(self):
		"""Number of entities."""
		return self.graph.num_node
	
	@property
	def num_triplet(self):
		"""Number of triplets"""
		return self.graph.num_edge
	
	@property
	def num_relation(self):
		"""Number of relations."""
		return self.graph.num_relation
	
	def __repr__(self):
		lines = [
			"#entity: %d" % self.num_entity,
			"#relation: %d" % self.num_relation,
			"#triplet: %d" % self.num_triplet,
		]
		return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


class Dataset(torch_data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		self.size = len(dataset)
	def __len__(self):
		return self.size
	
	def __getitem__(self, index):
		return self.dataset[index]


def random_choose(random_ratio, constrain_ratio, reverse_ratio):
	'''
	decide the selection of choose by a random number
	'''
	x = random.random()
	if x <= random_ratio:
		return 'random'
	elif x <= (random_ratio + constrain_ratio):
		return 'constrain'
	else:
		return 'reverse'