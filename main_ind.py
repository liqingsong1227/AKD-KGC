# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
# from torch.optim import Adam


from trainer_distlition_ind import Trainer
from data_loader import DataProcess
from models_ind import Model
from nbfnet import NBFNet
from lmke_ind import LMKE
import time
from torch.optim import SGD, Adam

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler

from functools import partial



from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
	init_process_group(backend="nccl")
	torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def multigraph_collator(batch, train_graphs, embeddings):
	num_graphs = len(train_graphs)
	probs = torch.tensor([graph.edge_index.shape[1] for graph in train_graphs]).float()
	probs /= probs.sum()
	graph_id = torch.multinomial(probs, 1, replacement=False).item()

	graph = train_graphs[graph_id]
	embedding = embeddings[graph_id]
	bs = len(batch)
	edge_mask = torch.randperm(graph.target_edge_index.shape[1])[:bs]

	batch = torch.cat([graph.target_edge_index[:, edge_mask], graph.target_edge_type[edge_mask].unsqueeze(0)]).t()
	return graph, batch, embedding

def singlegraph_collator(batch, train_graph):
	# print(batch)
	# real_triples = [(int(batch[i][0][0]), int(batch[i][0][1]), int(batch[i][0][2])) for i in range(len(batch))]
	# batch = torch.LongTensor(real_triples)
	batch = torch.stack(batch)
	return train_graph, batch


if __name__ == '__main__':
	# argparser
	parser = argparse.ArgumentParser()
	# connon hyperparams
	parser.add_argument('--seed', type=int, default=1024)
	parser.add_argument("--local_rank", default=-1)
	parser.add_argument('--weight_decay', type=float, default=1e-7) 
	parser.add_argument('--data', type=str, default=None)
	parser.add_argument('--version', type=str, default='v1')
	

	# language hyperparams
	parser.add_argument('--plm', type=str, default='bert', choices = ['bert', 'bert_tiny', 'deberta', 'deberta_large', 'roberta', 'roberta_large'])
	parser.add_argument('--description_sentence_num', type=int, default=1, help='sentence num of description of entity')
	parser.add_argument('--use_description', default=False, action = 'store_true', help='whether use description of entity')
	parser.add_argument('--max_desc_length', type=int, default=512)
	parser.add_argument('--add_tokens', default=False, action = 'store_true', help = 'add entity and relation tokens into the vocabulary')
	parser.add_argument('--p_tuning', default=False, action = 'store_true', help = 'add learnable soft prompts')
	



	# train hyperparams
	parser.add_argument('--optim', type=str, default='adam', choices =['sgd', 'adam', 'adamw'])
	parser.add_argument('--scheduler', type=str, default='no_scheduler', choices=['no_scheduler', 'linear', 'constant', 'cosine'])
	parser.add_argument('--scheduler_step', type=int, default=0)
	parser.add_argument('--feature_dim', type=int, default=64)
	parser.add_argument('--bert_lr', type=float, default=5e-5)
	parser.add_argument('--no_bert_lr', type=float, default=2e-3)
	parser.add_argument('--nbf_lr', type=float, default=2e-2)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--test_batch_size', type=int, default=64)
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--load_path', type=str, default=None)
	parser.add_argument('--teacher_load_path', type=str, default=None)
	parser.add_argument('--load_epoch', type=int, default=-1)
	parser.add_argument('--load_batch', type=int, default=-1)
	parser.add_argument('--load_metric', type=str, default='hits1')
	parser.add_argument('--finetune', default=False, action = 'store_true')
	parser.add_argument('--temperature', type=float, default=1e+10, help='temperature of softmax in teacher weight')
	parser.add_argument('--kl_temperature', type=float, default=1, help = 'temperature of softmax in kl-divergence')


	# directly run test
	parser.add_argument('--link_prediction', default=False, action = 'store_true')
	arg = parser.parse_args()

	

	# Set random seed
	random.seed(arg.seed)
	np.random.seed(arg.seed)
	torch.manual_seed(arg.seed)
	torch.cuda.manual_seed(arg.seed)
	torch.cuda.manual_seed_all(arg.seed)
	ddp_setup()
	
	local_rank = int(os.environ["LOCAL_RANK"])
	device = torch.device("cuda", local_rank)
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


	if arg.plm == 'bert':
		plm_name = "bert-base-uncased"
		t_model = 'bert'
	elif arg.plm == 'bert_tiny':
		plm_name = "prajjwal1/bert-tiny"
		t_model = 'bert'
	elif arg.plm =='deberta':
		plm_name = 'microsoft/deberta-v3-base'
		t_model = 'bert'
	elif arg.plm == 'deberta_large':
		plm_name = 'microsoft/deberta-v3-large'
		t_model = 'bert'
	elif arg.plm == 'roberta_large':
		plm_name = "roberta-large"
		t_model = 'roberta'
	elif arg.plm == 'roberta':
		plm_name = "roberta-base"
		t_model = 'roberta'

	lm_config = AutoConfig.from_pretrained(plm_name, cache_dir = './cached_model')
	lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False, cache_dir = './cached_model')
	lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir = './cached_model')

	if arg.data == 'fb15k-237':
		dataset_name = 'fb237' + '_' + arg.version
	elif arg.data == 'WN18RR':
		dataset_name = 'WN18RR' + '_' + arg.version
	
	
	train_path = {
			'dataset': dataset_name,
			'train': './data/inductive/{}/train.tsv'.format(dataset_name),
			'valid': './data/inductive/{}/valid.tsv'.format(dataset_name),
			'test': './data/inductive/{}/test.tsv'.format(dataset_name),
			'description': './data/pretrain/{}/entity2description.txt'.format(arg.data),
			'name':['./data/pretrain/{}/entity2name.txt'.format(arg.data), './data/pretrain/{}/relation2text.txt'.format(arg.data)],
			'entitydict':'./data/inductive/{}/entities.dict'.format(dataset_name),
			'relationdict':'./data/inductive/{}/relations.dict'.format(dataset_name)
		}
		
	test_path = {
			'dataset': dataset_name + '_ind',
			'train': './data/inductive/{}_ind/train.tsv'.format(dataset_name),
			'valid': './data/inductive/{}_ind/valid.tsv'.format(dataset_name),
			'test': './data/inductive/{}_ind/test.tsv'.format(dataset_name),
			'description': './data/pretrain/{}/entity2description.txt'.format(arg.data),
			'name':['./data/pretrain/{}/entity2name.txt'.format(arg.data), './data/pretrain/{}/relation2text.txt'.format(arg.data)],
			'entitydict':'./data/inductive/{}_ind/entities.dict'.format(dataset_name),
			'relationdict':'./data/inductive/{}/relations.dict'.format(dataset_name)
		}



	no_decay = ["bias", "LayerNorm.weight"]
	print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
	print('----------start load datasets----------')


	train_data = DataProcess(train_path, lm_tokenizer, lm_model, max_desc_length = arg.max_desc_length, add_tokens = arg.add_tokens, 
					p_tuning = arg.p_tuning, model = t_model, description_sentence_num=arg.description_sentence_num, use_description=arg.use_description, device=device)
	test_data = DataProcess(test_path, lm_tokenizer, lm_model, max_desc_length = arg.max_desc_length, add_tokens = arg.add_tokens, 
					p_tuning = arg.p_tuning, model = t_model, description_sentence_num=arg.description_sentence_num, use_description=arg.use_description, device=device)
	if arg.add_tokens:
		train_data.adding_tokens()
		lm_model.resize_token_embeddings(len(lm_tokenizer))

	train_dataset = torch.cat([train_data.train_data.target_edge_index, train_data.train_data.target_edge_type.unsqueeze(0)]).t()
	valid_dataset = torch.cat([train_data.valid_data.target_edge_index, train_data.valid_data.target_edge_type.unsqueeze(0)]).t()
	test_dataset = torch.cat([test_data.test_data.target_edge_index, test_data.test_data.target_edge_type.unsqueeze(0)]).t()

	train_graph = train_data.train_data
	test_graph = test_data.train_data


	train_data_loader = DataLoader(train_dataset, batch_size=arg.batch_size, pin_memory=False, shuffle=False, sampler=DistributedSampler(train_dataset, rank=local_rank),collate_fn=partial(singlegraph_collator, train_graph=train_graph))
	valid_data_loader = DataLoader(valid_dataset, batch_size=arg.test_batch_size, pin_memory=False, shuffle=False, sampler=SequentialSampler(valid_dataset))
	test_data_loader = DataLoader(test_dataset, batch_size=arg.test_batch_size, pin_memory=False, shuffle=False, sampler=SequentialSampler(test_dataset))
	
	data_loader_list = [train_data_loader, valid_data_loader, test_data_loader]
	identifier = 'both-teacher-temperature-{}-{}-batch_size={}-max_desc_length={}-data-{}'.format(arg.temperature, arg.plm, arg.batch_size, arg.max_desc_length, dataset_name)
	
	if arg.data == 'fb15k-237':
		nbf_model_cfg_15k = {'input_dim': 32,'hidden_dims': [32, 32, 32, 32, 32, 32],'message_func': 'distmult', 'aggregate_func': 'pna',
					'short_cut': True, 'layer_norm': True, 'num_relation':train_data.train_data.num_relations, 'remove_one_hop':True, 'dependent':True}
	elif arg.data == 'wn18rr':
		nbf_model_cfg = {'input_dim': 32,'hidden_dims': [32, 32, 32, 32, 32, 32],'message_func': 'distmult', 'aggregate_func': 'pna',
					'short_cut': True, 'layer_norm': True, 'num_relation':train_data.train_data.num_relations, 'remove_one_hop':True, 'dependent':True}
	

	ultra = Model(nbf_model_cfg_15k, lm_model) 

	

	
	no_bert_param = [{'lr': arg.no_bert_lr, 'params' : [p for n, p in ultra.lmke_model.named_parameters()
									if ('lm_model' not in n) and (not any(nd in n for nd in no_decay))],
		'weight_decay': arg.weight_decay},
		{'lr': arg.no_bert_lr, 'params': [p for n, p in ultra.lmke_model.named_parameters()
									if ('lm_model' not in n) and (any(nd in n for nd in no_decay))],
		'weight_decay': 0.0}]
	no_bert_param_name = list(set(['.'.join(n.split('.')[:2]) for n, p in ultra.lmke_model.named_parameters() if ('lm_model' not in n)])) 
	bert_param = [	
		{'lr': arg.bert_lr, 'params' : [p for n, p in ultra.lmke_model.named_parameters()
									if ('lm_model' in n) and (not any(nd in n for nd in no_decay))],
		'weight_decay': arg.weight_decay},
		{'lr': arg.bert_lr, 'params': [p for n, p in ultra.lmke_model.named_parameters()
									if ('lm_model' in n) and (any(nd in n for nd in no_decay))],
		'weight_decay': 0.0}
	]
	bert_param_name =list(set(['.'.join(n.split('.')[:2]) for n, p in ultra.lmke_model.named_parameters() if ('lm_model' in n) ]))
	mlp_param = [{'lr': arg.no_bert_lr, 'params' : [p for n, p in ultra.mlp_merge.named_parameters()
									if  (not any(nd in n for nd in no_decay))],
		'weight_decay': arg.weight_decay},
		{'lr': arg.no_bert_lr, 'params': [p for n, p in ultra.mlp_merge.named_parameters()
									if  (any(nd in n for nd in no_decay))],
		'weight_decay': 0.0}]
	mlp_param_name = list(set(['.'.join(n.split('.')[:2]) for n, p in ultra.mlp_merge.named_parameters()]))

	param_group =  bert_param + no_bert_param + mlp_param
	total_param_name =  bert_param_name + no_bert_param_name  + mlp_param_name

	if int(os.environ["LOCAL_RANK"])==0:
		print('all updated params:')
		print(total_param_name)


	
	if arg.optim == 'adamw':
		optimizer = AdamW(param_group)  # transformer AdamW
	elif arg.optim == 'adam':
		optimizer = Adam(param_group)  
	elif arg.optim == 'sgd':
		optimizer = SGD(param_group)
	else:
		optimizer = Adam(param_group)  
	


	if int(os.environ["LOCAL_RANK"])==0:
		print('Using schedular: {} with warmup steps: {}'.format(arg.scheduler, arg.scheduler_step))
	if arg.scheduler == 'no_scheduler':
		scheduler = None
	elif arg.scheduler == 'constant':
		scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=arg.scheduler_step)
	elif arg.scheduler == 'linear':
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=arg.scheduler_step, num_training_steps=arg.scheduler_step * arg.epoch)
	elif arg.scheduler == 'cosine':
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=arg.scheduler_step, num_training_steps=arg.scheduler_step * arg.epoch)
	# 
	hyperparams = {
		'data' : dataset_name,
		'batch_size': arg.batch_size,
		'epoch': arg.epoch,
		'optim': arg.optim,
		'scheduler' : arg.scheduler,
		'scheduler_step' : arg.scheduler_step,
		'bert_lr':arg.bert_lr,
		'no_bert_lr':arg.no_bert_lr,
		'nbf_lr':arg.nbf_lr,
		'identifier': identifier,
		'load_path': arg.load_path,
		'teacher_load_path': arg.teacher_load_path,
		'evaluate_every': 1, 
		'update_every': 1,
		'load_epoch': arg.load_epoch,
		'load_batch' : arg.load_batch,
		'load_metric': arg.load_metric,
		'num_negative' :32,
		'strict_negative':True, 
		'finetune' : arg.finetune,
		'nbf_hyper_param' : nbf_model_cfg_15k,
		'temperature' : arg.temperature,
		'kl_temperature': arg.kl_temperature,
		'adversarial_temperature' : 0.5

	}
	if int(os.environ["LOCAL_RANK"]) == 0:
		print(hyperparams)
	
	
	trainer = Trainer(train_data, test_data, data_loader_list, ultra, optimizer, scheduler, device=device, hyperparams=hyperparams) # , teacher_model=teacher_model
	if arg.finetune:
		trainer.run()
	if arg.link_prediction and int(os.environ["LOCAL_RANK"]) == 0:
		# trainer.link_prediction()

		trainer.link_prediction(split='test') # 
	

	destroy_process_group()
