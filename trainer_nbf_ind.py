import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import os
import pickle
import numpy as np
import wandb
import pytorch_warmup as warmup
from torchdrug.layers import functional
from torch.nn.parallel import DistributedDataParallel as DDP
import copy

from transformers import BatchEncoding
import tasks
torch.set_printoptions(profile='full')
save_folder = './params/'

max_length = 512
sample_limit = 10

margin = 9

class Trainer:
	def __init__(self,train_data, test_data, data_loader_list, model,  optimizer, scheduler, device, hyperparams, ):
		model.to(device)

		self.train_data = train_data
		self.test_data = test_data
		self.train_data_loader = data_loader_list[0]
		self.valid_data_loader = data_loader_list[1]
		self.test_data_loader = data_loader_list[2]
		

		self.model = DDP(model,device_ids=[device], find_unused_parameters=False)
		self.scheduler = scheduler

		self.num_negative = hyperparams['num_negative']
		self.strict_negative = hyperparams['strict_negative']
		self.adversarial_temperature = hyperparams['adversarial_temperature']

		self.optimizer = optimizer
		self.device = device
		self.identifier = hyperparams['identifier']
		self.hyperparams = hyperparams
		self.save_folder = save_folder + hyperparams['data'] + '/'
		self.load_epoch = hyperparams['load_epoch']
		self.load_batch = hyperparams['load_batch']
		


		self.result_log = self.save_folder + self.identifier + '.txt'
		self.param_path_template = self.save_folder + self.identifier + '-epc_{0}-batch_{1}_metric_{2}'  + '.pt'
		self.history_path = self.save_folder + self.identifier + '-history_{0}'  + '.pkl'


		self.best_metric = {'acc': 0, 'f1': 0, 
			'raw_mrr': 0, 'raw_hits1': 0, 'raw_hits3': 0, 'raw_hits10': 0,
			'fil_mr': 100000000000, 'fil_mrr': 0, 'fil_hits1': 0, 'fil_hits3': 0, 'fil_hits10': 0,
		}

		self.best_epoch = {'acc': -1, 'f1': -1, 
			'raw_mrr': -1, 'raw_hits1': -1, 'raw_hits3': -1, 'raw_hits10': -1,
			'fil_mr': -1, 'fil_mrr': -1, 'fil_hits1': -1, 'fil_hits3': -1, 'fil_hits10': -1,
		}
		self.best_batch = {'acc': -1, 'f1': -1, 
			'raw_mrr': -1, 'raw_hits1': -1, 'raw_hits3': -1, 'raw_hits10': -1,
			'fil_mr': -1, 'fil_mrr': -1, 'fil_hits1': -1, 'fil_hits3': -1, 'fil_hits10': -1,
		}
		self.history_value = {'acc': [], 'f1': [], 
			'raw_mrr': [], 'raw_hits1': [], 'raw_hits3': [], 'raw_hits10': [],
			'fil_mr': [], 'fil_mrr': [], 'fil_hits1': [], 'fil_hits3': [], 'fil_hits10': [],
		}


		if not os.path.exists(save_folder):
			os.makedirs(save_folder)


		load_path = hyperparams['load_path']
		if load_path == None and self.load_epoch >= 0:
			load_path = self.param_path_template.format(self.load_epoch, self.load_batch, hyperparams['load_metric'])
			history_path = self.history_path.format(self.load_epoch)
			if os.path.exists(history_path):
				with open(history_path, 'rb') as fil:
					self.history_value = pickle.load(fil)

		# self.nbf_score = self.save_belman_score()
		# print('Teacher Score save finished!!\n')
		
		if load_path != None:
			if not (load_path.startswith(save_folder) or load_path.startswith('./saveparams/')):
				load_path = save_folder + load_path

			if os.path.exists(load_path):
				
				try:
					checkpoint = torch.load(load_path)
					# model.load_state_dict(checkpoint['model'], strict=False)
					model.load_state_dict(checkpoint['model'], strict=False)
					# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					print('Model & Optimizer  Parameters loaded from {0}.'.format(load_path))
				except:
					model.load_state_dict(torch.load(load_path), strict=False)
					print('Parameters loaded from {0}.'.format(load_path))
			else:
				print('Parameters {0} Not Found'.format(load_path))

		self.load_path = load_path
		import signal
		signal.signal(signal.SIGINT, self.debug_signal_handler)

	def run(self):
		self.train()

	def train(self):
		nbf_model = self.model
		optimizer = self.optimizer
		scheduler = self.scheduler
		device = self.device
		hyperparams = self.hyperparams
		epoch = hyperparams['epoch'] 
		data = self.train_data
		data_loader = self.train_data_loader
		groundtruth = data.get_groundtruth()

		# rel_embedding = self.rel_embddding
		# )
		# data_loader = self.train_data_loader

		nbf_model.train()

		'''if hyperparams['finetune'] and int(os.environ["LOCAL_RANK"])==0:
			print('Evluating pretrained parameters!!!!')
			self.link_prediction(-1, batch_num=0)'''
		
		modes = ["link_prediction_t", "link_prediction_h"]
		for epc in range(self.load_epoch+1, epoch):

			data_loader.sampler.set_epoch(epoch)
			for i_b, batch_ in enumerate(data_loader):
				train_data, batch = batch_
				batch = batch.to(device)
				train_data.to(device)
				batch = tasks.negative_sampling(train_data, batch, self.num_negative,
											strict=self.strict_negative)
				pred = nbf_model(train_data, batch) #   # 

				target = torch.zeros_like(pred)
				target[:, 0] = 1
				loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
				neg_weight = torch.ones_like(pred)
				if self.adversarial_temperature > 0:
					with torch.no_grad():
						neg_weight[:, 1:] = F.softmax(pred[:, 1:] / self.adversarial_temperature, dim=-1)
				else:
					neg_weight[:, 1:] = 1 / self.num_negative
				loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
				loss = loss.mean()
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				if scheduler != None:
					scheduler.step()
				if (i_b+1) % 100 == 0 and int(os.environ["LOCAL_RANK"])==0:
					print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()),' Train Epoch {}, Batch {}, Batch_Loss: {}'.format(epc, i_b+1,  loss.item()))
				if (i_b+1) % 500  == 0 and int(os.environ["LOCAL_RANK"])==0:
					self.link_prediction(epc, batch_num=i_b+1)


			if int(os.environ["LOCAL_RANK"])==0:
				print('Evluating at Epoch: {}, \n'.format(epc))
				self.link_prediction(epc, batch_num=None)

		self.log_best()

	
	def link_prediction(self, epc=-1, batch_num=None, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams
		if split == 'valid':
			data = self.train_data
			data_loader = self.valid_data_loader
		elif split == 'test':
			data = self.test_data
			data_loader = self.test_data_loader
		
			


		model.eval()

		sigmoid = torch.nn.Sigmoid()


		ks = [1, 3, 10]
		MR =  {target: 0 for target in ['head', 'tail']} 


		MRR = {target: 0 for target in ['head', 'tail']} 
		hits = {target: {k: 0 for k in ks} for target in ['head', 'tail']} 
		h_10_50 = {target: 0 for target in ['head', 'tail']} 
	


		test_data = data.test_data
		
		groundtruth = data.get_groundtruth()

		batch_size_target = 128

		with torch.no_grad():
			modes = ["link_prediction_t", "link_prediction_h"]
			for mode in  modes:
				count_triples = 0
				score = 0
				for i_b, batch in enumerate(data_loader):

					batch = batch.to(device)
					test_data.to(device)
					t_batch, h_batch = tasks.all_negative(test_data, batch)

					if mode == "link_prediction_h":
						target = 'head'	
						preds =  model(test_data, h_batch, training=False)
					elif mode == "link_prediction_t":
						target = 'tail'	
						preds =  model(test_data, t_batch, training=False) #  # 
					
					for i_sample in range(batch.shape[0]):
						pred = preds[i_sample, :]
						triple = (int(batch[i_sample,0]), int(batch[i_sample,1]), int(batch[i_sample,2]))
						if mode == 'link_prediction_h':
							given_ent = triple[1]
							expect = triple[0]
						elif mode == 'link_prediction_t':
							given_ent = triple[0]
							expect = triple[1]
						

						given_key = (triple[-1], given_ent)
						
						corrects = groundtruth['all'][target][given_key]
						scores = pred.squeeze() 
						tops = scores.argsort(descending=True).tolist()
						other_corrects = [correct for correct in corrects if correct != expect]
						other_correct_ids = set([c for c in other_corrects])
						tops_ = [ t for t in tops if (not t in other_correct_ids)]
						rank = tops_.index(expect) + 1 
						MRR[target] += 1/rank 
						MR[target] += rank 		
						for k in ks:
							if rank <= k:
								hits[target][k] += 1
						num_sample = 50
						fp_rate = float((rank - 1)) / float(len(tops_))
						
						for i in range(10):
							num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
							score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i))
						count_triples += 1

				MR[target] /= count_triples
				MRR[target] /= count_triples
				for k in ks:
					hits[target][k] /= count_triples

				h_10_50[target] = score / count_triples
				print('MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f} 10_50 {7:.5f}, Dataset: {5} Target: {6} '.format(
					MR[target], MRR[target], hits[target][1], hits[target][3], hits[target][10],
					hyperparams['data'], target, score/count_triples
					))
		mr = (MR['head'] + MR['tail']) / 2
		mrr = (MRR['head'] + MRR['tail']) / 2
		hits1 = (hits['head'][1] + hits['tail'][1]) / 2
		hits3 = (hits['head'][3] + hits['tail'][3]) / 2
		hits10 = (hits['head'][10] + hits['tail'][10]) / 2
		hits_10_50 = (h_10_50['head'] + h_10_50['tail']) / 2

		print('Overall of Dataset {0:^10}: MR {1:.5f} MRR {2:.5f} hits 1 {3:.5f} 3 {4:.5f} 10 {5:.5f} 10_50 {6:.5f}, Setting : Filter.'.format(hyperparams['data'],
			mr, mrr, hits1, hits3, hits10, hits_10_50))
		




		if split != 'test':
			# self.save_model(epc, batch_num, 'fil_mr', fil_mr)
			self.update_metric(epc, batch, 'fil_mr', mr)
			#self.update_metric(epc, batch, 'fil_mrr', fil_mrr)
			self.save_model(epc, batch_num, 'fil_mrr', mrr)
			self.update_metric(epc, batch_num, 'fil_hits1', hits1)
			# self.save_model(epc, batch_num, 'fil_hits1', fil_hits1)
			# self.update_metric(epc, batch, 'fil_hits3', fil_hits3)
			self.update_metric(epc, batch_num, 'fil_hits10', hits10)
			# self.save_model(epc, batch_num, 'fil_hits10', fil_hits10)

		model.train()

	def update_metric(self, epc, batch, name, score):
		self.history_value[name].append(score)
		if ( name not in ['fil_mr', 'raw_mr'] and score > self.best_metric[name]) or ( name in ['fil_mr', 'raw_mr'] and score < self.best_metric[name]):
			self.best_metric[name] = score
			self.best_epoch[name] = epc
			self.best_batch[name] = batch
			if name in ['fil_mr', 'raw_mr']:
				print('! Metric {0} Updated as: {1:.2f}'.format(name, score))
			else:
				print('! Metric {0} Updated as: {1:.2f}'.format(name, score*100))
			return True
		else:
			return False

	def save_model(self, epc, batch, metric, metric_val):
		# print(epc, batch, metric)
		save_path = self.param_path_template.format(epc, batch, metric)
		# print(save_path)
		last_path = self.param_path_template.format(self.best_epoch[metric], self.best_batch[metric], metric)

		if self.update_metric(epc, batch, metric, metric_val):
			if os.path.exists(last_path) and save_path != last_path and epc >= self.best_epoch[metric]:
				os.remove(last_path)
				print('Last parameters {} deleted'.format(last_path))
			
			#torch.save(self.model.state_dict(), save_path)
			torch.save({
				'model': self.model.module.state_dict(), 
				'optimizer': self.optimizer.state_dict(),
			}, save_path)

			print('Parameters saved into ', save_path)


	def debug_signal_handler(self, signal, frame):
		pdb.set_trace()
	
	def log_best(self):
		print('Best Epoch {0} Best Batch {0} micro_f1 {1}'.format(self.best_epoch, self.best_batch, self.best_metric))
			