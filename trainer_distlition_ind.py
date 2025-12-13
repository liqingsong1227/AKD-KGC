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
	def __init__(self,train_data, test_data, data_loader_list, model,  optimizer, scheduler, device, hyperparams ):
		model.to(device)

		self.train_data = train_data
		self.test_data = test_data
		self.train_data_loader = data_loader_list[0]
		self.valid_data_loader = data_loader_list[1]
		self.test_data_loader = data_loader_list[2]
		
		# self.rel_embddding = data.relation_embedding
		teacher_model = copy.deepcopy(model.belman_model).to(device)
		self.teacher_model =DDP(teacher_model,device_ids=[device], find_unused_parameters=True)

		self.model = DDP(model,device_ids=[device], find_unused_parameters=True)
		self.scheduler = scheduler

		self.num_negative = hyperparams['num_negative']
		self.strict_negative = hyperparams['strict_negative']


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
		
		
		teacher_load_path = hyperparams['teacher_load_path']
		if teacher_load_path != None:
			if not (teacher_load_path.startswith(save_folder) or teacher_load_path.startswith('./saveparams/')):
				teacher_load_path = save_folder + teacher_load_path

			if os.path.exists(teacher_load_path):
				
				try:
					checkpoint = torch.load(teacher_load_path)
					teacher_model.load_state_dict(checkpoint['model'], strict=False)
					model.belman_model.load_state_dict(checkpoint['model'], strict=False)
					# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					
					print('Teacher Model  Parameters loaded from {0}.'.format(teacher_load_path))
					del checkpoint
					torch.cuda.empty_cache()
				except:
					model.load_state_dict(torch.load(teacher_load_path), strict=False)
					print('Parameters loaded from {0}.'.format(teacher_load_path))
			else:
				print('Parameters {0} Not Found'.format(teacher_load_path))


		
		if load_path != None:
			if not (load_path.startswith(save_folder) or load_path.startswith('./saveparams/')):
				load_path = save_folder + load_path

			if os.path.exists(load_path):
				
				try:
					checkpoint = torch.load(load_path)
					model.load_state_dict(checkpoint['model'], strict=False)
					optimizer.load_state_dict(checkpoint['optimizer'])
					print('Model & Optimizer  Parameters loaded from {0}.'.format(load_path))
					del checkpoint
					torch.cuda.empty_cache()
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
		teacher_model = self.teacher_model
		optimizer = self.optimizer
		scheduler = self.scheduler
		device = self.device
		hyperparams = self.hyperparams
		epoch = hyperparams['epoch'] 
		temperature = hyperparams['temperature']
		kl_temperature = hyperparams['kl_temperature']
		data = self.train_data
		data_loader = self.train_data_loader
		groundtruth = data.get_groundtruth()
		sigmoid = torch.nn.Sigmoid()



		nbf_model.train()
		
		modes = ["link_prediction_t", "link_prediction_h"]

		for epc in range(self.load_epoch+1, epoch):

			data_loader.sampler.set_epoch(epc)
			for i_b, batch_ in enumerate(data_loader):
				'''if i_b == 1:
					break'''
				
				train_data, batch = batch_
				real_triples = [(int(batch[i,0]), int(batch[i,1]), int(batch[i,2])) for i in range(batch.shape[0])]
				
				real_triple_tensor = torch.LongTensor(real_triples)
				# print(real_triple_tensor)
				pos_h_index, pos_t_index, pos_r_index = real_triple_tensor[:,0], real_triple_tensor[:,1], real_triple_tensor[:,2]
				h_index = pos_h_index.unsqueeze(-1).repeat(1,len(real_triples)).to(device)
				r_index = pos_r_index.unsqueeze(-1).repeat(1,len(real_triples)).to(device)
				t_index = pos_t_index.unsqueeze(-1).repeat(1,len(real_triples)).to(device)
				
				
				for mode in modes:
					hr_inputs, hr_positions = data.batch_tokenize(real_triples, mode=mode)
					hr_inputs.to(device)
					t_inputs, t_positions = data.batch_tokenize_target(real_triples, mode=mode)
					t_inputs.to(device)
					labels_idx_list = []
					labels = torch.zeros((len(real_triples), len(real_triples))).to(device)
					if mode == "link_prediction_t":
						t_targets = [t[1] for t in real_triples]
						t_index_t = t_index.t()
						batch = torch.stack([h_index, t_index_t, r_index], dim=-1)
						for i, triple in enumerate(real_triples):
							h , t, r = triple
							t_expects = set(groundtruth['train']['tail'][(r, h)])
							t_label_idx = [i_t for i_t, target in enumerate(t_targets) if target in t_expects]
							labels[i, t_label_idx] = 1
							labels_idx_list.append(t_label_idx)
					elif mode == "link_prediction_h":
						h_targets = [t[0] for t in real_triples]
						h_index_t = h_index.t()
						r_index_reverse = r_index + data.relation_num
						batch = torch.stack([t_index, h_index_t, r_index_reverse], dim=-1)
						for i, triple in enumerate(real_triples):
							h , t, r = triple
							h_expects = set(groundtruth['train']['head'][(r, t)])
							h_label_idx = [i_t for i_t, target in enumerate(h_targets) if target in h_expects]
							labels[i, h_label_idx] = 1
							labels_idx_list.append(h_label_idx)

					batch = batch.to(device)
					'''print(mode)
					print(batch)'''
					train_data.to(device)

					pred = nbf_model(train_data, batch, hr_inputs, hr_positions, t_inputs, t_positions, mode) #   # 
					
					loss = F.binary_cross_entropy_with_logits(pred, labels, reduction="none")
					with torch.no_grad():
						nbf_score = teacher_model(train_data, batch)
						teacher_CE = F.binary_cross_entropy_with_logits(nbf_score, labels, reduction="mean").item()
						student_CE = F.binary_cross_entropy_with_logits(pred, labels, reduction="mean").item()
						alpha = math.exp(teacher_CE / temperature) / (math.exp(student_CE / temperature) + math.exp(teacher_CE / temperature))

					
					p_softmax = (pred / kl_temperature).softmax(dim=-1) + 1e-9
					q_softmax = (nbf_score / kl_temperature).softmax(dim=-1) + 1e-9
					kl_1 = F.kl_div(q_softmax.log(), p_softmax, reduction='batchmean')
					kl_2 = F.kl_div(p_softmax.log(), q_softmax, reduction='batchmean')
					KL_loss = (kl_1 + kl_2) / 2
					# preds = sigmoid(pred)

					CE_loss = 0
					for i in range(len(real_triples)):
						pos_idx = sorted(labels_idx_list[i])
						pos_set = set(pos_idx)
						neg_idx = [_ for _ in range(labels.shape[1]) if not _ in pos_set]
						loss_one = loss[i]
						loss_pos = loss_one[pos_idx].mean()
						neg_selfadv_weight = pred[i, neg_idx]
						neg_weights = neg_selfadv_weight.softmax(dim=-1)
						neg_loss = (loss_one[neg_idx] * neg_weights).sum() 
						CE_loss += (loss_pos + neg_loss) / len(real_triples)


					final_loss = alpha * CE_loss + (1-alpha) * KL_loss
					# final_loss = CE_loss
					final_loss.backward()
					optimizer.step()
					if scheduler != None:
						scheduler.step()
					optimizer.zero_grad()
					if (i_b+1) % 100 == 0 and int(os.environ["LOCAL_RANK"])==0:
						print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()),' Train Epoch {}, Batch {}, Total loss: {}, CE Loss: {}, KL Loss: {}, Teacher_CE: {}, Student_CE: {}, alpha: {}, Mode: {}'.format(epc, i_b+1,  final_loss.item(),
																																															   CE_loss, KL_loss, teacher_CE, student_CE, alpha, mode))

				if (i_b+1) % 600  == 0 and int(os.environ["LOCAL_RANK"])==0:
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
		n_ent = data.entity_num
		ent_target_encoded = torch.zeros((n_ent, model.module.lmke_model.hidden_size)).to(device)
		entity_list = list(range(n_ent))
		with torch.no_grad():
			# calc entity target embeddings
			random_map = [ i for i in range(n_ent)]
			batch_list = [ random_map[i:i+batch_size_target] for i in range(0, n_ent, batch_size_target)] 
			for batch in batch_list:
				batch_targets = [ entity_list[_] for _ in batch]
				target_inputs, target_positions = data.batch_tokenize_target(targets=batch_targets)
				target_inputs.to(device)
				target_encodes = model.module.lmke_model.encode_target(target_inputs, target_positions, mode='link_prediction_t')
				ent_target_encoded[batch] = target_encodes
		with torch.no_grad():
			modes = ["link_prediction_t", "link_prediction_h"]
			for mode in  modes:
				count_triples = 0
				score = 0
				for i_b, batch in enumerate(data_loader):
					'''if i_b == 5:
						break'''
					batch = batch.to(device)
					test_data.to(device)
					t_batch, h_batch = tasks.all_negative(test_data, batch)
					triples = [(int(batch[i,0]), int(batch[i,1]), int(batch[i,2])) for i in range(batch.shape[0])]

					if mode == "link_prediction_h":
						target = 'head'	
						hr_inputs, hr_positions = data.batch_tokenize(triples, mode=mode)
						hr_inputs.to(device)
						language_preds = model.module.lmke_model.forward_test(test_data, h_batch, hr_inputs, hr_positions, ent_target_encoded, mode)
						preds =  model.module.forward_test( test_data, h_batch, language_preds)
					elif mode == "link_prediction_t":
						target = 'tail'	
						hr_inputs, hr_positions = data.batch_tokenize(triples, mode=mode)
						hr_inputs.to(device)
						language_preds = model.module.lmke_model.forward_test(test_data, t_batch, hr_inputs, hr_positions, ent_target_encoded, mode)
						preds =  model.module.forward_test( test_data, t_batch, language_preds) #  # 
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
			self.update_metric(epc, batch_num, 'fil_mrr', mrr)
			self.update_metric(epc, batch_num, 'fil_hits1', hits1)
			# self.save_model(epc, batch_num, 'fil_hits1', fil_hits1)
			# self.update_metric(epc, batch, 'fil_hits3', fil_hits3)
			self.save_model(epc, batch_num, 'fil_hits10', hits10)
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
			