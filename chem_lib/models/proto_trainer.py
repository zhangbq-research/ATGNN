import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

class Proto_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Proto_Trainer, self).__init__()

        self.args = args

        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr*0.01, weight_decay=args.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=args.weight_decay)
        # lambda_epoch = lambda e: 1.0 if e < 2500 else (0.1 if e < 4000 else 0.01 if e < 4500 else (0.001))
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        self.best_prauc=0 
        self.best_acc_ci95 = 0
        self.best_pracc_ci95 = 0
        
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits}
        else:
            logits,labels,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'logits': logits, 'labels': labels, 'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query

        losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
        losses_adapt = losses_adapt + self.criterion(pred_dict['s_logits'], batch_data['s_label'])

        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = torch.zeros_like(losses_adapt)
        return losses_adapt

    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db

        for k in range(self.update_step):
            losses_eval = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                self.model.train()
                
                pred_eval = self.get_prediction(self.model, train_data, train=True)
                loss_eval = self.get_loss(self.model, train_data, pred_eval, train=True, flag = 0)

                losses_eval.append(loss_eval)

            losses_eval = torch.stack(losses_eval)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())


    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        prauc_scores=[]
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            self.model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(self.model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                y_true = pred_eval['labels']
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                auc = auroc(y_score,y_true,pos_label=1).item()
                pr_auc = average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy())

            auc_scores.append(auc)
            prauc_scores.append(pr_auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4), ', PRAUC:', round(pr_auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        mid_prauc = np.median(prauc_scores)
        avg_prauc = np.mean(prauc_scores)
        self.best_auc = max(self.best_auc,avg_auc)

        self.best_prauc = max(self.best_prauc,avg_prauc)

        auc_ci95 = 1.96 * np.std(np.array(auc_scores)) / np.sqrt(len(self.test_tasks))

        prauc_ci95 = 1.96 * np.std(np.array(prauc_scores)) / np.sqrt(len(self.test_tasks))

        if avg_auc >= self.best_auc:
            self.best_acc_ci95 = auc_ci95
            self.best_pracc_ci95 = prauc_ci95
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        # print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
        #       ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
                ', Best_Avg_AUC: ', round(self.best_auc, 4), 'Best_Avg_AUC_ci95: ', round(self.best_acc_ci95, 4), 
                ', Best_Avg_PRAUC: ', round(self.best_prauc, 4), 'Best_Avg_PRAUC_ci95: ', round(self.best_pracc_ci95, 4))
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
