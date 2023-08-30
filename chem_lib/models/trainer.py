import random
import os
import numpy as np
import joblib
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc, average_precision, precision_recall
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

from sklearn.metrics import auc as auc_fun

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()

        self.args = args

        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
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
        log_names += ['AP-' + str(t) for t in args.test_tasks]
        log_names += ['AP-Avg', 'AP-Mid','AP-Best']
        log_names += ['PRE-' + str(t) for t in args.test_tasks]
        log_names += ['PRE-Avg', 'PRE-Mid','PRE-Best']
        log_names += ['REC-' + str(t) for t in args.test_tasks]
        log_names += ['REC-Avg', 'REC-Mid','REC-Best']
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
        self.best_auc = 0 
        self.best_ap = 0
        self.best_acc_ci95 = 0 
        self.best_pre = 0
        self.best_recall = 0
        self.best_prauc = 0
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
            s_logits, q_logits, adj, node_emb, s_logits_pn, q_logits_pn = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb, 's_logits_pn': s_logits_pn, 'q_logits_pn':q_logits_pn}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        fenc_node = lambda x: x[0]== 'mol_encoder' and 'x_embedding'  in x[2]
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
            # flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
            flag=lambda x: not (fenc_node(x) or fnode(x) or fedge(x))
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
                if p.requires_grad:
                    if flag(names):
                        adaptable_weights.append(p)
                        adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_test*n_query,2), batch_data['s_label'].repeat(n_query))
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
                # losses_adapt = losses_adapt + self.criterion(pred_dict['s_logits_pn'], batch_data['s_label'])
            else:
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])
                # losses_adapt = losses_adapt + self.criterion(pred_dict['q_logits_pn'], batch_data['q_label'])

        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = torch.zeros_like(losses_adapt)
        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].size(0)
            adj = pred_dict['adj'][-1]
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * n_support
                    label_edge = model.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    label_edge = model.label2edge(total_label)[:,:,-1,:-1]
                    pred_edge = adj[:,:,-1,:-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = torch.zeros_like(adj_loss_val)

            losses_adapt += self.args.reg_adj * adj_loss_val

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
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)
                
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)

                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)

                losses_eval.append(loss_eval)

            losses_eval = torch.stack(losses_eval)
            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

        return self.model.module

    def seed_torch(self, seed=21):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
        torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

    def test_step(self):
        seeds = [1234, 2022, 1987, 1998, 2345, 5678, 7890, 1024, 42, 21]
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        ap_scores = []
        acc_scores = []
        pre_scores = []
        recall_scores = []
        prauc_scores = []
        acc_ci95_scores = []
        pracc_ci95_scores = []
        pracc_scores =[]


        if self.args.tentimes:
            for seed, test_epoch in zip(seeds, range(10)):
                self.seed_torch(seed)
                auc_scores = []
                for task_id in range(len(self.test_tasks)):
                    adapt_data, eval_data = self.get_data_sample(task_id, train=False)
                    model = self.model.clone()
                    if self.update_step_test>0:
                        model.train()
                        
                        for i, batch in enumerate(adapt_data['data_loader']):
                            batch = batch.to(self.device)
                            cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                                'q_data': batch, 'q_label': None}

                            adaptable_weights = self.get_adaptable_weights(model)
                            pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                            loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

                            model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

                            if i>= self.update_step_test-1:
                                break

                    model.eval()
                    with torch.no_grad():
                        pred_eval = self.get_prediction(model, eval_data, train=False)
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

                    print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
                    if self.args.save_logs:
                        step_results['query_preds'].append(y_score.cpu().numpy())
                        step_results['query_labels'].append(y_true.cpu().numpy())
                        step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                        step_results['task_index'].append(self.test_tasks[task_id])

                mid_auc = np.median(auc_scores)
                avg_auc = np.mean(auc_scores)
                mid_prauc = np.median(prauc_scores)
                avg_prauc = np.mean(prauc_scores)
                acc_ci95 = 1.96 * np.std(np.array(auc_scores)) / np.sqrt(len(self.test_tasks))
                pracc_ci95 = 1.96 * np.std(np.array(prauc_scores)) / np.sqrt(len(self.test_tasks))
                acc_scores.append(avg_auc)
                acc_ci95_scores.append(acc_ci95)
                pracc_scores.append(avg_prauc)
                pracc_ci95_scores.append(pracc_ci95)
            avg_acc = np.mean(acc_scores)
            avg_acc_ci95 = np.mean(acc_ci95_scores)
            # self.best_auc = max(self.best_auc, avg_acc)
            if self.best_auc < avg_acc:
                self.best_auc = avg_acc
                self.best_acc_ci95 = avg_acc_ci95
            self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc, self.best_auc], verbose=False)

            print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
                ', Best_Avg_AUC: ', round(self.best_auc, 4),)
            
            if self.args.save_logs:
                self.res_logs.append(step_results)

            return self.best_auc, self.best_acc_ci95
        else:
            for task_id in range(len(self.test_tasks)):
                adapt_data, eval_data = self.get_data_sample(task_id, train=False)
                model = self.model.clone()
                if self.update_step_test>0:
                    model.train()
                    
                    for i, batch in enumerate(adapt_data['data_loader']):
                        batch = batch.to(self.device)
                        cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                            'q_data': batch, 'q_label': None}

                        adaptable_weights = self.get_adaptable_weights(model)
                        pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                        loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

                        model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

                        if i>= self.update_step_test-1:
                            break
                
                import pdb
                # pdb.set_trace()
                model.eval()
                with torch.no_grad():
                    pred_eval = self.get_prediction(model, eval_data, train=False)
                    y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                    y_true = pred_eval['labels']     ## real label
                    if self.args.eval_support:
                        y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                        y_s_true = eval_data['s_label']
                        y_score=torch.cat([y_score, y_s_score])
                        y_true=torch.cat([y_true, y_s_true])
                    auc = auroc(y_score, y_true, pos_label=1).item()
                    AP = average_precision(y_score, y_true, pos_label=1).item()
                    # precision, recall = precision_recall(y_score, y_true)
                    # print(y_true.cpu().numpy())
                    # print(y_score.cpu().numpy())
                    pr_auc = average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy())
                    # precision, recall, _ = precision_recall_curve(y_true.cpu().numpy(), y_score.cpu().numpy())
                    # # print(precision)
                    # # print(recall)
                    # pr_auc = auc_fun(precision, recall)
                    # print(pr_auc)
                    # precision = sklearn.metrics.precision_score(y_true=y_true.cpu().numpy(), y_pred=y_score.cpu().numpy(), pos_label="positive")
                    # recall = sklearn.metrics.recall_score(y_true=y_true.cpu().numpy(), y_pred=y_score.cpu().numpy(), pos_label="positive")
                    # AP = average_precision_score(y_true=y_true.cpu().numpy(), y_pred=y_score.cpu().numpy(), average='macro', pos_label=1, sample_weight=None)
                    precision = pr_auc
                    recall = pr_auc
                auc_scores.append(auc)
                ap_scores.append(AP)
                pre_scores.append(precision)
                recall_scores.append(recall)
                prauc_scores.append(pr_auc)

                print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4), ', PRAUC:', round(pr_auc, 4), 
                      ', AP:', round(AP, 4), ', Pre:', round(precision, 4), ', Recall:', round(recall, 4))
                if self.args.save_logs:
                    step_results['query_preds'].append(y_score.cpu().numpy())
                    step_results['query_labels'].append(y_true.cpu().numpy())
                    step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                    step_results['task_index'].append(self.test_tasks[task_id])

            import pdb
            # pdb.set_trace()
            mid_auc = np.median(auc_scores)
            avg_auc = np.mean(auc_scores)
            mid_prauc = np.median(prauc_scores)
            avg_prauc = np.mean(prauc_scores)

            mid_ap = np.median(ap_scores)
            avg_ap = np.mean(ap_scores)
            mid_pre = np.median(pre_scores)
            avg_pre = np.mean(pre_scores)
            mid_recall = np.median(recall_scores)
            avg_recall = np.mean(recall_scores)
            auc_ci95 = 1.96 * np.std(np.array(auc_scores)) / np.sqrt(len(self.test_tasks))

            prauc_ci95 = 1.96 * np.std(np.array(prauc_scores)) / np.sqrt(len(self.test_tasks))

            if avg_auc > self.best_auc:
                self.best_acc_ci95 = auc_ci95
            if avg_prauc > self.best_prauc:
                self.best_pracc_ci95 = prauc_ci95
            self.best_auc = max(self.best_auc, avg_auc)
            self.best_prauc = max(self.best_prauc, avg_prauc)
            self.best_ap = max(self.best_ap, avg_ap)
            self.best_pre = max(self.best_pre, avg_pre)
            self.best_recall = max(self.best_recall, avg_recall)
            # self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)
            self.logger.append([self.train_epoch] + auc_scores + [avg_auc, mid_auc,self.best_auc] 
                               + ap_scores + [avg_ap, mid_ap, self.best_ap] + pre_scores + [avg_pre, mid_pre, self.best_pre]
                               + recall_scores + [avg_recall, mid_recall, self.best_recall], verbose=False)

            print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
                ', Best_Avg_AUC: ', round(self.best_auc, 4), 'Best_Avg_AUC_ci95: ', round(self.best_acc_ci95, 4), 
                ', Best_Avg_PRAUC: ', round(self.best_prauc, 4), 'Best_Avg_PRAUC_ci95: ', round(self.best_pracc_ci95, 4), 
                ', Best_Avg_AP: ', round(self.best_ap, 4), ', Best_Avg_Pre: ', round(self.best_pre, 4), 
                ', Best_Avg_Rec: ', round(self.best_recall, 4))
            
            if self.args.save_logs:
                self.res_logs.append(step_results)

            return self.best_auc, self.best_acc_ci95

    def get_curve(self, known, novel, method=None):
        """
            known: true    
            novel: false   
        """
        tp, fp = dict(), dict()
        fpr_at_tpr95 = dict()

        known.sort()
        novel.sort()

        all = np.concatenate((known, novel))
        all.sort()

        num_k = known.shape[0]   ## real = ID  TP+FN
        num_n = novel.shape[0]   ## real = OOD FP+TN

        ## > threshold 为 ID
        if method == 'row':
            threshold = -0.5
        else:
            threshold = known[round(0.05 * num_k)]    ## 取 TPR 为 95%

        fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

        return tp, fp, fpr_at_tpr95

    ## display
    def test_display(self):
        seeds = [1234, 2022, 1987, 1998, 2345, 5678, 7890, 1024, 42, 21]
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        acc_scores = []
        acc_ci95_scores = []
        task_score = {}

        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            model = self.model.clone()
            if self.update_step_test>0:
                model.train()
                
                for i, batch in enumerate(adapt_data['data_loader']):
                    batch = batch.to(self.device)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                        'q_data': batch, 'q_label': None}

                    adaptable_weights = self.get_adaptable_weights(model)
                    pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                    loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=False)

                    model.adapt(loss_adapt, adaptable_weights=adaptable_weights)

                    if i>= self.update_step_test-1:
                        break

            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                y_true = pred_eval['labels']
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                auc = auroc(y_score,y_true,pos_label=1).item()

            task_score[task_id] = auc
            auc_scores.append(auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        auc_ci95 = 1.96 * np.std(np.array(auc_scores)) / np.sqrt(len(self.test_tasks))

        if avg_auc > self.best_auc:
            self.best_acc_ci95 = auc_ci95
        self.best_auc = max(self.best_auc, avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
            ', Best_Avg_AUC: ', round(self.best_auc, 4), 'Best_Avg_AUC_ci95: ', round(self.best_acc_ci95, 4))
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc, self.best_acc_ci95

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs, self.args.trial_path+'/logs.pkl', compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
