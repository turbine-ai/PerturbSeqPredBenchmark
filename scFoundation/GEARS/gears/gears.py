from copy import deepcopy
import argparse
from multiprocessing.util import LOGGER_NAME
from time import time
import sys, os
import pickle
import json
import logging
from pathlib import Path

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from .model import GEARS_Model
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis, compute_synergy_loss
from .utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params, \
                  compute_perturbation_metrics, prep_bulk_predict_artifacts

# # to import from scGPT dir
# cur_dir = Path(__file__).resolve()
# sys.path.append(str(cur_dir.parents[3]))

# from scgpt.utils.util import compute_perturbation_metrics, prep_bulk_predict_artifacts, prep_bulk_predict_artifacts

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


class GEARS:
    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',
                 pred_scalar = False,
                 gi_predict = False):
        
        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gi_go = pert_data.gi_go
        self.gi_predict = gi_predict
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(
                self.adata.X[
                    (self.adata.obs.condition == 'ctrl').values], axis = 0)
                    ).reshape(-1,).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        if gi_predict:
            self.dict_filter = None
        else:
            self.dict_filter = {pert_full_id2pert[i]: j for i,j in self.adata.uns['non_zeros_gene_idx'].items() if i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p:gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list}
        
    def tunable_parameters(self):
        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False, 
                         cell_fitness_pred = False,
                         go_path = None,
                         #pert2gene = None,
                         model_type=None,
                         bin_set=None,
                         load_path=None,
                         finetune_method=None,
                         accumulation_steps=1,
                         mode='v1',
                         highres=0
                        ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'cell_fitness_pred': cell_fitness_pred,
                       #'pert2gene': self.pert2gene
                        'model_type': model_type,
                        'bin_set': bin_set,
                        'load_path': load_path,
                        'finetune_method': finetune_method,
                        'accumulation_steps': accumulation_steps,
                        'mode':mode,
                        'highres':highres
                      }
        print('Use accumulation steps:',accumulation_steps)
        print('Use mode:',mode)
        print('Use higres:',highres)

        # pretty print the config
        LOGGER.info('Model Configuration:')
        for k, v in self.config.items():
            LOGGER.info(f'{k}: {v}')
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type = 'co-express', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_co_express_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions)
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            print('No G_go')
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type = 'go', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_go_graph, gene_list = self.pert_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions, gi_go = self.gi_go, dataset = go_path)
            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " not in the perturbation graph. Please select from PertNet.gene_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata, self.pert_list, self.device)
            loader = DataLoader(cg, 5, shuffle = False)
            # batch = next(iter(loader))
            # batch.to(self.device)
            
            predall=[]
            for step, batch in enumerate(loader):
                batch.to(self.device)
                with torch.no_grad():
                    if self.config['uncertainty']:
                        p, unc = self.best_model(batch)
                        results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                        results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                    else:
                        p = self.best_model(batch)
                    predall.append(p.detach().cpu().numpy())
            preadall = np.concatenate(predall,axis=0)
            # results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
            results_pred['_'.join(pert)] = np.mean(preadall, axis = 0)
            
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        ## given a gene pair, return (1) transcriptome of A,B,A+B and (2) GI scores. 
        ## if uncertainty mode is on, also return uncertainty score.
        
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file = None):
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        
        de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        pred = self.predict([query.split('+')])['_'.join(query.split('+'))][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values
        
        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    
    def train(self, epochs = 20, result_dir='./results',
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        test_loader = self.dataloader['test_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)

        if self.config['finetune_method'] == 'frozen':
            for name, p in self.model.named_parameters():
                if "singlecell_model" in name:
                    p.requires_grad = False
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        elif self.config['finetune_method'] == 'finetune_lr_1':
            ignored_params = list(map(id, self.model.singlecell_model.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

            optimizer = optim.Adam([
                                    {'params': base_params, 'lr': lr},
                                    {'params': self.model.singlecell_model.parameters(), 'lr': lr*1e-1},
                                    ], weight_decay = weight_decay)        
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        max_val = -np.inf
        # print_sys('Start Training...')
        
        cnt_train_batch = len(train_loader)
        cnt_val_batch = len(val_loader)
        LOGGER.info(f"Number of training batches: {cnt_train_batch}")
        LOGGER.info(f"Number of validation batches: {cnt_val_batch}")
        LOGGER.info(f"Number of test batches: {len(test_loader)}")
        LOGGER.info('Start Training...')

        for epoch in range(epochs):
            self.model.train()
            if self.config['finetune_method'] == 'frozen':
                self.model.singlecell_model.eval()

            epoch_step = 0
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'])
                    
                # loss = loss / self.config['accumulation_steps']
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

                if (((step+1)%self.config['accumulation_steps'])==0) or (step+1==len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = f"Epoch {epoch + 1} | Step {step + 1} / {cnt_train_batch:,} | Train Loss: {loss.item():.4f}" 
                    # print_sys(log.format(epoch + 1, step + 1, loss.item()))
                    LOGGER.info(log)

                epoch_step += 1

                if epoch_step == 10:
                    break

            start_time = time() 
            scheduler.step()

            LOGGER.info("Start evaluating...")
            val_res = evaluate(val_loader, self.model, self.config['uncertainty'], self.device)
            val_metrics = compute_perturbation_metrics(
                results=val_res,
                ctrl_adata=self.ctrl_adata,
            )
            LOGGER.info(f"Validation metrics: {val_metrics}")
            # bulk the results
            val_res = prep_bulk_predict_artifacts(val_res)
            np.savez(f'{result_dir}/val_res_e{epoch}.npz', **val_res)

            LOGGER.info("Evaluating time: {:.2f}s".format(time() - start_time))

            if val_metrics['pearson_delta'] > max_val:
                max_val = val_metrics['pearson_delta']
                best_model = deepcopy(self.model)
                self.best_model = best_model
                self.save_model(result_dir)

        # eval the best model on the test set
        LOGGER.info("Evaluating the best model on the test set...")
        test_res = evaluate(test_loader, self.best_model, self.config['uncertainty'], self.device)
        test_metrics = compute_perturbation_metrics(
            results=test_res,
            ctrl_adata=self.ctrl_adata,
        )
        # turn test_metrics values to float
        test_metrics = {k: float(v) for k, v in test_metrics.items()}
        with open(f'{result_dir}/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)
        LOGGER.info(f"Test metrics: {test_metrics}")
        # bulk the results
        test_res = prep_bulk_predict_artifacts(test_res)
        np.savez(f'{result_dir}/test_res_best_model.npz', **test_res)

            # Print epoch performance
            # log = "Epoch {}: Train Overall MSE: {:.4f} " \
            #       "Validation Overall MSE: {:.4f}. "
            # # print_sys(log.format(epoch + 1, train_metrics['mse'], 
            #                  val_metrics['mse']))
            # LOGGER.info(log.format(epoch + 1, train_metrics['mse'], val_metrics['mse']))
            # print all val_metrics
            # for k, v in val_metrics.items():
            #     LOGGER.info(f"Validation {k}: {v:.4f}")
            
            # Print epoch performance for DE genes
            # log = "Train Top 20 DE MSE: {:.4f} " \
            #       "Validation Top 20 DE MSE: {:.4f}. "
            # print_sys(log.format(train_metrics['mse_de'],
            #                  val_metrics['mse_de']))
            # LOGGER.info(log.format(train_metrics['mse_de'], val_metrics['mse_de']))
            
            # if self.wandb:
            #     metrics = ['mse', 'pearson']
            #     for m in metrics:
            #         self.wandb.log({'train_' + m: train_metrics[m],
            #                    'val_'+m: val_metrics[m],
            #                    'train_de_' + m: train_metrics[m + '_de'],
            #                    'val_de_'+m: val_metrics[m + '_de']})
               
            # if val_metrics['mse_de'] < min_val:
            #     min_val = val_metrics['mse_de']
            #     best_model = deepcopy(self.model)

            #     print_sys("Best epoch:{} mse_de:{}!".format(epoch + 1, min_val))
            #     LOGGER.info("Best epoch:{} mse_de:{}!".format(epoch + 1, min_val))
            #     self.best_model = best_model
            #     self.save_model(result_dir)

        # print_sys("Done!")
        # self.best_model = best_model

        # if 'test_loader' not in self.dataloader:
        #     print_sys('Done! No test dataloader detected.')
        #     return
            
        # # Model testing
        # test_loader = self.dataloader['test_loader']
        # print_sys("Start Testing...")
        # test_res = evaluate(test_loader, self.best_model, self.config['uncertainty'], self.device)   
        # test_metrics, test_pert_res = compute_metrics(test_res)    
        # log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        # print_sys(log.format(test_metrics['mse_de']))
        
        # if self.wandb:
        #     metrics = ['mse', 'pearson']
        #     for m in metrics:
        #         self.wandb.log({'test_' + m: test_metrics[m],
        #                    'test_de_'+m: test_metrics[m + '_de']                     
        #                   })
                
        # out = deeper_analysis(self.adata, test_res)
        # out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        # metrics = ['pearson_delta']
        # metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout', 'frac_sigma_below_1_non_dropout', 'mse_top20_de_non_dropout']
        
        # if self.wandb:
        #     for m in metrics:
        #         self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

        #     for m in metrics_non_dropout:
        #         self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        # if self.split == 'simulation':
        #     print_sys("Start doing subgroup analysis for simulation split...")
        #     subgroup = self.subgroup
        #     subgroup_analysis = {}
        #     for name in subgroup['test_subgroup'].keys():
        #         subgroup_analysis[name] = {}
        #         for m in list(list(test_pert_res.values())[0].keys()):
        #             subgroup_analysis[name][m] = []

        #     for name, pert_list in subgroup['test_subgroup'].items():
        #         for pert in pert_list:
        #             for m, res in test_pert_res[pert].items():
        #                 subgroup_analysis[name][m].append(res)

        #     for name, result in subgroup_analysis.items():
        #         for m in result.keys():
        #             subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
        #             if self.wandb:
        #                 self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

        #             print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

        #     ## deeper analysis
        #     subgroup_analysis = {}
        #     for name in subgroup['test_subgroup'].keys():
        #         subgroup_analysis[name] = {}
        #         for m in metrics:
        #             subgroup_analysis[name][m] = []

        #         for m in metrics_non_dropout:
        #             subgroup_analysis[name][m] = []

        #     for name, pert_list in subgroup['test_subgroup'].items():
        #         for pert in pert_list:
        #             for m in metrics:
        #                 subgroup_analysis[name][m].append(out[pert][m])

        #             for m in metrics_non_dropout:
        #                 subgroup_analysis[name][m].append(out_non_dropout[pert][m])

        #     for name, result in subgroup_analysis.items():
        #         for m in result.keys():
        #             subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
        #             if self.wandb:
        #                 self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

        #             print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        # print_sys('Done!')


