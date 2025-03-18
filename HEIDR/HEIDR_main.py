import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict
from torch_geometric.data import Data
from torch.utils.data.dataloader import DataLoader
from data_loader_new import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace, pad_batch_v2_val

import sys
sys.path.append("..")
# from COGNet_ablation import COGNet_wo_copy, COGNet_wo_visit_score, COGNet_wo_graph, COGNet_wo_diag, COGNet_wo_proc
from HEIDR_model import HEIDR, policy_network
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result
from recommend_heidr import eval, test

torch.manual_seed(1203)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = '' 
past_name = 'past'
resume_path = '' 


# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
parser.add_argument('--topk', type=int, default=1, help='hyperparameter top-k')
parser.add_argument('--gumbel_tau', type=float, default=0.6, help='hyperparameter gumbel_tau')
parser.add_argument('--att_tau', type=int, default=20, help='hyperparameter att_tau')
parser.add_argument('--final_top_idx_data_type_train', type=str, default='', help="final_top_idx_data_type_train path")
parser.add_argument('--final_top_idx_data_type_eval', type=str, default='', help="final_top_idx_data_type_eval path")
parser.add_argument('--final_top_idx_data_type_test', type=str, default='', help="final_top_idx_data_type_test path")
parser.add_argument('--ehr_graph', type=str, default='../data/weighted_confidence_directed_ehr_graph.pkl', help="ehr_graph path")

args = parser.parse_args()
model_name = args.model_name
resume_path = args.resume_path
ehr_graph = args.ehr_graph

print("\n---setting-----")
print("model_name: ",args.model_name)
print("topk: ", args.topk, "gumbel_tau: ", args.gumbel_tau, "att_tau: ", args.att_tau)
print("ehr_graph: ", args.ehr_graph)

os.makedirs(os.path.join("HEIDR/saved", model_name, past_name), exist_ok=True)

def data_visits(data_train):
    # data_train_visits = sum(data_train, [])
    
    data_train_visits = []
    count = 0
    for index, patient in enumerate(data_train):
        for idx, visit in enumerate(patient):
            if idx ==0 : pass  
            else:
                seq_input = patient[:idx+1]
                count += len([seq_input[-1]])
                data_train_visits += [seq_input[-1]]
    print("count: ", count)
    print("data_visits: ", len(data_train_visits))
    
    # visit_med_list_train = []
    # for i in data_train_visits:
    #     visit_med_list_train.append(i[2])
    # print("visit_med_list: ", len(visit_med_list_train))
    return data_train_visits

def main(args):
    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    ehr_adj_path = args.ehr_graph # 'data/weighted_confidence_directed_ehr_graph.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    
    ## pretrain_top_embedding 
    final_top_idx_data_type_train_path = args.final_top_idx_data_type_train
    final_top_idx_data_type_eval_path = args.final_top_idx_data_type_eval
    final_top_idx_data_type_test_path = args.final_top_idx_data_type_test
    final_top_idx_data_type_train = dill.load(open(final_top_idx_data_type_train_path, 'rb'))
    final_top_idx_data_type_eval = dill.load(open(final_top_idx_data_type_eval_path, 'rb'))
    final_top_idx_data_type_test = dill.load(open(final_top_idx_data_type_test_path, 'rb'))
        
    device = torch.device('cuda')
    print(device)

    data = dill.load(open(data_path, 'rb'))

    data = [x for x in data if len(x) >= 2] ## Input only for hospitalizations that occurred two or more times
    voc = dill.load(open(voc_path, 'rb'))
    weighted_confidence_directed_ehr_graph = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications

    
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    print("data_train: ", len(data_train))
    data_train_visits = data_visits(data_train)
    print("data_train_visits: ", len(data_train_visits))

    print("data_eval: ", len(data_eval))
    data_eval_visits = data_visits(data_eval)
    # data_eval_visits = sum(data_eval, [])
    print("data_eval_visits: ", len(data_eval_visits))
    
    print("data_test: ", len(data_test))
    # data_test_visits = sum(data_test, [])
    data_test_visits = data_visits(data_test)
    print("data_test_visits: ", len(data_test_visits))

    top_visits_train = [[data_train_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_train)] 
    top_visits_eval = [[data_eval_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_eval)] 
    top_visits_test = [[data_test_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_test)] 
    print("top_visits_train: ", len(top_visits_train) , "\ntop_visits_eval: ", len(top_visits_eval), "\ntop_visits_test: ", len(top_visits_test))
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    
    ## Prepare EHR data
    edge_index = torch.tensor(np.argwhere(weighted_confidence_directed_ehr_graph != 0).T, dtype=torch.long).to(device)
    print("ehr edge_index: ", edge_index.shape)
    edge_weight = torch.tensor(weighted_confidence_directed_ehr_graph[edge_index[0].cpu(), edge_index[1].cpu()], dtype=torch.float).to(device)
    print("ehr edge_weight: ", edge_weight.shape)
    x = torch.eye(weighted_confidence_directed_ehr_graph.shape[0], device=device)  # The features of each node are one-hot encoded
    ehr_adj = Data(x=x, edge_index=edge_index, edge_weight=edge_weight).to(device)
    print("ehr_data: ", ehr_adj)
    
    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = HEIDR(voc_size, ehr_adj, ddi_adj, ddi_mask_H, topk=args.topk, gumbel_tau=args.gumbel_tau, att_tau=args.att_tau, emb_dim=args.emb_dim, device=device)
    
    if args.Test:
        ddi_rate_list_t, ja_list_t, prauc_list_t, avg_p_list_t, avg_r_list_t, avg_f1_list_t, avg_med_list_t = [],[],[],[],[],[],[]
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        all_people = []
        all_score = []
        people_length = []
        predicted_med = []
        patient_count = 0
        for step, input in enumerate(data_test):
            step_l = []
            for idx, adm in enumerate(input):
                if idx == 0: 
                    pass             
                else:
                    
                    adm_list = []
                    seq_input = input[:idx+1]
                    seq_input = seq_input[:-1] + top_visits_test[patient_count][:args.topk]  + [seq_input[-1]]
           
                    adm_list.append(seq_input)
                    test_dataset = mimic_data(adm_list)
                    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)

                    smm_record, ja, prauc, precision, recall, f1, med_num, gumbel_pick_index, cross_visit_scores_numpy = test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, ehr_adj, args)
                    
                    patient_count += 1 
                    
                    data_num = len(ja)
                    final_length = int(data_num)
                    idx_list = list(range(data_num))
                    random.shuffle(idx_list)
                    idx_list = idx_list[:final_length]
                    avg_ja = np.mean([ja[i] for i in idx_list])
                    avg_prauc = np.mean([prauc[i] for i in idx_list])
                    avg_precision = np.mean([precision[i] for i in idx_list])
                    avg_recall = np.mean([recall[i] for i in idx_list])
                    avg_f1 = np.mean([f1[i] for i in idx_list])
                    avg_med = np.mean([med_num[i] for i in idx_list])
                    cur_smm_record = [smm_record[i] for i in idx_list]
                    ddi_rate = ddi_rate_score(cur_smm_record, path='../data/ddi_A_final.pkl')
                    isnan_list = [np.isnan(i) for i in [ddi_rate,avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med]]
                    if True not in isnan_list:
                        result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
                        llprint('\nJaccard: {}, PRAUC: {}, AVG_F1: {}, DDI Rate: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_MED: {}\n'.format(
                                avg_ja, avg_prauc, avg_f1, ddi_rate, avg_precision, avg_recall, avg_med))
                    predicted_med.append(smm_record)
                    all_people.append(gumbel_pick_index)
                    all_score.append(cross_visit_scores_numpy)
                    people_length.append(len(input))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        dill.dump(all_people, open(os.path.join('HEIDR/saved', model_name, 'gumbel_pick.pkl'), 'wb'))
        dill.dump(all_score,open(os.path.join('HEIDR/saved', model_name, 'all_score.pkl'), 'wb'))
        dill.dump(people_length, open(os.path.join('HEIDR/saved', model_name, 'people_length.pkl'), 'wb'))
        dill.dump(predicted_med, open(os.path.join('HEIDR/saved', model_name, 'predicted_med.pkl'), 'wb'))
        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 200
    print("EPOCH: ", EPOCH)
    temp_min = 0.5
    ANNEAL_RATE = 0.000003
    temp_max = 25
    ANNEAL_RATE2 = 0.000003

    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))
        loss_record = []
        gumble_list = []
        patient_count = 0
        for step, input in enumerate(data_train):
            patient_list = []
            patient_list.append(len(input))
            for idx, adm in enumerate(input):
                if idx ==0 : pass  
                else:
                    adm_list = []
                    seq_input = input[:idx+1]
                    seq_input = seq_input[:-1] + top_visits_train[patient_count][:args.topk] + [seq_input[-1]]
                    adm_list.append(seq_input)
                    train_dataset = mimic_data(adm_list)
                    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)
                    patient_count += 1 
                    model.train()
                    for ind, data in enumerate(train_dataloader):
                        diseases, procedures, medications, seq_length, \
                        d_length_matrix, p_length_matrix, m_length_matrix, \
                            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                                dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                                    dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

                        diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
    
                        procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
                        dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
                        
                        stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
                        dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
                        stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
                        medications = medications.to(device)
                        m_mask_matrix = m_mask_matrix.to(device)
                        d_mask_matrix = d_mask_matrix.to(device)
                        p_mask_matrix = p_mask_matrix.to(device)
                        dec_disease_mask = dec_disease_mask.to(device)
                        stay_disease_mask = stay_disease_mask.to(device)
                        dec_proc_mask = dec_proc_mask.to(device)
                        stay_proc_mask = stay_proc_mask.to(device)
                        output_logits, count, gumbel_pick_index, cross_visit_scores_numpy = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, ehr_adj)
                        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
                        patient_list.append(gumbel_pick_index)
                        loss = F.nll_loss(predictions, labels.long())
                        optimizer.zero_grad()
                        loss_record.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        llprint('\rencoder_gumbel_training step: {} / {}'.format(step, len(data_train)))
            gumble_list.append(patient_list)
            
            # gumbel tau schedule 
            if step % 100 == 0:
                #model.gumbel_tau = np.maximum(model.gumbel_tau * np.exp(-ANNEAL_RATE * step), temp_min)
                #model.att_tau = np.minimum(model.att_tau * np.exp(ANNEAL_RATE2 * step), temp_max)
                print(" New Gumbel Temperature: {}".format(model.gumbel_tau))
                print(" New Attention Temperature: {}".format(model.att_tau))
                
         
        dill.dump(gumble_list, open(os.path.join('HEIDR/saved', model_name, past_name,'{}_epoch_train_gumbel_pick.pkl'.format(epoch)), 'wb'))

        print ()
        tic2 = time.time()
        ddi_rate_list, ja_list, prauc_list, avg_p_list, avg_r_list, avg_f1_list, avg_med_list = [],[],[],[],[],[],[]
        all_people = []
        patient_count = 0
        for step, input in enumerate(data_eval):
            #step_l = []
            for idx, adm in enumerate(input):    
                if idx == 0:
                    pass
                else:  
                    adm_list = []
                    seq_input = input[:idx+1]
                    seq_input = seq_input[:-1] + top_visits_eval[patient_count][:args.topk] + [seq_input[-1]]
                    adm_list.append(seq_input)
                
                    eval_dataset = mimic_data(adm_list)
                    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_val, shuffle=True, pin_memory=True)
                    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, gumbel_pick_index, cross_visit_scores_numpy = eval(model, eval_dataloader, voc_size, device, TOKENS, ehr_adj, args)
                    patient_count += 1 

                    print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

                    history['ja'].append(ja)
                    history['ddi_rate'].append(ddi_rate)
                    history['avg_p'].append(avg_p)
                    history['avg_r'].append(avg_r)
                    history['avg_f1'].append(avg_f1)
                    history['prauc'].append(prauc)
                    history['med'].append(avg_med)
                    ddi_rate_list.append(ddi_rate)
                    ja_list.append(ja)
                    prauc_list.append(prauc)
                    avg_p_list.append(avg_p)
                    avg_r_list.append(avg_r)
                    avg_f1_list.append(avg_f1)
                    avg_med_list.append(avg_med)
        
        llprint('\n\rTrain--Epoch: %d, loss: %d' % (epoch, np.mean(loss_record)))
        
        torch.save(model.state_dict(), open(os.path.join('HEIDR/saved', args.model_name, \
            'Epoch_{}_top_{}_JA_{:.4}_DDI_{:.4}_LOSS_{}.model'.format(epoch, args.topk, np.mean(ja_list), np.mean(ddi_rate_list), np.mean(loss_record))), 'wb'))
        #dill.dump(all_people, open(os.path.join('saved', model_name, jaccard_name,'{}epoch_eval_jaccard_past_concat_1.pkl'.format(epoch)), 'wb'))

        if best_ja < np.mean(ja_list):
            best_epoch = epoch
            best_ja = np.mean(ja_list)
        
        print ('best_jaccard: {}'.format(best_ja))
        print ('best_epoch: {}'.format(best_epoch))
        print('JA_{:.4}_DDI_{:.4}_PRAUC_{:.4}_F1_{:.4}'.format(np.mean(ja_list), np.mean(ddi_rate_list), np.mean(prauc_list), np.mean(avg_f1_list)))
        
        dill.dump(history, open(os.path.join('HEIDR/saved', args.model_name, 'history_{}_{}.pkl'.format(args.model_name, args.topk)), 'wb'))
    
if __name__ == '__main__':
    main(args)
    print(model_name)