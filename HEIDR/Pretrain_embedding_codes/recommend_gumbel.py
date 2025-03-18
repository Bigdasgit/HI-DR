import pickle
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

from torch.utils.data.dataloader import DataLoader
from data_loader_new import mimic_data, pad_num_replace
from beam import Beam

import sys
sys.path.append("..")
from models import Leap 
from VITA_model import VITA
from util import llprint, sequence_metric, sequence_metric_v2, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result

torch.manual_seed(1203)



def eval_recommend_batch(model, batch_data, epoch_saved_patient_embedding_eval, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
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

    batch_size = medications.size(0)
    max_visit_num = medications.size(1)
    input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, count, people,query = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)
   
    epoch_saved_patient_embedding_eval.append(query)
    print("epoch_saved_patiepoch_saved_patient_embedding_evalent_embedding: ", len(epoch_saved_patient_embedding_eval))    

    partial_input_medication = torch.full((batch_size, max_visit_num, 1), SOS_TOKEN).to(device)
    parital_logits = None
    cross_visit_scores_numpy = cross_visit_scores.cpu().detach().numpy

    for i in range(args.max_len):
        partial_input_med_num = partial_input_medication.size(2)
        partial_m_mask_matrix = torch.zeros((batch_size, max_visit_num, partial_input_med_num), device=device).float()

        parital_logits = model.decode(partial_input_medication, input_disease_embdding,  encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)
        _, next_medication = torch.topk(parital_logits[:, :, -1, :], 1, dim=-1)
        partial_input_medication = torch.cat([partial_input_medication, next_medication], dim=-1)

    return parital_logits, people, cross_visit_scores_numpy, epoch_saved_patient_embedding_eval



def test_recommend_batch(model, batch_data, epoch_saved_patient_embedding_test, device, TOKENS, ddi_adj, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
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

    batch_size = medications.size(0)
    visit_num = medications.size(1)
    
    input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory,count, gumbel_pick_index, query = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)
    cross_visit_scores_numpy = cross_visit_scores.cpu().detach().numpy()
    
    epoch_saved_patient_embedding_test.append(query)
    print("epoch_saved_patient_embedding_test: ", len(epoch_saved_patient_embedding_test))
    
    assert batch_size == 1
    # visit_num个batch
    beams = [Beam(args.beam_size, MED_PAD_TOKEN, SOS_TOKEN, END_TOKEN, ddi_adj, device) for _ in range(visit_num)]


    input_disease_embdding = input_disease_embdding.repeat_interleave(args.beam_size, dim=0)
    
    encoded_medication = encoded_medication.repeat_interleave(args.beam_size, dim=0)
    last_seq_medication = last_seq_medication.repeat_interleave(args.beam_size, dim=0)
    cross_visit_scores = cross_visit_scores.repeat_interleave(args.beam_size, dim=0)

    d_mask_matrix = d_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    p_mask_matrix = p_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    last_m_mask = last_m_mask.repeat_interleave(args.beam_size, dim=0)

    for i in range(args.max_len):
        len_dec_seq = i + 1
        
        dec_partial_inputs = torch.cat([b.get_current_state().unsqueeze(dim=1) for b in beams], dim=1)

        partial_m_mask_matrix = torch.zeros((args.beam_size, visit_num, len_dec_seq), device=device).float()
        parital_logits = model.decode(dec_partial_inputs, input_disease_embdding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)

        word_lk = parital_logits[:, :, -1, :]

        active_beam_idx_list = []  
        for beam_idx in range(visit_num):
            if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
                active_beam_idx_list.append(beam_idx)

        if not active_beam_idx_list: break

    # Return useful information
    all_hyp = []
    all_prob = []
    for beam_idx in range(visit_num):
        scores, tail_idxs = beams[beam_idx].sort_scores()   
        hyps = beams[beam_idx].get_hypothesis(tail_idxs[0])
        probs = beams[beam_idx].get_prob_list(tail_idxs[0])
        all_hyp += [hyps]  
        all_prob += [probs]

    return all_hyp, all_prob, gumbel_pick_index, cross_visit_scores_numpy, epoch_saved_patient_embedding_test


# evaluate
def eval(model, eval_dataloader, epoch_saved_patient_embedding_eval, voc_size, device, TOKENS, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0


    for idx, data in enumerate(eval_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, people, cross_visit_scores_numpy, saved_patient_embedding_eval = eval_recommend_batch(model, data, epoch_saved_patient_embedding_eval, device, TOKENS, args)

        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=False, max_len=args.max_len)
       
        y_gt = []       
        y_pred = []     
        y_pred_prob = []   
        y_pred_label = []   
        for label, prediction in zip(labels, predictions):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1   
            y_gt.append(y_gt_tmp)

            out_list, sorted_predict = sequence_output_process(prediction, [voc_size[2], voc_size[2]+1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(prediction[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        try:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                    sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rtest step: {} / {}'.format(idx, len(eval_dataloader)))
        except ValueError:  #Input contains NaN, infinity or a value too large for dtype('float32').
            pass  

    ddi_rate = ddi_rate_score(smm_record, path='data/ddi_A_final.pkl')

    llprint('\nDDI Rate: {}, Jaccard: {},  PRAUC: {}, AVG_PRC: {}, AVG_RECALL: {}, AVG_F1: {}, AVG_MED: {}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt, people ,cross_visit_scores_numpy, saved_patient_embedding_eval


# test 
def test(model, test_dataloader, epoch_saved_patient_embedding_test, diag_voc, pro_voc, med_voc, voc_size, epoch, device, TOKENS, ddi_adj, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    all_pred_list = []
    all_label_list = []

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, output_probs, gumbel_pick_index, cross_visit_scores_numpy, saved_patient_embedding_test = test_recommend_batch(model, data, epoch_saved_patient_embedding_test, device, TOKENS, ddi_adj, args)

        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        _, probs = output_flatten(medications, output_probs, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        y_gt = []       
        y_pred = []    
        y_pred_label = [] 
        y_pred_prob = [] 

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []
        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a/b
        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        for label, prediction, prob_list in zip(labels, predictions, probs):
            try:
                label_hisory += label.tolist()  ### case study

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[label] = 1 
                y_gt.append(y_gt_tmp)

                out_list = []
                out_prob_list = []
                for med, prob in zip(prediction, prob_list):
                    if med in [voc_size[2], voc_size[2]+1]:
                        break
                    out_list.append(med)
                    out_prob_list.append(prob[:-2])

                ## case study
                if label_hisory:
                    jaccard_list.append(cal_jaccard(prediction, label_hisory))
                pred_list.append(out_list)
                label_hisory_list.append(label.tolist())

                pred_out_prob_list = np.max(out_prob_list, axis=0)
                
                for i in range(131):
                    if i in out_list:
                        pred_out_prob_list[i] = out_prob_list[out_list.index(i)][i]

                y_pred_prob.append(pred_out_prob_list)
                y_pred_label.append(out_list)

                # prediction label
                y_pred_tmp = np.zeros(voc_size[2])
                y_pred_tmp[out_list] = 1
                y_pred.append(y_pred_tmp)
                med_cnt += len(prediction)
                med_cnt_list.append(len(prediction))
            except ValueError:  #raised if `y` is empty.
                pass  

        smm_record.append(y_pred_label)
        for i in range(min(len(labels), 5)):
            try:
                single_ja, single_auc, single_p, single_r, single_f1 = sequence_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]),np.array([y_pred_label[i]]))
                ja_by_visit[i].append(single_ja)
                auc_by_visit[i].append(single_auc)
                pre_by_visit[i].append(single_p)
                recall_by_visit[i].append(single_r)
                f1_by_visit[i].append(single_f1)
                smm_record_by_visit[i].append(y_pred_label[i:i+1])
            except IndexError:
                pass
            

        all_pred_list.append(pred_list)
        all_label_list.append(labels)
        try:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                    sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))
        except IndexError:
            pass

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('jaccard:', [np.mean(buf) for buf in ja])
    print('prauc:', [np.mean(buf) for buf in prauc])
    print('f1:', [np.mean(buf) for buf in avg_f1])
    print('DDI:', [ddi_rate_score(buf) for buf in smm_record_by_visit])
    
    pickle.dump(all_pred_list, open('out_list.pkl', 'wb'))
    pickle.dump(all_label_list, open('out_list_gt.pkl', 'wb'))

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list, gumbel_pick_index, cross_visit_scores_numpy, saved_patient_embedding_test
