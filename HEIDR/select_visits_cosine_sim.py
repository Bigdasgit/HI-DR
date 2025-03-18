import dill
import os
import torch
import argparse
from util import llprint 
import time
import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
from torch.nn import functional as F
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='model_name', help="model name")
parser.add_argument('--epoch', type=str, default='epoch', help="VITA another pretrain best epoch")
parser.add_argument('--topk', type=int, default=3, help='hyperparameter top-k')
parser.add_argument('--data_type',type=str, default='data_type', help='data_type')

args = parser.parse_args()
model_name = args.model_name
epoch = args.epoch
topk = args.topk
data_type = args.data_type

os.makedirs(os.path.join("HEIDR/final_top_embedding", model_name), exist_ok=True)
os.makedirs(os.path.join("HEIDR/top_sim", model_name), exist_ok=True)
# Insert the visits corresponding to the indices (another visit). Only insert the embeddings of the selected index visits.
def select_index_cosine_sim_visit_emb(data_train, best_epoch_embedding, model_name, data_type, epoch): # , data_path):
     
    print("\n----select_index_cosine_sim_visit start----\n",
          "data_type: ", data_type,"\n",
          "epoch: ", epoch, "\n",
          "best_epoch_embedding: ", len(best_epoch_embedding),"\n")
    
    same_check_best_emb = []
    concat_all_emb = []
    for index, patient in enumerate(best_epoch_embedding):

        concat_all_emb.append(patient)
        same_check_best_emb.append(patient.shape[0])
    concat_all_emb = torch.cat(concat_all_emb)
    print("concat_all_emb: ", concat_all_emb.shape)
    print("concat_all_emb: ", concat_all_emb.shape[0])
    
    # concat_all_emb = cos_linear1(concat_all_emb).squeeze(2)
    # concat_all_emb = cos_linear2(cos_query) #.squeeze(1)
    # print("change concat_all_emb: ", concat_all_emb.shape)
    # print("change concat_all_emb: ", concat_all_emb.shape[0])

    # cosine similarity 
    # cos_sim = F.cosine_similarity(concat_all_emb.unsqueeze(1), concat_all_emb.unsqueeze(0), dim=2)
    chunk_size = 10  
    cos_sim = torch.zeros_like(concat_all_emb.matmul(concat_all_emb.t()))  # Initialize matrices to store the results.

    for i in range(0, concat_all_emb.size(0), chunk_size):
        chunk_emb = concat_all_emb[i:i+chunk_size]
        cos_sim_chunk = F.cosine_similarity(chunk_emb.unsqueeze(1), concat_all_emb.unsqueeze(0), dim=2)
        cos_sim[i:i+chunk_size] = cos_sim_chunk

    # Remove the diagonal elements from the cosine similarity matrix (similarity with oneself is not needed).
    cos_sim.fill_diagonal_(0)

    # Obtain the indices of embeddings corresponding to the top-10 rows with the highest similarity.
    topk = args.topk # 100
    top_sim, top_idx = torch.topk(cos_sim, topk, dim=1)
    # print("top_10_sim: ", top_10_sim)
    # print("top_100_idx: ", top_idx)
    top_idx_np = top_idx.cpu().numpy()
    top_emb = concat_all_emb.index_select(0, top_idx.view(-1)).view(concat_all_emb.shape[0], topk, -1) ## (row_size,top-k,-1)
    print("top_emb: ", top_emb.shape)
    print("top_idx_np: ", top_idx_np.shape)
    print("top_idx_np: ", top_idx_np)
    # ### --- save --- ###
    # dill.dump(top_sim, open(os.path.join('HEIDR/top_sim', model_name ,'final_top_%d_similarity_%s_%s_epoch_%s.pkl' % (topk, model_name, data_type, epoch)), 'wb'))
    # dill.dump(top_emb, open(os.path.join('HEIDR/final_top_embedding', model_name ,'final_top_%d_embedding_%s_%s_epoch_%s.pkl' % (topk, model_name, data_type, epoch)), 'wb'))
    # dill.dump(top_idx_np, open(os.path.join('HEIDR/final_top_embedding', model_name ,'final_top_%d_index_%s_%s_epoch_%s.pkl' % (topk, model_name, data_type, epoch)), 'wb'))    

    test_vita_model_input(data_train, top_idx_np)
    

def data_visits_vita(data_train):
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
    
def test_vita_model_input(data_train, top_idx_np):
    def model(seq_input,sim_visits_emb,sim_visits_med):
        print("---input model-----")
        # print("\nseq_input: ",len(seq_input))
        # print("sim_visits: ",sim_visits_emb.shape)
        # print("sim_visits_med: ",len(sim_visits_med))
    
    patient_count = 0
    same_check_visit = []
    
    print("data_train: ", len(data_train))
    # data_train_visits = sum(data_train, [])
    data_train_visits = data_visits(data_train)
    print("data_train_visits: ", len(data_train_visits))
    # print("data_train_visits: ", data_train_visits)

    top_visits = [[data_train_visits[idx] for idx in idx_set] for i, idx_set in enumerate(top_idx_np)]

    print("top_visits: ", len(top_visits))

    for index, patient in enumerate(data_train):
        for idx, visit in enumerate(patient):
            if idx ==0 : pass  
            else:
                seq_input = patient[:idx+1]
                sim_visits = top_visits[patient_count] # [top_visits[patient_count][0]]

                print("sim_visits: ", len(sim_visits))
                
                same_check_visit.append(len(seq_input))
                patient_count += 1
    # print("final_top_embedding: ", final_top_embedding.shape)
    print("patient_count: ", patient_count)

## ====== only past cosine similarity ===== ## 
def data_visits(data_train):
    # data_train_visits = sum(data_train, [])
    visits_per_patient = {}
    data_train_visits = []
    count = 0
    for index, patient in enumerate(data_train):
        visits_per_patient[index] = 0
        for idx, visit in enumerate(patient):
            if idx ==0 : pass  
            else:
                seq_input = patient[:idx+1]
                count += len([seq_input[-1]])
                data_train_visits += [seq_input[-1]]
                visits_per_patient[index] += 1
    print("count: ", count)
    print("data_visits: ", len(data_train_visits))
    print("visits_per_patient\n", visits_per_patient)
    # Collecting only the values from the visits_per_patient dictionary
    visits_values_count = list(visits_per_patient.values())
    print(visits_values_count)
    print(sum(visits_values_count))
    
    # visit_med_list_train = []
    # for i in data_train_visits:
    #     visit_med_list_train.append(i[2])
    # print("visit_med_list: ", len(visit_med_list_train))
    return data_train_visits, visits_values_count

def group_embeddings_by_visits(embeddings, visits):
    grouped_embeddings = []
    start_idx = 0
    for num_visits in visits:
        # Group the embeddings for the current patient's number of visits
        group = embeddings[start_idx:start_idx + num_visits]
        grouped_embeddings.append(torch.cat(group, dim=0))  # Concatenate along the first dimension
        start_idx += num_visits
    return grouped_embeddings

def group_by_visits(data_train_visits, visits):
    grouped_visits = []
    start_idx = 0
    for num_visits in visits:
        # Group the embeddings for the current patient's number of visits
        group = data_train_visits[start_idx:start_idx + num_visits]
        grouped_visits.append(group)  # Concatenate along the first dimension
        start_idx += num_visits
    return grouped_visits

# Function to replace top-3 cosine similarity indices with the actual visit information
def replace_indices_with_grouped_visits(top3_indices, grouped_visits):
    top3_per_visit_real_group = []

    # Iterate over each group and its corresponding top-3 indices
    for group_index, group_top3_indices in enumerate(top3_indices):
        group_real_visits = []
        
        # Iterate over each visit and its corresponding top-3 indices within the group
        for visit_index, visit_top3_indices in enumerate(group_top3_indices):
            visit_top3_real = []
            
            # Iterate over each index in the top-3 and replace with the actual visit information
            for idx in visit_top3_indices:
                if idx < len(grouped_visits[group_index]):
                    visit_top3_real.append(grouped_visits[group_index][idx])
                else:
                    # If the index is out of bounds, it indicates there are less than 3 visits in the group
                    # We fill the remaining with the last visit information to maintain the size
                    visit_top3_real.append(grouped_visits[group_index][-1])
            
            group_real_visits.append(visit_top3_real)
        
        top3_per_visit_real_group.append(group_real_visits)
    
    return top3_per_visit_real_group

# Function to compute top-3 cosine similarity and replace indices with actual visits
def compute_top3_cosine_similarity_and_replace_indices(groups, grouped_visits):
    top3_per_visit_real_group = []

    for index, group in enumerate(groups):
        similarities = []
        max_top_k = min(group.size(0), 4)  # Calculate the maximum number of top similarities we can find
        
        # Compute cosine similarity for each visit against all others in the group
        for i in range(len(group)):
            similarities.append(F.cosine_similarity(group[i].unsqueeze(0), group, dim=1))
        
        # Get the indices of the top similarities for each visit, excluding self-similarity
        top3_per_visit_indices = [
            sim.topk(max_top_k, largest=True).indices[1:max_top_k] if sim.numel() > 0 else []
            for sim in similarities
        ]

        # Replace indices with the actual visits from grouped_visits
        top3_per_visit_real = []
        for visit_indices in top3_per_visit_indices:
            if len(visit_indices) > 0:  # Check if the list of indices is not empty
                visit_real = [grouped_visits[index][i.item()] for i in visit_indices]
                top3_per_visit_real.append(visit_real)
            else:
                top3_per_visit_real.append([])  # Add an empty list if there are no indices
        
        top3_per_visit_real_group.append(top3_per_visit_real)

    return top3_per_visit_real_group


# Function to compute top-3 cosine similarity for visits within each group with error handling for groups with less than 4 visits
def compute_top3_cosine_similarity(groups, grouped_visit):
    top3_similarities_per_group = []
    top3_similarities_per_real_group = []
    # print("grouped_visit: ", grouped_visit)
    for index, group in enumerate(groups):
        # print("group: ", group)
        similarities = []
        # Calculate the maximum number of top similarities we can find (minimum is 1)
        max_top_k = min(group.size(0), 4)
        # Compute cosine similarity for each visit against all others in the group
        for i in range(len(group)):
            similarities.append(
                F.cosine_similarity(group[i].unsqueeze(0), group, dim=1)
            )
        # Get the indices of the top similarities for each visit
        top3_per_visit = [
            sim.topk(max_top_k, largest=True).indices[1:max_top_k]  # Exclude self-similarity
            for sim in similarities
        ]
        # print(top3_per_visit)
        # print("grouped_visit: ",grouped_visit[index])
        top3_per_visit_real_group = replace_indices_with_grouped_visits(top3_per_visit, grouped_visit[index])
        top3_similarities_per_group.append(top3_per_visit)
        top3_similarities_per_real_group.append(top3_per_visit_real_group)
        
    return top3_similarities_per_group, top3_similarities_per_real_group

# Function to map the top-3 indices to the actual visits in data_train_visits
# top3_similarities, grouped_visit, data_train
def map_top3_indices_to_visits(groups, top3_indices, visits):
    # Flatten the data_train_visits to match the grouped structure
    visits_flat = [item for sublist in visits for item in sublist]
    top3_visits = []
    current_visit_index = 0

    for group_index, group in enumerate(groups):
        group_top3_visits = []
        for visit_index, top3 in enumerate(group):
            # Map the top-3 indices to the actual visits
            top3_actual_visits = [visits_flat[current_visit_index + idx] for idx in top3.tolist()]
            group_top3_visits.append(top3_actual_visits)
            current_visit_index += 1  # Move to the next visit index
        top3_visits.append(group_top3_visits)
        # Adjust the current index to skip the visits of the current group as they are already processed
        current_visit_index += len(visits[group_index]) - len(group)
    
    return top3_visits



def select_index_cosine_only_past(visits_values_count, data_train_visits, best_epoch_embedding_train):
    # print(len(data_train_visits))
    # print(data_train_visits)
    # for i in data_train_visits:
    #     print(i)
    # Group the embeddings based on visits_values
    grouped_visit = group_by_visits(data_train_visits, visits_values_count)
    # for i, group in enumerate(grouped_visit):
    #     # print(f"Group {i} shape: {len(group)}")
    #     print(group)
        
    grouped_embeddings = group_embeddings_by_visits(best_epoch_embedding_train, visits_values_count)
    # For demonstration, print out the shape of each group
    # for i, group in enumerate(grouped_embeddings):
    #     print(f"Group {i} shape: {group.shape}")

    # Compute top-3 cosine similarities for each group
    top3_similarities_per_real_group = compute_top3_cosine_similarity_and_replace_indices(grouped_embeddings, grouped_visit)
    # top3_similarities, top3_similarities_per_real_group = compute_top3_cosine_similarity(grouped_embeddings, grouped_visit)
    for index, patient in enumerate(top3_similarities_per_real_group):
        print("index: ", index)
        for idx, visit in enumerate(patient):
            print("idx: ", idx)
            print(len(visit))
            print(visit)
    # dill.dump(top3_similarities_per_real_group, open(os.path.join('HEIDR/final_top_embedding', model_name ,'mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train.pkl'), 'wb'))
    # dill.dump(top3_similarities_per_real_group, open(os.path.join('HEIDR/final_top_embedding', model_name ,'mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval.pkl'), 'wb'))
    # dill.dump(top3_similarities_per_real_group, open(os.path.join('HEIDR/final_top_embedding', model_name ,'mimic_iii_29epoch_rq2_i1_i2_top3_only_past_test.pkl'), 'wb'))

def top_3_another_visits(data, top_visits):
    patient_count = 0
    for step, input in enumerate(data):
        for idx, adm in enumerate(input):
            if idx == 0: 
                pass             
            else:
                print(len(top_visits[patient_count][:args.topk]))
                print(top_visits[patient_count][:args.topk]) 
                patient_count += 1
                
def check_data(data_train, mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train, top_visits):
    print(len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train))
    print(len(data_train))
    patient_count = 0
    for index, patient in enumerate(data_train): # mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train
        # print("index: ", index)
        for idx, visit in enumerate(patient):
            if idx == 0:
                pass
            else:  
                count = 0
                if len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count]) == 3:
                    print("more three idx: ", len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count]))
                    # print(len(visit))
                    # print("visit: ", visit)  
                    # print("mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train: ",mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count])
                else:
                    print("past visit not three: ", len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count]))
                    another_len = 3-len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count])
                    mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count] = mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count] + top_visits[patient_count][:another_len]
                    print("total count: ", len(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count]))
                    # print(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train[index][count])
                count += 1  
                patient_count += 1
                
             
def main():
    device = torch.device('cpu')
    tic = time.time()
    print(device)

    records_final_path = 'data/records_final.pkl' # 'data/mimic-iv/records_final2.pkl'
    
    data=dill.load(open(records_final_path, 'rb'))
    
    data = [x for x in data if len(x) >= 2]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]



    print ('training time: {}'.format(time.time() - tic))
    
    # best_epoch_embedding_train_path = 'HEIDR/saved_embedding/train/VITA_another_pretrain_for_embedding_make_query_gumbel_08_att_10/Epoch_29_patient_embedding_train.pkl'# 'HEIDR/saved_embedding/train/mimic_iv_VITA_pretrain_for_embedding_1/Epoch_37_patient_embedding_train.pkl'
    # best_epoch_embedding_eval_path = 'HEIDR/saved_embedding/eval/VITA_another_pretrain_for_embedding_make_query_gumbel_08_att_10/Epoch_29_patient_embedding_eval.pkl'# 'HEIDR/saved_embedding/eval/mimic_iv_VITA_pretrain_for_embedding_1/Epoch_37_patient_embedding_eval.pkl'
    # best_epoch_embedding_test_path = 'HEIDR/saved_embedding/test/VITA_another_pretrain_for_embedding_make_query_gumbel_08_att_10/patient_embedding_test.pkl'# 'HEIDR/saved_embedding/test/mimic_iv_VITA_pretrain_for_embedding_1/patient_embedding_test.pkl'
    # best_epoch_embedding_train=dill.load(open(best_epoch_embedding_train_path, 'rb'))
    # best_epoch_embedding_eval=dill.load(open(best_epoch_embedding_eval_path, 'rb'))
    # best_epoch_embedding_test=dill.load(open(best_epoch_embedding_test_path, 'rb'))
    
   
    ## Insert only the visits corresponding to the selected indices.
    model_name = args.model_name # 'mimic_iv_VITA_pretrain_for_embedding_1'
    epoch = args.epoch # 37
    
    ## ===== only past ==== 
    final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_train_epoch_27_path = 'HEIDR/final_top_embedding/VITA_another_pretrain_for_embedding_top_100_1/final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_train_epoch_27.pkl'
    final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_eval_epoch_27_path = 'HEIDR/final_top_embedding/VITA_another_pretrain_for_embedding_top_100_1/final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_eval_epoch_27.pkl'
    final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_test_epoch_27_path = 'HEIDR/final_top_embedding/VITA_another_pretrain_for_embedding_top_100_1/final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_test_epoch_27.pkl'
    final_top_idx_data_type_train=dill.load(open(final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_train_epoch_27_path, 'rb'))
    final_top_idx_data_type_eval=dill.load(open(final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_eval_epoch_27_path, 'rb'))
    final_top_idx_data_type_test=dill.load(open(final_top_100_index_VITA_another_pretrain_for_embedding_top_100_1_test_epoch_27_path, 'rb'))    
    
    print("data_train: ", len(data_train))
    data_train_visits = data_visits_vita(data_train)
    print("data_train_visits: ", len(data_train_visits))

    print("data_eval: ", len(data_eval))
    data_eval_visits = data_visits_vita(data_eval)
    # data_eval_visits = sum(data_eval, [])
    print("data_eval_visits: ", len(data_eval_visits))
    
    print("data_test: ", len(data_test))
    # data_test_visits = sum(data_test, [])
    data_test_visits = data_visits_vita(data_test)
    print("data_test_visits: ", len(data_test_visits))
    
    top_visits_train = [[data_train_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_train)] # final_top_idx_data_type_eval
    top_visits_eval = [[data_eval_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_eval)] 
    top_visits_test = [[data_test_visits[idx] for idx in idx_set] for i, idx_set in enumerate(final_top_idx_data_type_test)] 
    print("top_visits_train: ", len(top_visits_train) , "\ntop_visits_eval: ", len(top_visits_eval), "\ntop_visits_test: ", len(top_visits_test))    
    
    # refine_data_train_visits, visits_values_count_train = data_visits(data_train)
    # refine_data_eval_visits, visits_values_count_eval = data_visits(data_eval)
    # refine_data_test_visits, visits_values_count_test = data_visits(data_test)
    # top_3_another_visits(data_test, top_visits_test)
    # select_index_cosine_only_past(visits_values_count_train, refine_data_train_visits, best_epoch_embedding_train)
    # select_index_cosine_only_past(visits_values_count_eval, refine_data_eval_visits, best_epoch_embedding_eval)
    # select_index_cosine_only_past(visits_values_count_test, refine_data_test_visits, best_epoch_embedding_test)
    
    mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train_path = 'HEIDR/final_top_embedding/mimic_iii_29epoch_rq2_i1_i2_top3_only_past/mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train.pkl'
    # mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval_path = 'HEIDR/final_top_embedding/mimic_iii_29epoch_rq2_i1_i2_top3_only_past/mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval.pkl'
    # mimic_iii_29epoch_rq2_i1_i2_top3_only_past_test_path = 'HEIDR/final_top_embedding/mimic_iii_29epoch_rq2_i1_i2_top3_only_past/mimic_iii_29epoch_rq2_i1_i2_top3_only_past_test.pkl'
    
    mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train=dill.load(open(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train_path, 'rb'))
    # mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval=dill.load(open(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval_path, 'rb'))
    # mimic_iii_29epoch_rq2_i1_i2_top3_only_past_test=dill.load(open(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_test_path, 'rb'))
    
    # check_data(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train, data_train)
    # check_data(mimic_iii_29epoch_rq2_i1_i2_top3_only_past_eval, data_eval)
    check_data(data_train, mimic_iii_29epoch_rq2_i1_i2_top3_only_past_train, top_visits_train)
    
    
    
    print("---finish---")
   
if __name__ == '__main__':
    main()
    
