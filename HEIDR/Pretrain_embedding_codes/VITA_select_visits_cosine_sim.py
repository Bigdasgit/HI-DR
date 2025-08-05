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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='model_name', help="model name")
parser.add_argument('--epoch', type=str, default='epoch', help="VITA another pretrain best epoch")
parser.add_argument('--topk', type=int, default=1, help='hyperparameter top-k')

args = parser.parse_args()
model_name = args.model_name
epoch = args.epoch
topk = args.topk

os.makedirs(os.path.join("Pretrain_embedding_codes/final_top_embedding", model_name), exist_ok=True)


def select_index_cosine_sim_visit_emb(data_train, best_epoch_embedding, model_name, data_type, epoch, train_embeddings=None): # , data_path):
    # dim1 = 64
    # dim2 = 71
    # cos_linear1 = nn.Linear(dim1,1).to(device='cuda')
    # cos_linear2 = nn.Linear(dim2,1).to(device='cuda')   
    print("\n----select_index_cosine_sim_visit start----\n",
          "data_type: ", data_type,"\n",
          "epoch: ", epoch, "\n",
          "best_epoch_embedding: ", len(best_epoch_embedding),"\n")
    
    same_check_best_emb = []
    concat_all_emb = []
    for index, patient in enumerate(best_epoch_embedding):
        # print("patient: ", patient.shape)
        concat_all_emb.append(patient)
        same_check_best_emb.append(patient.shape[0])
    concat_all_emb = torch.cat(concat_all_emb)
    print("concat_all_emb: ", concat_all_emb.shape)
    print("concat_all_emb: ", concat_all_emb.shape[0])
    
    # concat_all_emb = cos_linear1(concat_all_emb).squeeze(2)
    # concat_all_emb = cos_linear2(cos_query) #.squeeze(1)
    # print("change concat_all_emb: ", concat_all_emb.shape)
    # print("change concat_all_emb: ", concat_all_emb.shape[0])

    # concat_all_emb = torch.randn(concat_all_emb.shape[0], 64)
    # print("concat_all_emb: ", concat_all_emb)

    # cosine similarity
    # if data_type == 'train', calculate similarity within itself and own previous visits
    if data_type == 'train':
        chunk_size = 10
        cos_sim = torch.zeros_like(concat_all_emb.matmul(concat_all_emb.t()))

        for i in range(0, concat_all_emb.size(0), chunk_size):
            chunk_emb = concat_all_emb[i:i+chunk_size]
            cos_sim_chunk = F.cosine_similarity(chunk_emb.unsqueeze(1), concat_all_emb.unsqueeze(0), dim=2)
            cos_sim[i:i+chunk_size] = cos_sim_chunk

        cos_sim.fill_diagonal_(0)
        
        total_visit = []
        for index, patient in enumerate(data_train):
            total_visit.append(len(patient))
        
        split_tensors = []
        start_index = 0
        for num in total_visit:
            end_index = start_index + num
            split_tensors.append(concat_all_emb[start_index:end_index])
            start_index = end_index
        
        global_visit_idx = 0
        for patient_idx, patient_emb in enumerate(split_tensors):
            num_visits = patient_emb.size(0)
            
            for visit_idx in range(num_visits):
                current_visit_idx = global_visit_idx + visit_idx
                
                patient_start_idx = global_visit_idx
                patient_end_idx = global_visit_idx + num_visits
                
                cos_sim[current_visit_idx, current_visit_idx] = 0
                
                if visit_idx < num_visits - 1:
                    future_start_idx = global_visit_idx + visit_idx + 1
                    future_end_idx = global_visit_idx + num_visits
                    cos_sim[current_visit_idx, future_start_idx:future_end_idx] = 0
                
                if visit_idx > 0:
                    pass
                else:
                    cos_sim[current_visit_idx, patient_start_idx:patient_end_idx] = 0
            
            global_visit_idx += num_visits
    else:
        if train_embeddings is None:
            raise ValueError("train_embeddings must be provided for eval/test data")
        
        if isinstance(train_embeddings, list):
            concat_train_emb = []
            for index, patient in enumerate(train_embeddings):
                concat_train_emb.append(patient)
            concat_train_emb = torch.cat(concat_train_emb)
            print("Processed train embeddings shape:", concat_train_emb.shape)
        else:
            concat_train_emb = train_embeddings
        
        chunk_size = 10
        cos_sim_train = torch.zeros(concat_all_emb.size(0), concat_train_emb.size(0), device=concat_all_emb.device)

        for i in range(0, concat_all_emb.size(0), chunk_size):
            chunk_emb = concat_all_emb[i:i+chunk_size]
            cos_sim_chunk = F.cosine_similarity(chunk_emb.unsqueeze(1), concat_train_emb.unsqueeze(0), dim=2)
            cos_sim_train[i:i+chunk_size] = cos_sim_chunk
        
        cos_sim_own = torch.zeros(concat_all_emb.size(0), concat_all_emb.size(0), device=concat_all_emb.device)
        
        total_visit = []
        for index, patient in enumerate(data_train):
            total_visit.append(len(patient))
        
        split_tensors = []
        start_index = 0
        for num in total_visit:
            end_index = start_index + num
            split_tensors.append(concat_all_emb[start_index:end_index])
            start_index = end_index
        
        global_visit_idx = 0
        total_visits_eval = sum(total_visit)
        
        for patient_idx, patient_emb in enumerate(split_tensors):
            num_visits = patient_emb.size(0)
            
            for visit_idx in range(num_visits):
                current_visit_idx = global_visit_idx + visit_idx
                
                if visit_idx > 0:
                    current_visit_emb = concat_all_emb[current_visit_idx:current_visit_idx+1]  # [1, 64]
                    
                    previous_visits_start = global_visit_idx
                    previous_visits_end = global_visit_idx + visit_idx
                    previous_visits_emb = concat_all_emb[previous_visits_start:previous_visits_end]  # [visit_idx, 64]
                    
                    if previous_visits_emb.size(0) > 0:
                        sim_with_own = F.cosine_similarity(current_visit_emb.unsqueeze(1), previous_visits_emb.unsqueeze(0), dim=2)  # [1, visit_idx]
                        cos_sim_own[current_visit_idx, previous_visits_start:previous_visits_end] = sim_with_own.squeeze(0)
                            
            global_visit_idx += num_visits
                
        combined_embeddings = torch.cat([concat_train_emb, concat_all_emb], dim=0)
        cos_sim_combined = torch.cat([cos_sim_train, cos_sim_own], dim=1)
        
        print("Combined embeddings shape:", combined_embeddings.shape)
        print("Combined similarity matrix shape:", cos_sim_combined.shape)

    topk = args.topk # 100
    
    if data_type == 'train':
        top_sim, top_idx = torch.topk(cos_sim, topk, dim=1)
    else:
        top_sim, top_idx = torch.topk(cos_sim_combined, topk, dim=1)
    
    # print("top_10_sim: ", top_10_sim)
    # print("top_100_idx: ", top_idx)
    top_idx_np = top_idx.cpu().numpy()
    
    top_sim_np = top_sim.detach().cpu().numpy()
    avg_cosine_sim = np.mean(top_sim_np)
    print(f"Average cosine similarity for top-{topk} indices ({data_type}): {avg_cosine_sim:.4f}")
    
    jaccard_similarities = []
    data_train_visits = data_visits(data_train)
    
    for i, idx_set in enumerate(top_idx_np):
        if i < len(data_train_visits):
            current_visit_meds = set(data_train_visits[i][2])
            
            visit_jaccard_sims = []
            for neighbor_idx in idx_set:
                if neighbor_idx < len(data_train_visits):
                    neighbor_visit_meds = set(data_train_visits[neighbor_idx][2])
                    
                    intersection = len(current_visit_meds.intersection(neighbor_visit_meds))
                    union = len(current_visit_meds.union(neighbor_visit_meds))
                    
                    if union > 0:
                        jaccard_sim = intersection / union
                        visit_jaccard_sims.append(jaccard_sim)
                    else:
                        visit_jaccard_sims.append(0.0)
            
            if visit_jaccard_sims:
                avg_jaccard = np.mean(visit_jaccard_sims)
                jaccard_similarities.append(avg_jaccard)
    
    if jaccard_similarities:
        avg_jaccard_sim = np.mean(jaccard_similarities)
        print(f"Average Jaccard similarity for medication sets ({data_type}): {avg_jaccard_sim:.4f}")
    else:
        print(f"No valid Jaccard similarities calculated for {data_type}")
    
    if data_type == 'train':
        top_emb = concat_all_emb.index_select(0, top_idx.view(-1)).view(concat_all_emb.shape[0], topk, -1) ## (row_size,top-k,-1)
    else:
        train_emb_size = concat_train_emb.size(0)
        
        top_idx_device = top_idx.to(combined_embeddings.device)
        top_emb = combined_embeddings.index_select(0, top_idx_device.view(-1)).view(concat_all_emb.shape[0], topk, -1)
    
    print("top_emb: ", top_emb.shape)
    print("top_idx_np: ", top_idx_np.shape)
    print("top_idx_np: ", top_idx_np)
    # ### --- save --- ###
    # # dill.dump(top_emb, open(os.path.join('Pretrain_embedding_codes/final_top_embedding', 'final_top_embedding_%s_data_type_%s_epoch_%s.pkl' % (model_name, data_type, epoch)), 'wb'))
    dill.dump(top_idx_np, open(os.path.join('Pretrain_embedding_codes/final_top_embedding', model_name ,'final_top_%d_index_%s_%s_epoch_%s.pkl' % (topk, model_name, data_type, epoch)), 'wb'))

    test_vita_model_input(data_train, top_idx_np, data_type, train_embeddings)



def select_index_cosine_sim_visit_emb_only_past(data_train, best_epoch_embedding, model_name, data_type, epoch): # , data_path):
    # dim1 = 64
    # dim2 = 71
    # cos_linear1 = nn.Linear(dim1,1).to(device='cuda')
    # cos_linear2 = nn.Linear(dim2,1).to(device='cuda')   
    print("\n----select_index_cosine_sim_visit start----\n",
          "data_type: ", data_type,"\n",
          "epoch: ", epoch, "\n",
          "best_epoch_embedding: ", len(best_epoch_embedding),"\n")
    
    same_check_best_emb = []
    concat_all_emb = []
    for index, patient in enumerate(best_epoch_embedding):
        # print("patient: ", patient.shape)
        concat_all_emb.append(patient)
        same_check_best_emb.append(patient.shape[0])
    concat_all_emb = torch.cat(concat_all_emb)
    print("concat_all_emb: ", concat_all_emb.shape)
    print("concat_all_emb: ", concat_all_emb.shape[0])
    
    sum_visit = 0
    total_visit = []
    for index, patient in enumerate(data_train):
        # print("visit: ", len(patient))
        total_visit.append(len(patient))
        sum_visit += len(patient)
    
    data_train_visits = []
    
    for index, patient in enumerate(data_train):
        patient_num = []
        count = 0
        for idx, visit in enumerate(patient):
            if idx ==0 : pass  
            else:
                seq_input = patient[:idx+1]
                count += len([seq_input[-1]])
                patient_num.append(count)
                data_train_visits.append(patient_num) # [seq_input[-1]]
    print("count: ", count)
    print("data_train_visits: ", len(data_train_visits))
    print("data_train_visits: ", data_train_visits)
    total_len = []
    for i in data_train_visits:
        # print(i)
        total_len.append(len(i))
    print("total_len: ", sum(total_len))
    # print("sum_visit: ", sum_visit)
    # print("total_visit: ", len(total_visit))
    # print(total_visit)
    
    cut_result = []
    index = 0
    for num in total_visit:
        cut_result.append(list(range(index, index + num)))
        index += num
    print("cut_result: ", len(cut_result))
    total_visit_num = []
    for i in cut_result:
        total_visit_num.append(len(i))
    print("total_visit_num: ", len(total_visit_num))
    print(total_visit_num)
    print(sum(total_visit_num))
    # split
    split_tensors = []
    start_index = 0
    for num in total_visit_num:
        end_index = start_index + num
        split_tensors.append(concat_all_emb[start_index:end_index])
        start_index = end_index

    # check
    print(len(split_tensors))
    # for tensor in split_tensors:
    #     print(tensor.shape)

    topk = 3
    all_top_emb = []
    all_top_idx = []

    for tensor in split_tensors:
        if tensor.shape[0] == 0:  # Skip if the number of embeddings is 0.
            continue

        # cosine similarity 
        cos_sim = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=2)
        cos_sim.fill_diagonal_(0)
        
        # Obtain the indices of the top-3 embeddings with the highest similarity.
        top_sim, top_idx = torch.topk(cos_sim, min(topk, tensor.shape[0]), dim=1)
        
        # Save the selected embeddings and indices.
        top_emb = tensor.index_select(0, top_idx.view(-1)).view(tensor.shape[0], -1, tensor.shape[1])
        all_top_emb.append(top_emb)
        all_top_idx.append(top_idx)

    # check
    # for i, (top_emb, top_idx) in enumerate(zip(all_top_emb, all_top_idx)):
    #     print(f"Group {i+1}:")
    #     print("Top embeddings:", top_emb.shape)
    #     print("Top indices:", top_idx)
    #     print("------")
    print("all_top_idx\n",len(all_top_idx))
    # count = 0
    # for index, patient in enumerate(data_train):
    #     for idx, visit in enumerate(patient):
    #         if idx ==0 : pass  
    #         else:
    #             count += 1
    #             # seq_input = patient[:idx+1]
    #             # count += len([seq_input[-1]])
    #             # data_train_visits += [seq_input[-1]]
    # print("count: ", count)
    
    """
    # ### --- save --- ###
    # # dill.dump(top_emb, open(os.path.join('Pretrain_embedding_codes/final_top_embedding', 'final_top_embedding_%s_data_type_%s_epoch_%s.pkl' % (model_name, data_type, epoch)), 'wb'))
    # dill.dump(top_idx_np, open(os.path.join('Pretrain_embedding_codes/final_top_embedding', model_name ,'final_top_%d_index_%s_%s_epoch_%s.pkl' % (topk, model_name, data_type, epoch)), 'wb'))    
    """
    # test_vita_model_input(data_train, top_idx_np)
    

    
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
    
def test_vita_model_input(data_train, top_idx_np, data_type='train', train_embeddings=None):
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

    # Now, each element of the selected_rows list contains the rows corresponding to the medications used in the respective visit.
    # print("top_10_idx: ", top_10_idx_np.shape)
    # top_10_idx_np = top_idx.cpu().numpy() # tensor를 numpy array로 변환
    # print("top_10_idx: ", top_10_idx)

    # Retrieve the elements from visit_med_list corresponding to each index set.
    # top_10_meds = [visit_med_list[idx_set] for i, idx_set in enumerate(top_10_idx_np)]
    # top_10_meds = [[visit_med_list[i][idx] for idx in idx_set] for i, idx_set in enumerate(top_10_idx_np)]
    
    # For eval/test, we need to handle the index mapping properly
    if data_type in ['eval', 'test']:
        # Map indices: if index < len(data_train_visits), use data_train_visits, otherwise skip
        top_visits = []
        for i, idx_set in enumerate(top_idx_np):
            visit_set = []
            for idx in idx_set:
                if idx < len(data_train_visits):
                    visit_set.append(data_train_visits[idx])
            top_visits.append(visit_set)
    else:
        # For train, use data_train_visits directly
        top_visits = [[data_train_visits[idx] for idx in idx_set] for i, idx_set in enumerate(top_idx_np)]

    print("top_visits: ", len(top_visits))
    # for i in top_meds:
        # print(i)
        # print(len(i))

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
    
def data_visit_num(data, final_top_idx_GAMENet_another_pretrain_for_embedding_1_data_type_train_epoch, data_type):
    print("data_type: ", data_type)
    sum_visit = 0
    total_visit = []
    for index, patient in enumerate(data):
        # print("visit: ", len(patient))
        total_visit.append(len(patient))
        sum_visit += len(patient)

    cut_result = []
    index = 0
    for num in total_visit:
        cut_result.append(list(range(index, index + num)))
        index += num

    print("result: ", len(cut_result))
    print("final_top_idx_GAMENet_another_pretrain_for_embedding_1_data_type_train_epoch: ", len(final_top_idx_GAMENet_another_pretrain_for_embedding_1_data_type_train_epoch))
    final_top_idx_result = []
    original_indices = []
    start_index = 0
    for visit in total_visit:
        end_index = start_index + visit
        group = final_top_idx_GAMENet_another_pretrain_for_embedding_1_data_type_train_epoch[start_index:end_index]
        original_indices.extend(range(start_index, end_index))
        final_top_idx_result.append(group)
        start_index = end_index
    print("final_top_idx_result: ", len(final_top_idx_result))
    
    saved_index = []
    for index, visit_num in enumerate(final_top_idx_result):
        for idx, visit in enumerate(visit_num):
            if all(item in visit_num for item in cut_result) == True: saved_index.append(index)
    print("saved_index: ", saved_index)

    return total_visit


def main():
    device = torch.device('cuda')
    tic = time.time()
    print(device)

    records_final_path = '../../data/records_final.pkl'
    
    data=dill.load(open(records_final_path, 'rb'))
    
    data = [x for x in data if len(x) >= 2]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]


    print ('training time: {}'.format(time.time() - tic))
    
    # Insert best epoch embedding from VITA_another_pretrain_main.py
    best_epoch_embedding_train_path = 'Pretrain_embedding_codes/saved_embedding/train/Epoch_40_patient_embedding_train.pkl'
    best_epoch_embedding_eval_path = 'Pretrain_embedding_codes/saved_embedding/eval/Epoch_40_patient_embedding_eval.pkl'
    best_epoch_embedding_test_path = 'Pretrain_embedding_codes/saved_embedding/test/Epoch_40_patient_embedding_test.pkl'
    best_epoch_embedding_train=dill.load(open(best_epoch_embedding_train_path, 'rb'))
    best_epoch_embedding_eval=dill.load(open(best_epoch_embedding_eval_path, 'rb'))
    best_epoch_embedding_test=dill.load(open(best_epoch_embedding_test_path, 'rb'))
    
    ## Insert only the visits corresponding to the selected indices.
    model_name = args.model_name # 'VITA_another_pretrain_for_embedding_top_100_1'
    epoch = args.epoch
    # select_index_cosine_sim_visit_emb_only_past(data_train, best_epoch_embedding_train, model_name = model_name, data_type='train', epoch=epoch) # , data_path='RefineGraphNet/data/pretrain_saved/change_index_to_visit_epoch_28_loss_1_select_another_pateint_index_train.pkl')
    select_index_cosine_sim_visit_emb(data_train,best_epoch_embedding_train, model_name = model_name, data_type='train', epoch=epoch) # , data_path='RefineGraphNet/data/pretrain_saved/change_index_to_visit_epoch_28_loss_1_select_another_pateint_index_train.pkl')
    select_index_cosine_sim_visit_emb(data_eval,best_epoch_embedding_eval, model_name = model_name, data_type='eval', epoch=epoch, train_embeddings=best_epoch_embedding_train) # , data_path='RefineGraphNet/data/pretrain_saved/change_index_to_visit_epoch_28_loss_1_select_another_pateint_index_eval.pkl')
    select_index_cosine_sim_visit_emb(data_test,best_epoch_embedding_test, model_name = model_name, data_type='test', epoch=epoch, train_embeddings=best_epoch_embedding_train) # , data_path='RefineGraphNet/data/pretrain_saved/change_index_to_visit_epoch_28_loss_1_select_another_pateint_index_test.pkl')
    # test_gamenet_model_input(data_train, final_top_embedding_data_type_train_epoch_7)
    # test_gamenet_model_input(data_eval, final_top_embedding_data_type_eval_epoch_7)
    # test_gamenet_model_input(data_test, final_top_embedding_data_type_test_epoch_7)
    print("---Done---")
   
if __name__ == '__main__':
    main()
    
