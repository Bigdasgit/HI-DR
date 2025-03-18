import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import math
import numpy as np
from torch.nn.modules.linear import Linear
# from data.processing import process_visit_lg2
from torch.nn import Sequential, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from layers import SelfAttend
from layers import GraphConvolution


class HEIDR(nn.Module):
    """在CopyDrug_batch基础上将medication的encode部分修改为transformer encoder"""
    def __init__(self, voc_size, ehr_adj, ddi_adj, ddi_mask_H, topk,  gumbel_tau, att_tau, emb_dim=64, device=torch.device('cpu')):
        super(HEIDR, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = 2
        self.SOS_TOKEN = voc_size[2]        # start token initialization
        self.END_TOKEN = voc_size[2]+1      # end token initialization
        self.MED_PAD_TOKEN = voc_size[2]+2      # Used for padding (all zeros) in the embedding matrix
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        
        
        # concat embedding 
        self.concat_embedding = nn.Sequential( 
            nn.Embedding(voc_size[0]+3 + voc_size[1]+3, emb_dim, self.DIAG_PAD_TOKEN + self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )
        self.linear_layer = nn.Linear(emb_dim,emb_dim)
        
        self.mlp_layer = nn.Linear(71,1) # From diagnosis, procedures to the maximum dimension, to one dimension
        
        # Layers for creating inputs for Gumbel
        self.gumbel_layer1 = nn.Linear(64,1)
        self.gumbel_layer2 = nn.Linear(71,2)

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # Add padding_idx, indicating to take a zero vector
            nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # Used to encode the medication of the previous visit
        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        # TwoLayerDirectedGCN
        self.ehr_direct_gcn = TwoLayerDirectedGCN(in_channels=voc_size[2], hidden_channels=emb_dim, out_channels=emb_dim) # DirectedGCNConv(in_channels=voc_size[2], out_channels=emb_dim)# DirectedGCN(num_features=voc_size[2],hidden_channels=emb_dim) # num_features, hidden_channels):
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ddi_adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        # Aggregate diag and proc within a single visit to obtain a visit-level representation
        self.diag_self_attend = SelfAttend(emb_dim)

        self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim*2, dropout=0.2, 
                 layer_norm_eps=1e-5)


        # Used for generating drug sequences
        self.dec_gru = nn.GRU(emb_dim*3, emb_dim, batch_first=True)
        

        self.diag_attn = nn.Linear(emb_dim*2, 1)
        self.proc_attn = nn.Linear(emb_dim*2, 1)
        self.W_diag_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_attn = nn.Linear(emb_dim, emb_dim)

        # weights
        self.Ws = nn.Linear(emb_dim*2, emb_dim)  # only used at initial stage
        self.Wo = nn.Linear(emb_dim, voc_size[2]+2)  # generate mode
        self.Wc = nn.Linear(emb_dim, emb_dim) 

        self.W_dec = nn.Linear(emb_dim, emb_dim)
        self.W_stay = nn.Linear(emb_dim, emb_dim)

        # swtich network to calculate generate probablity
        self.W_z = nn.Linear(emb_dim, 1)

        self.MLP_layer = nn.Linear(emb_dim * 2,1)
        self.MLP_layer2 = nn.Linear(71,1)
        self.MLP_layer3 = nn.Linear(emb_dim, 1)
        self.MLP_layer4 = nn.Linear(2, 1)
        
        # hyperparameters; gumbel tau, att tau and top-k
        self.gumbel_tau = gumbel_tau #0.6
        self.att_tau = att_tau # 20
        self.topk = topk #3 #2 #1
        
        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], voc_size[2], False)

    def make_query(self, input_disease_embdding, another_visit_emb):
        batch_size = 1
        max_visit_num = input_disease_embdding.size()[0]
        
        emb_dim = 64
        visit_score_emb = torch.cat((input_disease_embdding[:-1, :, :], another_visit_emb, input_disease_embdding[-1:, :, :]), dim=0)  # shape: [K, 71, 64]
        all_visit_max_visit_num = visit_score_emb.size()[0]
        
        ## make query; patient representation
        input1 = self.MLP_layer3(input_disease_embdding).squeeze(-1) # (seq, Total number of diagnoses/procedures)
        input2 = self.MLP_layer2(input1) # (seq,1), make dense embedding
        current = input2[-1:, :]
        current2 = current.repeat(input2.size()[0],1)
        concat = torch.cat([input2, current2],dim = -1)
        """
        concat = torch.cat([input2, current2],dim = -1)
        gumbel_input = torch.sigmoid(self.MLP_layer4(concat)) #torch.cat([input_g, 1-input_g], dim = -1)
        """
        concat2 = torch.sigmoid(self.MLP_layer4(concat))
        gumbel_input = torch.cat([concat2, 1 - concat2], dim = -1)
        pre_gumbel = F.gumbel_softmax(gumbel_input, tau = self.gumbel_tau, hard = True)[:, 0]
        gumbel  = torch.cat([pre_gumbel[:-1], torch.ones(1, device = self.device)])
        ## gumbel pick relevent past visit embedding
        picked = input_disease_embdding.mul(gumbel.unsqueeze(-1).unsqueeze(-1).expand(-1, 71, 64)) 

        visit_diag_embedding = self.mlp_layer(picked.transpose(-2,-1)).view(batch_size, max_visit_num, emb_dim) # Each diagnosis + procedures per visit stay consolidated into one per stay
        
        ## Changing dimensions also for the added another visit
        visit_score_emb = self.mlp_layer(visit_score_emb.transpose(-2,-1)).view(batch_size, all_visit_max_visit_num, emb_dim) 
        
        cross_visit_scores, scores_encoder = self.calc_cross_visit_scores(visit_diag_embedding, gumbel, visit_score_emb) 
        
        score_emb = input_disease_embdding.mul(cross_visit_scores.unsqueeze(-1).unsqueeze(-1).expand(-1, 71, 64))
        q_t = torch.sum(score_emb, dim = 0, keepdim = True)
        
        gumbel_numpy = pre_gumbel.cpu().detach().numpy()
        gumbel_pick_index = [i+1 for i in (list(filter(lambda x: gumbel_numpy[x] == 1, range(len(gumbel_numpy)))))]
        
        if gumbel_pick_index == []:
            gumbel_pick_index = [0]
            
        
        return q_t, gumbel_pick_index, scores_encoder
    
    def encode(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, ehr_adj, max_len=20):
        device = self.device
        
        batch_size, max_visit_num, max_med_num = medications.size() 
        emb_dim = 64

        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        ## concatenation of two vectors
        p_change = torch.tensor([1958]).to(device).repeat(torch.tensor(procedures).size(0)) + torch.tensor(procedures) # Add the maximum index of diagnosis +1 to the surgery integer encoding
        adm_1_2 = torch.cat([torch.tensor(diseases), p_change],dim = -1)
        input_disease_embdding_include_another = self.concat_embedding(adm_1_2).view(batch_size * max_visit_num, max_diag_num + max_proc_num, self.emb_dim)      # [batch, seq, max_diag_num, emb]
        
        # Separately extract the top-k rows from the back
        another_visit_emb = input_disease_embdding_include_another[-(self.topk+1):-1, :, :]  # shape: [1, 71, 64]
        # Combine the rows excluding the top-k rows from the back
        input_disease_embdding = torch.cat((input_disease_embdding_include_another[:-(self.topk+1), :, :], input_disease_embdding_include_another[-1:, :, :]), dim=0)  # shape: [K-1, 71, 64]
        
        ##concatenation of two mask matrix
        d_p_mask_matrix = torch.cat([d_mask_matrix, p_mask_matrix], dim = -1) 
        d_enc_mask_matrix = d_p_mask_matrix.view(batch_size * max_visit_num, max_diag_num + max_proc_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_diag_num + max_proc_num,1) # [batch*seq, nhead, input_length, output_length]
        
        d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num + max_proc_num, max_diag_num + max_proc_num) # 변경한거
        
        # make query using gumbel softmax
        queries = []
        for i in range(1, input_disease_embdding.size()[0]):
            q_t, gumbel_pick_index, cross_visit_scores = self.make_query(input_disease_embdding[:i+1, :, :], another_visit_emb)
            queries.append(q_t)
        pre_queries = torch.cat(queries)
 
        input_disease_embddinges = torch.cat([input_disease_embdding[:1, :, :], pre_queries]) # (seq, 71, emb_dim)
        ## add another visit emb
        input_disease_embdding = torch.cat([input_disease_embddinges[:-1, :, :],another_visit_emb, input_disease_embddinges[-1:, :, :]])
        
        input_disease_embdding = input_disease_embdding.unsqueeze(dim=0) # (batch_size, max_visit, Maximum number of diagnoses/procedures, emb_dim)
        
        counts = 0

        # Construct a last_seq_medication to represent the medication of the previous visit. For the first visit, since there is no previous medication, fill it with 0 (anything can be used for filling, as it won't be used anyway).
        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)  

        # The m_mask_matrix matrix also needs to be shifted backwards.
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device)
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)  
       
        # Encode the last_seq_medication
        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,self.nhead,max_med_num,1)
        last_m_enc_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        encoded_medication = self.medication_encoder(last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim), src_mask=last_m_enc_mask) # (batch*seq, max_med_num, emb_dim)
        encoded_medication = encoded_medication.view(batch_size, max_visit_num, max_med_num, self.emb_dim)

        # vocab_size, emb_size
        ddi_embedding = self.gcn()
        # using weighted directed ehr graph
        ehr_embedding = self.ehr_direct_gcn(ehr_adj)
        drug_memory = ehr_embedding  - ddi_embedding * self.inter
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        return input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, counts, gumbel_pick_index


    def decode(self, input_medications, input_disease_embedding, last_medication_embedding, last_medications, cross_visit_scores,
        d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory): 
        
        batch_size = input_medications.size(0)
        max_visit_num = input_medications.size(1) 
        max_med_num = input_medications.size(2)
        
        max_diag_num = input_disease_embedding.size(2) # number of diag and proc


        input_medication_embs = self.med_embedding(input_medications).view(batch_size * max_visit_num, max_med_num, -1)
        input_medication_memory = drug_memory[input_medications].view(batch_size * max_visit_num, max_med_num, -1)

        m_self_mask = m_mask_matrix
        
        d_p_mask_matrix = torch.cat([d_mask_matrix, p_mask_matrix], dim = -1) # concatnation diag and proc mask matrix

        last_m_enc_mask = m_self_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        medication_self_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        
        m2d_mask_matrix = d_p_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        m2d_mask_matrix = m2d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num)
        
        dec_hidden = self.decoder(input_medication_embedding=input_medication_embs, input_medication_memory=input_medication_memory,
            input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num, max_diag_num, -1), 
            input_medication_self_mask=medication_self_mask, 
            d_mask=m2d_mask_matrix)

        score_g = self.Wo(dec_hidden) # (batch * max_visit_num, max_med_num, voc_size[2]+2)
        score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
        prob_g = F.softmax(score_g, dim=-1)
        score_c = self.copy_med(dec_hidden.view(batch_size, max_visit_num, max_med_num, -1), last_medication_embedding, last_m_mask, cross_visit_scores)
        # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)
        prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(batch_size, max_visit_num * max_med_num, -1) # (batch, max_visit_num * input_med_num, voc_size[2]+2)

        # According to the indices in last_seq_medication, add the values in score_c to score_c_to_g
        copy_source = last_medications.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)

        prob_c_to_g.scatter_add_(2, copy_source, score_c)
        prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)

        generate_prob = F.sigmoid(self.W_z(dec_hidden)).view(batch_size, max_visit_num, max_med_num, 1)
        prob =  prob_g * generate_prob + prob_c_to_g * (1. - generate_prob)
        prob[:, 0, :, :] = prob_g[:, 0, :, :] 
        

        return torch.log(prob)

    # def forward(self, input, last_input=None, max_len=20):
    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, ehr_adj, max_len=20):
        device = self.device
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, count, gumbel_pick_index = self.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, ehr_adj, max_len=20) 

        # Construct medications for the decoder, used for teacher forcing during the decoding process. Note that an additional dimension is added because an END_TOKEN will be generated.
        
        input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(device)    # [batch_size, seq, 1]
        input_medication = torch.cat([input_medication, medications], dim=2)      # [batch_size, seq, max_med_num + 1]
        m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float() 
        m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)

        output_logits = self.decode(input_medication, input_disease_embdding,encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory) 

        cross_visit_scores_numpy = cross_visit_scores.cpu().detach().numpy()
        return output_logits, count, gumbel_pick_index, cross_visit_scores_numpy

    def calc_cross_visit_scores(self, embedding, gumbel, visit_score_emb):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        
        max_visit_num = embedding.size(1)
        batch_size = embedding.size(0)
  
        ## attention calculation 
        diag_keys = embedding[:, :, :] # key: past and current visit 
        diag_query = embedding[:, -1: ,:] # query: current visit
        diag_scores = torch.bmm(self.linear_layer(diag_query), diag_keys.transpose(-2,-1)) / math.sqrt(diag_query.size(-1))  # attention weight
        
        diag_scores = diag_scores.squeeze(0).squeeze(0).masked_fill(gumbel == 0 ,-1e9)
        scores = F.softmax(diag_scores / self.att_tau, dim = -1)
        
        # visit-score 
        visit_keys = visit_score_emb[:, :, :]
        visit_query = visit_score_emb[:, -1:, :]
        all_visit_scores = torch.bmm(self.linear_layer(visit_query), visit_keys.transpose(-2,-1)) / math.sqrt(visit_query.size(-1))  # attention weight
        
        diag_scores_encoder = all_visit_scores.squeeze(0).squeeze(0)
       
        scores_encoder = F.softmax(diag_scores_encoder / self.att_tau, dim = -1) ## health status aware attention
        
        return scores , scores_encoder 

    def visit_selection_calculate_scores(self, visit_score_emb):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        max_visit_num = visit_score_emb.size(1)
        batch_size = visit_score_emb.size(0)
       
        visit_keys = visit_score_emb[:, :, :]
        visit_query = visit_score_emb[:, -1:, :]
        all_visit_scores = torch.bmm(self.linear_layer(visit_query), visit_keys.transpose(-2,-1)) / math.sqrt(visit_query.size(-1))  # attention weight
        
        diag_scores_encoder = all_visit_scores.squeeze(0).squeeze(0)
        
        all_scores = F.softmax(diag_scores_encoder / self.att_tau, dim = -1)
        return all_scores


    def copy_med(self, decode_input_hiddens, last_medications, last_m_mask, cross_visit_scores):
        """
        decode_input_hiddens: [batch_size, max_visit_num, input_med_num, emb_size]
        last_medications: [batch_size, max_visit_num, max_med_num, emb_size]
        last_m_mask: [batch_size, max_visit_num, max_med_num]
        cross_visit_scores: [batch_size, max_visit_num, max_visit_num]
        """
        max_visit_num = decode_input_hiddens.size(1)
        input_med_num = decode_input_hiddens.size(2)
        max_med_num = last_medications.size(2)
        
        
        copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num*input_med_num, self.emb_dim)

        attn_scores = torch.matmul(copy_query, last_medications.view(-1, max_visit_num*max_med_num, self.emb_dim).transpose(-2, -1)) / math.sqrt(self.emb_dim)
        
        med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
    
        attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

        visit_scores = cross_visit_scores.unsqueeze(0).unsqueeze(-1).repeat(1,1,max_med_num).view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
        scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
        
        row_scores = scores.sum(dim=-1, keepdim=True)
        scores = scores / row_scores    # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)
        

        return scores


class MedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead


    def forward(self, input_medication_embedding, input_medication_memory, input_disease_embdding, 
        input_medication_self_mask, d_mask): 
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        # [batch_size*visit_num, max_med_num+1, max_med_num+1]
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead, input_disease_embdding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        
        x = self.norm2(x + self._m2d_mha_block(x, input_disease_embdding, d_mask))
        x = self.norm3(x + self._ff_block(x))
        
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


    
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim) 
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DirectedGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):

        # # If edge weights are not provided, initialize them to 1
        # if edge_weight is None:
        #     edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        # Linearly transform node feature matrix.
        x = self.lin(x)

        # Calculate the normalization
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j    


class TwoLayerDirectedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TwoLayerDirectedGCN, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = ReLU()(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x

    
class DirectedGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(DirectedGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x
    
class policy_network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(policy_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)