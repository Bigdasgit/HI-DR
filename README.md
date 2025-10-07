# HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation

> [!IMPORTANT]
> **Code was updated on Aug 8.**

This repository provides a reference implementation of HI-DR as described in the following paper:

> HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation
> Taeri Kim, Jiho Heo, Hyunjoon Kim, and Sang-Wook Kim
> The 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025).



### Overview of HI-DR

![image](https://github.com/user-attachments/assets/cf536c7d-c076-4345-ab75-2e4c6ef3754f)


### Authors

- Taeri Kim ([taerik@hanyang.ac.kr](mailto:taerik@hanyang.ac.kr))
- Jiho Heo ([linda0123@hanyang.ac.kr](mailto:linda0123@hanyang.ac.kr))
- Hyunjoon Kim ([hyunjoonkim@hanyang.ac.kr](mailto:hyunjoonkim@hanyang.ac.kr))
- Sang-Wook Kim ([wook@hanyang.ac.kr](mailto:wook@hanyang.ac.kr))



### Requirements

The code has been tested running under Python 3.10.6. The required packages are as follows:

- pandas: 1.5.1

- dill: 0.3.6

- torch: 1.8.0+cu111

- rdkit: 2022.9.1

- scikit-learn: 1.1.3

- numpy: 1.23.4

- install other packages if necessary

  ```
  pip install rdkit-pypi
  pip install scikit-learn, dill, dnc
  pip install torch
  
  pip install [xxx] # any required package if necessary, maybe do not specify the version
  ```

  

### Dataset Preparation

Change the path in processing_iii.py and processing_iv.py processing the data to get a complete records_final.pkl.
For the MIMIC-III and -IV datasets, the following files are required:
(Here, we do not share the MIMIC-III and -IV datasets due to reasons of personal privacy, maintaining research standards, and legal considerations.)

- MIMIC-III
  - PRESCRIPTIONS.csv
  - DIAGNOSES_ICD.csv
  - PROCEDURES_ICD.csv
  - D_ICD_DIAGNOSES.csv
  - D_ICD_PROCEDURES.csv
- MIMIC-IV
  - prescriptions2.csv
  - diagnoses_icd2.csv
  - procedures_icd2.csv
  - atc32SMILES.pkl
  - ndc2atc_level4.csv (We provide a sample of this file due to size constraints.)
  - ndc2rxnorm_mapping.txt
  - drug-atc.csv
  - drugbank_drugs_info.csv (We provide a sample of this file due to size constraints.)
  - drug-DDI.csv (We provide a sample of this file due to size constraints.)

### processing file

- run data processing file

  ```
  python processing_iii.py
  python processing_iv.py
  ```

- run ddi_mask_H.py

  ```
  python ddi_mask_H.py
  ```

The processed files are already stored in the directories "data/mimic-iii" and "data/mimic-iv".

### run the code

- Navigate to the directory where the file "HEIDR_main.py" is located and execute the following.

- Usage:

  ```
  python HEIDR_main.py
  ```

- optional arguments:

  ```
    -h, --help            show this help message and exit
    --Test                test mode
    --model_name MODEL_NAME
                          model name
    --resume_path RESUME_PATH
                          resume path
    --lr LR               learning rate
    --batch_size BATCH_SIZE
                          batch_size
    --emb_dim EMB_DIM     embedding dimension size
    --max_len MAX_LEN     maximum prediction medication sequence
    --beam_size BEAM_SIZE
                          max num of sentences in beam searching
    --topk TOPK           hyperparameter top-k
    --gumbel_tau GUMBEL_TAU
                          hyperparameter gumbel_tau
    --att_tau ATT_TAU     hyperparameter att_tau
    --final_top_idx_data_type_train FINAL_TOP_IDX_DATA_TYPE_TRAIN
                          final_top_idx_data_type_train path
    --final_top_idx_data_type_eval FINAL_TOP_IDX_DATA_TYPE_EVAL
                          final_top_idx_data_type_eval path
    --final_top_idx_data_type_test FINAL_TOP_IDX_DATA_TYPE_TEST
                          final_top_idx_data_type_test path
    --ehr_graph EHR_GRAPH
                          ehr_graph path
  ```
