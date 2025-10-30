#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Date]
@author: [Your Name]
Description: Predict gene essentialities based on GSVA score using the GATDep model
"""

# ========== Import necessary modules ==========
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GeneDependencyGAT import GeneDependencyGAT


# ========== 1. Load Gene Interaction Network ==========
def get_edge_index():
    """
    Load gene-gene interaction network and return edge_index tensor.
    读取基因相互作用网络，返回 PyTorch 图结构的 edge_index。
    """
    ppi_df = pd.read_csv("/home/fan_qiangqiang/home_bk/gnn/data//ggi_id.txt", sep=" ")
    edges = ppi_df[['g1_id', 'g2_id']].values
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    return edge_index


# ========== 2. Load Gene Sets ==========
def get_genesets():
    """
    Load gene set dictionary (geneset → ID mapping).
    读取基因集与其索引编号的映射关系。
    """
    with open('/home/fan_qiangqiang/home_bk/gnn/data/genesets_select_6k.json', 'r', encoding='utf-8') as file:
        geneset2ID = json.load(file)
    return geneset2ID

# ========== 2.1 Load Genes ==========
def get_genes():
    """
    Load gene set dictionary (gene → ID mapping).
    读取基因与其索引编号的映射关系。
    """
    with open('/home/fan_qiangqiang/home_bk/gnn/data/genes2ID_2k.json', 'r', encoding='utf-8') as file:
        gene2ID = json.load(file)
    return gene2ID

# ========== 3. Ensure All Gene Sets Exist in GSVA Matrix ==========
def get_rows_with_fallback(df, row_names):
    """
    Align dataframe rows with a given list, filling missing ones with zeros.
    对输入DataFrame进行重索引，缺失的基因集用0填充。
    """
    existing = df.reindex(row_names).fillna(0)
    return existing


# ========== 4. Construct Gene Features per Sample ==========
def get_gene_features_per_sample(sample_index, gsva_scores_np, genes2ID_2k):
    """
    Construct per-sample gene features using GSVA scores and gene-set adjacency.
    基于GSVA打分与基因-基因集邻接矩阵，生成每个样本的基因特征矩阵。
    
    Args:
        sample_index (int): 样本索引 (sample index/id)
        gsva_scores_np (ndarray): GSVA score(samples × genesets)
        genes2ID_2k (dict): 基因到ID映射字典 (gene index/id)
    """
    gene_geneset = np.load("/home/fan_qiangqiang/home_bk/gnn/data//adj_gene_geneset.npy")  # adjacency matrix gene <-> geneset
    tmp = []
    for gene in genes2ID_2k.keys():
        tmp.append(gene_geneset[genes2ID_2k[gene], :] * gsva_scores_np[sample_index])
    return np.array(tmp, dtype=np.float32)


# ========== 5. Model Prediction Function ==========
def model_predict(gsva_file, model, edge_index, geneset2ID):
    """
    Perform prediction using trained GATDep model.
    使用训练好的GATDep模型进行预测。
    
    Args:
        gsva_file (str): 输入GSVA文件路径 (gsva file)
        model (torch.nn.Module): 训练好的模型 (trained model)
        edge_index (torch.Tensor): 图的边索引 (node-edge index)
        geneset2ID (dict): 基因集到索引映射 (genesets index/id)
    """
    # read in GSAV Score
    gsva_scores_df = pd.read_csv(gsva_file, delimiter=' ')
    result_df = get_rows_with_fallback(gsva_scores_df.T, geneset2ID.keys())
    gsva_scores_np = np.array(result_df.T, dtype='float32')  # (samples × genesets)
    
    sample_list=result_df.columns.values.tolist()

    # load gene ID 
    with open('/home/fan_qiangqiang/home_bk/gnn/data/genes2ID_2k.json', 'r', encoding='utf-8') as file:
        genes2ID_2k = json.load(file)

    # get features of genes by samples
    gene_feature_list = []
    for sample in range(gsva_scores_np.shape[0]):
        gene_feature = get_gene_features_per_sample(sample, gsva_scores_np, genes2ID_2k)
        gene_feature_list.append(gene_feature)

    #transform to PyG Data 
    query_data = [Data(x=torch.from_numpy(x_i), edge_index=edge_index) for x_i in gene_feature_list]
    data_loader = DataLoader(query_data, batch_size=5)

    # prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            all_preds.append(out.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    return y_pred,sample_list


# ========== 6. Save Output ==========
def save_results(y_pred,sample_list, gene2ID, out_file):
    """
    Save predicted results to CSV file.
    将预测结果保存为CSV文件。
    """
    #pd.DataFrame({'preds': y_pred.flatten()}).to_csv(f'{out_file}.csv', index=False)
    pd.DataFrame({
    'genes': list(gene2ID.keys()) * len(sample_list),
    'samples': [item for item in sample_list for _ in range(2350)],
    'preds':y_pred.flatten()
    }).to_csv(f'{out_file}.csv', index=False)


# ========== 7. Main Execution Logic ==========
def main():
    """
    Main function that parses arguments and runs prediction.
    主函数：解析输入参数并执行预测流程。
    """
    parser = argparse.ArgumentParser(description="Predict gene essentialities based on GSVA score")
    parser.add_argument('--gsva', type=str, required=True, help="Input GSVA score file (*.txt)")
    parser.add_argument('--out', type=str, required=True, help="Output file name (without extension)")
    args = parser.parse_args()

    # load trained GATDep Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeneDependencyGAT(in_channels=6561, hidden_channels=64, out_channels=1)
    model.load_state_dict(torch.load('/home/fan_qiangqiang/home_bk/gnn/models/best_model_v1_basedon_v4_scale_data.pt',
                                     map_location=device))

    # load required dataset
    geneset2ID = get_genesets()
    gene2ID=get_genes()
    edge_index = get_edge_index()

    # prediction
    y_pred,sample_list = model_predict(args.gsva, model, edge_index, geneset2ID)

    # saving
    save_results(y_pred,sample_list, gene2ID, args.out)


# ========== Entry Point ==========
if __name__ == "__main__":
    main()
