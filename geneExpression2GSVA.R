#!/usr/bin/env Rscript
#geneExpression2GSVA
library(data.table)
library(dplyr)
library(GSVA)
#library(argparse)
#parser <- ArgumentParser(description = "calculate GSVA score based on Gene expression by GSVA method")
#parser$add_argument("-f", "--file", required = TRUE, help = "gene expression file (row: official gene symbol, column: samples), txt/csv")
#args <- parser$parse_args()
args <- commandArgs(trailingOnly = TRUE)


#gene_expression =fread(args$file)# gene symbol(row) x sample(column)
fread(args[1]) %>% dplyr::select(-c(1)) %>% as.data.frame() -> gene_expression
fread(args[1]) %>% pull(1) -> rownames(gene_expression)

signature_go_kegg_set<-readRDS('./datasets/signature_go_kegg_set_v2.rds')#6561
gsva_score_go_kegg <- gsva(as.matrix(gene_expression), 
                           signature_go_kegg_set,
                           kcdf="Gaussian", 
                           verbose=TRUE)

t(round(gsva_score_go_kegg,digits = 3)) %>% write.table(.,'gsva_score_go_kegg.txt',quote = F)
