import pandas as pd
dataset=pd.read_csv("loan_small.csv")

subset=dataset.iloc[0:3, 1:3]

subsetN=dataset[['Gender','ApplicantIncome']][0;3]

datasetT=pd.read_csv("loan_small_tsv.txt",sep='\t')