import pandas as pd
import numpy as np

df = pd.read_csv("data/pharos/third_merge.csv")
df_sorted = df.sort_values(by="Target Development Level", ascending=False)

with open("data/pharos/pharos.fasta", "w") as f:
    for index, row in df_sorted.iterrows():
        uniProt_id = row["UniProt"]
        dev_level = row["Target Development Level"]
        sequence = row["sequence"]

        # Write FASTA entry
        f.write(f">{uniProt_id}|{dev_level}\n{sequence}\n")
