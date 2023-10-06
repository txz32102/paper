import pandas as pd
import torch
from torch.utils.data import Dataset
import esm
import numpy as np
import csv
import os
from tqdm import tqdm

class pharos(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): Pandas DataFrame containing your data.
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        
        # Extract data and label from the DataFrame
        data = sample['sequence']  # Replace 'data_column_name' with the actual name of your data column
        label = sample['Target Development Level']  # Replace 'label_column_name' with the actual name of your label column
        
        # Convert data and label to PyTorch tensors (you can apply transforms here if needed)
        
        if self.transform:
            UniProt_id = sample['UniProt']
            data = [(UniProt_id, data)]
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results
            batch_tokens = batch_converter(data)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens[2], repr_layers=[33], return_contacts=True)
            data = results
            if label == 'Tclin':
                label = torch.tensor([1])
            else:
                label = torch.tensor([0])
        
        return data, label

    def Tclin(self):
        Tclin_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tclin']
        return Tclin_df
    
    def Tbio(self):
        Tbio_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tbio']
        return Tbio_df
    
    def Tdark(self):
        Tdark_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tdark']
        return Tdark_df
    
    def Tchem(self):
        Tchem_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tchem']
        return Tchem_df
    
    def sequence_len(self):
        LEN = self.dataframe['SequenceColumn'].apply(lambda x: len(x))
        return LEN
    
    def get_lowest_500_sequences(self):
        # Calculate the length of each sequence
        self.dataframe['SequenceLength'] = self.dataframe['sequence'].apply(lambda x: len(x))
        
        # Sort the DataFrame by SequenceLength in ascending order
        sorted_df = self.dataframe.sort_values(by='SequenceLength', ascending=True)
        
        # Select the lowest 500 sequences
        lowest_500_df = sorted_df.head(500)
        
        # Drop the 'SequenceLength' column if you don't need it in the final DataFrame
        lowest_500_df = lowest_500_df.drop(columns=['SequenceLength'])
        
        # Reset the index
        lowest_500_df = lowest_500_df.reset_index(drop=True)
        
        return lowest_500_df
    
    def vector_for_esm_embedding(self):
        UniProt_id = self.dataframe['UniProt'].to_list()
        sequence = self.dataframe['sequence'].to_list()
        res = []
        for i in range(len(UniProt_id)):
            temp = (UniProt_id[i], sequence[i])
            res.append(temp)
        return res

def esm_embeddings(peptide_sequence_list):
    # model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()  # disables dropout for deterministic results
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    ## batch tokens are the embedding results of the whole data set
    batch_tokens = batch_tokens.to(device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        # Here we export the last layer of the EMS model output as the representation of the peptides
        # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)  
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append((peptide_sequence_list[i][0], token_representations[i, 1 : tokens_len - 1].mean(0)))
    return sequence_representations

def to_csv(data, filename="output.csv"):
    # Check if the file already exists
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file is empty
        if not file_exists:
            header = ["UniProt_id"] + [str(i) for i in range(1, 321)]
            writer.writerow(header)
        
        for i in range(len(data)):
            file.write(f'{data[i][0]}')
            for j in range(320):
                file.write(f',{data[i][1][j]}')
            file.write('\n')

def min_batch(df_sorted, start_, end_):
    df = df_sorted.iloc[start_:end_]
    df = pd.DataFrame(df)
    data = pharos(df).get_lowest_500_sequences()
    data = pharos(data).vector_for_esm_embedding()
    batch_size = 10
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    with tqdm(total=total_batches, desc="Processing Batches") as pbar:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            embeddings_data = esm_embeddings(batch_data)
            to_csv(embeddings_data, "output.csv")
            pbar.update(1)
    pbar.clear()


def main():
    if os.path.exists('output.csv'):
        os.remove('output.csv')
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv('third_merge.csv')
    df['sequence_length'] = df['sequence'].apply(len)
    df_sorted = df.sort_values(by='sequence_length', ascending=True)
    batch_size = 500

    for epoch in range((len(df_sorted) - 3000) // batch_size):
        print(f"{epoch + 1} epoch(s)")
        start_ = 0 + batch_size * epoch
        end_ = start_ + batch_size
        df_data = df_sorted.iloc[start_:end_]
        df_data = pd.DataFrame(df_data)
        min_batch(df_data, 0, 500)