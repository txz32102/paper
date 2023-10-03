import pandas as pd
import torch
from torch.utils.data import Dataset
import esm

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
    