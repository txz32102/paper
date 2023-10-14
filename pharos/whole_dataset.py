import torch 
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
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
        if 'Target Development Level' in self.dataframe.columns:
            Tclin_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tclin']
        elif 'label' in self.dataframe.columns:
            Tclin_df = self.dataframe[self.dataframe['label'] == 1]
        return Tclin_df
    
    def Tbio(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tbio_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tbio']
        elif 'label' in self.dataframe.columns:
            Tbio_df = self.dataframe[self.dataframe['label'] == -1]
        return Tbio_df
    
    def Tdark(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tdark_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tdark']
        elif 'label' in self.dataframe.columns:
            Tdark_df = self.dataframe[self.dataframe['label'] == -2]
        return Tdark_df
    
    def Tchem(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tchem_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tchem']
        elif 'label' in self.dataframe.columns:
            Tchem_df = self.dataframe[self.dataframe['label'] == -3]
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

def balanced_data(df):
    df_Tclin = pharos(df).Tclin()
    df_Tbio = pharos(df).Tbio()
    df_Tchem = pharos(df).Tchem()
    df_Tdark = pharos(df).Tdark()
    train_df = pd.concat([df_Tclin.iloc[0:300], df_Tdark.iloc[0:300]], ignore_index=True)
    test_df = pd.concat([df_Tclin.iloc[300:400], df_Tdark.iloc[300:400]], ignore_index=True)
    return train_df, test_df

def data_fit(train_df, test_df):
    np.random.seed(42)
    X_train = train_df.iloc[:, 1:321]
    y_train = train_df['label']
    y_train[y_train != 1] = 0
    X_test = test_df.iloc[:, 1:321]
    y_test = test_df['label']
    y_test[y_test != 1] = 0
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Shuffle the test data
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train, y_train, X_test, y_test


def train_and_test():
    # Check if the '/content' directory exists (for Colab)
    if os.path.exists('/content'):
        # Check if '/content/drive/MyDrive' exists (typical location in Colab)
        if os.path.exists('/content/drive/MyDrive'):
            os.chdir('/content/drive/MyDrive')
            df = pd.read_csv('esm2_320_dimensions_with_labels.csv') 
        else:
            # Change to '/home/musong/Desktop' if '/content/drive/MyDrive' doesn't exist
            os.chdir('/home/musong/Desktop')
            df = pd.read_csv('/home/musong/Desktop/esm2_320_dimensions_with_labels.csv') 
    else:
        # Change to '/home/musong/Desktop' if '/content' doesn't exist
        os.chdir('/home/musong/Desktop')
        df = pd.read_csv('/home/musong/Desktop/esm2_320_dimensions_with_labels.csv') 

    train_df, test_df = balanced_data(df)
    X_train, y_train, X_test, y_test = data_fit(train_df, test_df)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # normalize X to 0-1 range 
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test