# Predictive Interpretable Neural Network for Druggability (PINNED)

## How to get started?

You can easliy enter the directory and run the python script, you are supposed to run the script at the root folder, namely ./paper, all the pathes are tested at the folder.It's quite easy to run the script to see the model performance.All is contained in one python script. Usually, .ipynb is used for model training, while .py is used for reference.

`data` folder is a special folder, which containes the data used by other paper and including the preprocessing script used for `this paper` in other directory for comparing. Some data is more than 25MB, you may find it in my google colab.

Some folder, for example ./drugfinder/plot, this folder is used to plot the graph used in thesis and not needed for understanding the work unless you want to see the entire picture of this work.

Dependencies can be seen in requirements.txt, almost all the code is written in python,except some latex code for plot and paper formatting and matlab code for plotting nicer picture than matplotlib.

Happy hacking!

## Directory

- &#128462; `fpocket_pipeline.ipynb`

- &#128462; `feature_processing.ipynb`

- &#128462; `PINNED_model.ipynb`

- &#128193; `raw_data`
  
  - &#128462; `all_proteins.csv`
  - &#128462; `dezso_features.csv`
  - &#128462; `fpocket_output.csv`
  - &#128462; `gdpc_10-14-22.csv`
  - &#128462; `go_components_10-14-22.csv`
  - &#128462; `go_functions_10-14-22.csv`
  - &#128462; `go_processes_10-14-22.csv`
  - &#128462; `paac_10-14-22.csv`

- &#128193; `processed_data`
  
  - &#128462; `bio_func_names.csv`
  - &#128462; `localization_names.csv`
  - &#128462; `network_info_names.csv`
  - &#128462; `seq_and_struc_names.csv`

- &#128462; `README.md`

## Authors

## External software

Rights to AlphaFold and fpocket are governed by their respective licenses

- [AlphaFold pdb models](https://ftp.ebi.ac.uk/pub/databases/alphafold/)
- [fpocket (2020) source code](https://github.com/Discngine/fpocket)
- [data source](https://pharos.nih.gov/) 
- [drugbank](https://go.drugbank.com/releases/latest)
