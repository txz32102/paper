from pssm import *
import pandas as pd
import os


# you should place the pssm files in a folder, then it will output the 1200 dimensional matrix
def pssm_1200d(pssm_path, output_csv):
    data = []
    header = ["id", "label"] + list(range(1, 1201))
    for file_name in os.listdir(pssm_path):
        if os.path.isfile(os.path.join(pssm_path, file_name)):
            file_path = os.path.join(pssm_path, file_name)
            pssm_data = read_pssm_matrix(file_path)

            try:
                dpc_pssm_400 = dpc_pssm(pssm_data)
                k_separated_bigrams_pssm_400 = k_separated_bigrams_pssm(pssm_data)
                s_fpssm_400 = s_fpssm(pssm_data)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

            combined_features = np.concatenate(
                (dpc_pssm_400, k_separated_bigrams_pssm_400, s_fpssm_400), axis=None
            )
            row = [file_name] + [0] + list(combined_features)
            data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=header)

    # Save to CSV
    df.to_csv(output_csv, index=False)
