import kaggle
import os

def download_kaggle_dataset(dataset_name, path='data'):
    """
    Downloads a Kaggle dataset to the specified path.

    Parameters:
    - dataset_name: str, the name of the Kaggle dataset.


    """
    if not os.path.exists(path):
        os.makedirs(path)

    print(f"Downloading dataset '{dataset_name}' to '{path}'.")
    kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
    print(f"Dataset '{dataset_name}' downloaded and extracted to '{path}'.")

    print("All datasets downloaded successfully.")
    print(f"Data is stored in {out_dir}")



if __name__ == "__main__":

    output_data = ['enwiki_namespace_0', 'frwiki_namespace_0']
    dataset = "wikimedia-foundation/wikipedia-structured-contents"
    out_dir = "data"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Output directory created at {out_dir}.")
    data_exists = True in [os.path.exists(_data_dir) for _data_dir in output_data]

    if not data_exists:
        print("Data already exists, skipping download.")
    elif input('Should we start the download [y/n]: ').strip().lower() == 'y':
        download_kaggle_dataset(dataset_name=dataset, path=out_dir)

