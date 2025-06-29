import kaggle
import os

def download_kaggle_dataset(dataset_name, path='data'):
    """
    Downloads a Kaggle dataset to the specified path.

    Parameters:
    - dataset_name: str, the name of the Kaggle dataset (e.g., 'zillow/zecon').
    """
    if not os.path.exists(path):
        os.makedirs(path)

    kaggle.api.dataset_download_files(dataset_name, path=path, unzip=True)
    print(f"Dataset '{dataset_name}' downloaded and extracted to '{path}'.")


if __name__ == "__main__":
    # Example usage
    dataset = "wikimedia-foundation/wikipedia-structured-contents"
    out_dir = "data"
    download_kaggle_dataset('zillow/zecon', path='data')
    download_kaggle_dataset(dataset, path=out_dir)
    print("All datasets downloaded successfully.")
    print(f"Data is stored in {out_dir}")
