import json
import kaggle
import os

from tqdm import tqdm

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


def load_dir(data_dir=None, n_files=None):
    """
    Loads data from the specified directory.

    Parameters:
    - data_dir: str, the directory containing the data files.

    Returns:
    - List of data files in the directory.
    """
    print(data_dir)
    if data_dir and not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    # Cycle through the directory and load JSON files up until `n_files`
    print(f"Loading data from {data_dir}...")
    for k, fname in tqdm(enumerate(os.listdir(data_dir))):
        if fname.endswith('.jsonl') and (n_files is None or k > n_files):
            print(f"Loading file: {fname}")

    return {
        fname: load_json_file(os.path.join(data_dir, fname))
        for k, fname in tqdm(enumerate(os.listdir(data_dir)))
        if fname.endswith('.jsonl') and (n_files is None or k > n_files)
    }


def load_json_file(json_filename=None):
    """
    Loads data from the specified directory.

    Parameters:
    - data_dir: str, the directory containing the data files.

    Returns:
    - List of data files in the directory.
    """
    if json_filename and not os.path.exists(json_filename):
        raise FileNotFoundError(f"The file {json_filename} does not exist.")

    with open(json_filename) as json_in:
        return [json.loads(line) for line in json_in]


if __name__ == "__main__":

    output_data = ['enwiki_namespace_0', 'frwiki_namespace_0']
    dataset = "wikimedia-foundation/wikipedia-structured-contents"
    data_dir = "data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Output directory created at {data_dir}.")

    data_exists = [os.path.exists(_data_dir) for _data_dir in output_data]
    data_exists = True in data_exists

    if not data_exists:
        print("Data already exists, skipping download.")
    elif input('Should we start the download [y/n]: ').strip().lower() == 'y':
        download_kaggle_dataset(dataset_name=dataset, path=data_dir)

    # Load the data
    n_files = 10  # Number of files to load, set to None to load all
    loaded_data = {}
    for lang_dir in output_data:
        if lang_dir not in loaded_data:
            loaded_data[lang_dir] = {}

        # Check if the directory exists and is a directory
        dirlang = os.path.join(data_dir, lang_dir)
        langdir_exists = os.path.exists(dirlang)
        langdir_isdir = os.path.isdir(dirlang)

        print(
            f"Checking directory: {dirlang} "
            f"(Exists: {langdir_exists}, IsDir: {langdir_isdir})"
        )

        # If the directory exists and is a directory, load the data
        if langdir_exists and langdir_isdir:
            loaded_data[lang_dir] = load_dir(data_dir=dirlang, n_files=n_files)
        else:
            print(f"Directory {dirlang} does not exist or is not a directory.")
