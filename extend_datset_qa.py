import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from load_dataset_disc import get_dataset_split_generator


def merge_metadata_with_dataset(dataset, metadata_df_):
    updated_samples_ = []
    for sample in tqdm(dataset):
        img_file_name = sample["img_file_name"]

        metadata_row = metadata_df_.loc[metadata_df_['img_file_name'] == img_file_name]

        if metadata_row.empty:
            print(f'{img_file_name} could not be found in the metadata CSV!')

        if not metadata_row.empty:
            for _, row in metadata_row.iterrows():
                new_sample = sample.copy()
                for col in metadata_df_.columns:
                    if col != 'img_file_name' and col in row:
                        new_sample[col] = str(row[col])
                updated_samples_.append(new_sample)

    return updated_samples_


if __name__ == '__main__':
    # Load the original dataset splits
    original_train = list(get_dataset_split_generator(split="train"))
    original_val = list(get_dataset_split_generator(split="validation"))
    original_test = list(get_dataset_split_generator(split="test"))

    # Load your CSV files
    train_df = pd.read_csv('validated_answers_cat.csv', encoding='utf-8', sep=";")
    val_df = pd.read_csv('validated_answers_val_cat.csv', encoding='utf-8', sep=";")
    test_df = pd.read_csv('validated_answers_test_cat.csv', encoding='utf-8', sep=";")

    merged_train = merge_metadata_with_dataset(original_train, train_df)
    merged_val = merge_metadata_with_dataset(original_val, val_df)
    merged_test = merge_metadata_with_dataset(original_test, test_df)
    # Convert to Hugging Face Dataset

    train_dataset = Dataset.from_list(merged_train)
    val_dataset = Dataset.from_list(merged_val)
    test_dataset = Dataset.from_list(merged_test)

    dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    # Save the dataset to disk
    dataset.save_to_disk('Data\\VQAMetaQA')
