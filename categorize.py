import numpy as np
import pandas as pd
from pandas import DataFrame

from gemini_api import GeminiModel
from load_dataset_disc import get_dataset_split_generator


def write_rows_to_csv(csv_path, rows, mode='a'):
    with open(csv_path, mode=mode, newline='\n', encoding="utf-8") as f:
        for row in rows.split("\n"):
            if not "img_file_name" in row:
                f.write(row + "\n")


def clean_context(df: DataFrame, path):
    df['context'] = df['context'].apply(lambda x: np.nan if len(str(x)) < 5 else x)
    df.to_csv(path, index=False, sep=";")


if __name__ == '__main__':
    model = GeminiModel()

    # dataset = iter(get_dataset_split_generator(only_visual=True, split="train"))
    # dataset = get_dataset_split_generator(only_visual=True, split="validation")
    dataset = get_dataset_split_generator(only_visual=True, split="test")

    validated_csv_path = 'validated_answers.csv'
    validated_df = pd.read_csv(validated_csv_path, encoding='utf-8')

    validated_df.drop(columns=['answer_german', 'answer_english', 'chain_of_thought', 'chain_of_thought2', 'confidence',
                               'confidence_explanation'], inplace=True)
    # drop rows of filtered out questions
    validated_df.drop(validated_df[(validated_df['corrected_answer_german'] == 'True') & (validated_df['corrected_answer_english'] == 'True')].index, inplace=True)

    validated_answers_cat_path = "validated_answers_test_cat.csv"
    validated_df_cat = pd.read_csv(validated_answers_cat_path, encoding='utf-8', sep=";", )

    try:
        while True:
            ds_sample = next(dataset)

            if not validated_df_cat.loc[validated_df_cat['img_file_name'] == ds_sample["img_file_name"]].empty:
                continue

            sample_df = validated_df[validated_df['img_file_name'] == ds_sample['img_file_name']]
            if sample_df.empty:
                continue
            sample_df_csv = sample_df.to_csv(sep=";")

            response = model.categorize_qa(ds_sample, sample_df_csv)
            print(response.usage_metadata)

            new_rows = response.text
            write_rows_to_csv(validated_answers_cat_path, new_rows)

    except StopIteration:
        pass
