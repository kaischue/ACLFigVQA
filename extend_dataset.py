import pandas as pd
import requests
from datasets import load_dataset, DatasetDict, Features, Image, Value, Dataset
from tqdm import tqdm

# Login using e.g. `huggingface-cli login` to access this dataset
from constants import HUGGINGFACE_CACHE_PATH
from pathlib import Path
import fitz
from pymupdf4llm import to_markdown

metadata_df = pd.read_csv("Data\\scientific_figures_pilot.csv")
papers_pdf_path = "Data\\VQAMeta\\papers\\pdfs"
papers_markdown_path = "Data\\VQAMeta\\papers\\markdown"
png_path = "Data\\VQAMeta\\png"


def download_pdf(acl_paper_id):
    pdf_path = f'{papers_pdf_path}\\{acl_paper_id}.pdf'
    if not Path(pdf_path).exists():
        url = f'https://aclanthology.org/{acl_paper_id}.pdf'
        if ".Dataset" in acl_paper_id:
            url = f'https://aclanthology.org/attachments/{acl_paper_id}.pdf'
        response = requests.get(url)
        if response.status_code == 200:
            pdf_path = f'{papers_pdf_path}\\{acl_paper_id}.pdf'
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Error downloading {url}: {response.status_code} - {response.reason}")
            return None
    return pdf_path


def extract_text_from_pdf(pdf_path):
    markdown_path = f'{papers_markdown_path}\\{Path(pdf_path).stem}.md'
    if not Path(markdown_path).exists():
        try:
            md_text = to_markdown(pdf_path)
        except ValueError:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            md_text = text
        Path(markdown_path).write_bytes(md_text.encode())
    else:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
    return md_text


def merge_metadata_with_dataset(dataset, metadata_df_):
    updated_samples_ = []
    for sample in tqdm(dataset):
        img_file_name = Path(sample["image"].filename).name

        metadata_row = metadata_df_[metadata_df_['file_name'] == img_file_name]

        if metadata_row.empty:
            print(f'{img_file_name} could not be found in the metadata csv!')

        if not metadata_row.empty:
            for col in metadata_df_.columns:
                if col != 'file_name' and col in metadata_row:
                    sample[col] = str(metadata_row[col].values[0])

            acl_paper_id = metadata_row['acl_paper_id'].values[0]
            pdf_path = download_pdf(acl_paper_id)
            if pdf_path:
                sample['pdf_text'] = extract_text_from_pdf(pdf_path)

        updated_samples_.append({
            'img_file_name': img_file_name,
            'image': sample['image'],
            'label': sample['label'],
            'caption': sample.get('caption', ''),
            'inline_reference': sample.get('inline_reference', ''),
            'metadata': sample.get('metadata', ''),
            'acl_paper_id': sample.get('acl_paper_id', ''),
            'pdf_text': sample['pdf_text'] if not metadata_row.empty and pdf_path else ''
        })

    return updated_samples_


if __name__ == '__main__':
    #ds = load_dataset("citeseerx/ACL-fig", cache_dir=HUGGINGFACE_CACHE_PATH)
    ds = load_dataset("imagefolder", data_dir="Data/VQAMeta/training_data")

    features = Features({
        'img_file_name': Value('string'),
        'image': Image(),
        'caption': Value('string'),
        'inline_reference': Value('string'),
        'metadata': Value('string'),
        'label': Value('string'),
        'acl_paper_id': Value('string'),
        'pdf_text': Value('string')
    })

    updated_splits = {}
    for split in ds.keys():
        updated_samples = merge_metadata_with_dataset(ds[split], metadata_df)
        updated_splits[split] = Dataset.from_list(updated_samples, features=features)
        print(f'finished {split} split')

    updated_dataset = DatasetDict(updated_splits)

    updated_dataset.save_to_disk('Data\\VQAMeta')
