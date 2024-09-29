from datasets import load_from_disk
import matplotlib.pyplot as plt

DATASET_PATH = 'Data\\VQAMeta'


def iterate_dataset_split(ds_split):
    for s in ds_split:
        yield s


def get_dataset_split_generator(dataset_path=DATASET_PATH, split="train"):
    ds = load_from_disk(dataset_path)
    ds_split = ds[split]
    return iterate_dataset_split(ds_split)
