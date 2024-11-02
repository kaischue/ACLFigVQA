from datasets import load_from_disk

DATASET_PATH = 'Data\\VQAMeta'
VISUAL_LABELS = ["bar charts", "boxplots", "confusion matrix", "Line graph_chart",
                 "pareto", "pie chart", "scatter plot", "venn diagram"]

def iterate_dataset_split(ds_split, only_visual):
    for s in ds_split:
        if only_visual and s["label"] in VISUAL_LABELS:
            yield s

def get_dataset_split_generator(dataset_path=DATASET_PATH, split="train", only_visual=True):
    ds = load_from_disk(dataset_path)
    ds_split = ds[split]
    return iterate_dataset_split(ds_split, only_visual)
