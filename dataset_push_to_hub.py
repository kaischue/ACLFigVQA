from datasets import load_from_disk

if __name__ == '__main__':
    # Load the dataset from disk
    merged_dataset = load_from_disk('Data\\VQAMetaQA')

    merged_dataset.push_to_hub("kaischue/ACLFigVQA",private=True)