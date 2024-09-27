from datasets import load_from_disk
import matplotlib.pyplot as plt

dataset_path = 'Data\\VQAMeta'

if __name__ == '__main__':
    ds = load_from_disk(dataset_path)
    for sample in ds["train"]:
        img = sample["image"]
        # plt.imshow(img)
        # plt.show()
