from random import shuffle

import yaml


def load_yaml_to_dict(file_path):
    with open(file_path, "r") as f:
        d = yaml.load(f, Loader=yaml.SafeLoader)
    return d


def shuffle_lists_in_dict(yaml_dict):
    for key in yaml_dict.keys():
        for k_key in yaml_dict[key]:
            shuffle(yaml_dict[key][k_key])


if __name__ == '__main__':
    qa_dict = load_yaml_to_dict("Data\\test.yaml")
    shuffle_lists_in_dict(qa_dict)
