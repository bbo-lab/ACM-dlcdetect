import numpy as np

def load_labels(labels_file) -> dict:
    labels = np.load(labels_file, allow_pickle=True)['arr_0'][()]

    if "version" in labels:  # New style labels file
        labels = labels["labels"]

    return labels

def config_from_module(module):
    config = {}
    for value in dir(module):
        config[value] = getattr(module, value)

    if "feat_list" not in config or config["feat_list"] is None:
        labels = load_labels(config["filePath_labels"])
        config["feat_list"] = list(labels.keys())

    return config

def flatten(l):
    return [item for sublist in l for item in sublist]
def get_size_from_reader(reader):
    return reader.get_meta_data(0)["size"]
