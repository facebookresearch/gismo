from hydra.experimental import compose, initialize

from inv_cooking.scheduler import RawConfig
from inv_cooking.training.trainer import load_data_set


def load_config():
    with initialize(config_path="../conf"):
        cfg_init = compose(
            config_name="config",
            overrides=[
                "task=ingrsubs",
                "checkpoint=baharef",
                "dataset.loading.batch_size=1",
                "dataset.pre_processing.save_path=../preprocessed_data",
            ],
        )
        configurations = RawConfig.to_config(cfg_init)
    return configurations[0]


def load_data():
    cfg = load_config()
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    return train_dataloader, val_dataloader, test_dataloader, data_module


def ids_to_words(list_ing, vocab):
    res = []
    for ing in list_ing:
        ing_word = vocab.idx2word[ing]
        if ing_word not in ["<end>", "<pad>"]:
            res.append(ing_word)
    return res


def get_vocabs(data_module):
    vocab_ing = data_module.dataset_train.ingr_vocab
    vocab_title = data_module.dataset_train.title_vocab
    return vocab_ing, vocab_title
