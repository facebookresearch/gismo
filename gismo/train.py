import os

import hydra
from recommender_gcn import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """
    Run the distributed training on the selected configuration.
    """
    print("Config completion...")
    cfg.base_dir = os.path.expanduser(cfg.base_dir)
    cfg.flavorgraph_path = os.path.expanduser(cfg.flavorgraph_path)
    cfg.substitution_path = os.path.expanduser(cfg.substitution_path)
    cfg.subs_dir = os.path.expanduser(cfg.subs_dir)
    print(cfg)

    print("Trainer started...")
    trainer = Trainer()
    trainer.train_recommender_gcn(cfg)


if __name__ == "__main__":
    main()
