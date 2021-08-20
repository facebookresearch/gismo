import hydra
from recommender_gcn import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """
    Run the distributed training on the selected configuration.
    """
    print(cfg)
    trainer = Trainer()
    print("Trainer Initiated")
    trainer.train_recommender_gcn(cfg)


if __name__ == "__main__":
    main()
