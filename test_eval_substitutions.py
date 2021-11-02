import hydra

from inv_cooking.config.config import IngredientTeacherForcingFlag
from inv_cooking.scheduler import RawConfig
from inv_cooking.scheduler.scheduler import TrainingMode, schedule_job


@hydra.main(config_path="conf", config_name="config")
def main(cfg: RawConfig) -> None:
    """
    Run all the combinations of ingredients substitutions for the paper
    """
    for config in RawConfig.to_config(cfg):
        base_name = config.name + "_use_"
        config.dataset.ablation.alternate_substitution_set
        config.dataset.ablation.alternate_substitution_set = ""

        config.dataset.ablation.with_substitutions = True

        config.name = base_name + "ground_truth"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_ground_truth
        schedule_job(config, training_mode=TrainingMode.EVALUATE)


if __name__ == "__main__":
    main()
