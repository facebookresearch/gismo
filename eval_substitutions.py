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

        config.dataset.ablation.alternate_substitution_set = ""
        config.dataset.ablation.with_substitutions = False

        config.name = base_name + "full_ground_truth"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_ground_truth
        schedule_job(config, training_mode=TrainingMode.EVALUATE)

        config.name = base_name + "full_predictions"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_predictions
        schedule_job(config, training_mode=TrainingMode.EVALUATE)

        config.dataset.ablation.with_substitutions = True

        config.name = base_name + "predictions"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_predictions
        schedule_job(config, training_mode=TrainingMode.EVALUATE)

        config.name = base_name + "ground_truth"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_ground_truth
        schedule_job(config, training_mode=TrainingMode.EVALUATE)

        config.name = base_name + "gt_substitutions"
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_substitutions
        schedule_job(config, training_mode=TrainingMode.EVALUATE)

        for i in [1, 2, 3, 4, 5]:
            config.name = base_name + f"pred_substitutions_{i}"
            config.dataset.ablation.alternate_substitution_set = f"/checkpoint/qduval/inversecooking2.0/gist_output/val_output_{i}.pkl"
            schedule_job(config, training_mode=TrainingMode.EVALUATE)

        for i in [1, 2, 3, 4, 5]:
            config.name = base_name + f"rand_substitution_{i}"
            config.dataset.ablation.alternate_substitution_set = f"/checkpoint/qduval/inversecooking2.0/gist_output/val_output_rand_{i}.pkl"
            schedule_job(config, training_mode=TrainingMode.EVALUATE)

        '''
        config.name = base_name + "ground_truth_no_image"
        config.dataset.ablation.gray_images = True
        config.ingr_teachforce.test = IngredientTeacherForcingFlag.use_ground_truth
        config.dataset.ablation.alternate_substitution_set = ""
        schedule_job(config, training_mode=TrainingMode.EVALUATE)
        '''


if __name__ == "__main__":
    main()
