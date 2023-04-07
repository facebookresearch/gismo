# Graph-based Ingredient Substitution Module (GISMo)

In this folder lies the code of GISMo, a graph based neural network which handles ingredient substitution in recipes.

- [Installation](#Installation)
- [Data Preparation](#Data-preparation)
- [Reproducing Experiments](#Reproducing-experiments)
- [Model ZOO](#Model-ZOO)

## Installation

Create the conda environment:

    conda create --name inverse_cooking_gismo python=3.8

Install PyTorch dependencies:

    conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
    conda install -c conda-forge -c fvcore -c iopath fvcore iopath

Install Deep Graph Library:

    conda install -c dglteam dgl-cuda10.2==0.7.0

Install Inverse Cooking requirement:

    cd ..  # Go up up one folder outside the "gismo folder"
    pip install -e .
    cd gismo

Install the other dependencies:

    pip install -r requirements.txt --upgrade


## Data preparation


1. Download ```nodes_191120.csv``` and ```edges_191120.csv``` from [FlavorGraph](https://github.com/lamypark/FlavorGraph/tree/master/input) into ```./checkpoints/graph```.
2. Download subtitution data: [train](https://dl.fbaipublicfiles.com/gismo/train_comments_subs.pkl), [valid](https://dl.fbaipublicfiles.com/gismo/val_comments_subs.pkl), [test](https://dl.fbaipublicfiles.com/gismo/test_comments_subs.pkl), [vocab](https://dl.fbaipublicfiles.com/gismo/vocab_ingrs.pkl) into ```./checkpoints```


## Reproducing experiments

### Setup

The code under the "gismo" folder organizes the experiments in the following way:

- Experiments create a folder dedicated to a given set of hyper-parameters
- Upon training with the same set of parameters, the same folder is used, and training is restarted where it left of

The folder in which all experiments will be written to is configurable in `conf/config.yaml` under the field `base_dir`.

### Baselines

To train the best performing baseline based on lookup-table with frequency, use the following command:

    python train.py name=LTFreq setup=context-free max_context=0

You can run the other baselines in a similar fashion (here shown for the Lookup table without the frequency):

    python train.py name=LT setup=context-free max_context=0


### Training

To train the best performing model, use this command in the `gismo` folder:

```
python train.py name=GIN_MLP setup=context-full max_context=43 \
    lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 \
    nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random \
    with_titles=False with_set=True filter=False
```

This command will load the data, create a folder to hold checkpoints, run the training where it last left off (by looking into the checkpoint folder), and once the training is over, will run a pass on validation and test sets.

### Inference

Once the model has been trained, running the same command as for training will run inference on the validation and test set (the code will automatically look for the last checkpoint and skip training if training is already done).  

To run inference on your own dataset, you can replace the validation and test set to contain the data your are interested in (after making a copy of both to avoid having to run the data preparation step again).

## Model ZOO

You can find the best GisMO model [here](https://dl.fbaipublicfiles.com/gismo/best_model.chkpnt). To run inference, save the model to ```./out/lr_5e-05_w_decay_0.0001_hidden_300_emb_d_300_dropout-0.25_nlayers_2_nr_400_neg_sampling_regular_with_titels_False_with_set_True_init_emb_random_lambda_0.0_i_1_data_augmentation_False_context_emb_mode_avg_pool_avg_p_augmentation_0.5_filter_False/```


