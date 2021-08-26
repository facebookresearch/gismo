import copy
import os

import torch


def create_output_dir(base_dir, conf):
    """
    If an output dir is given, then make sure it exists. Otherwise, create one based on time stamp.
    """
    params = "lr_" + str(conf.lr)
    params += "_w_decay_" + str(conf.w_decay)
    params += "_hidden_" + str(conf.hidden)
    params += "_emb_d_" + str(conf.emb_d)
    params += "_dropout-" + str(conf.dropout)
    params += "_nlayers_" + str(conf.nlayers)
    params += "_nr_" + str(conf.nr)
    params += "_margin_" + str(conf.margin)

    output_dir = os.path.join(base_dir, params)

    if not os.path.exists(os.path.join(base_dir, output_dir)):
        os.makedirs(os.path.join(base_dir, output_dir))
        print("Created output directory {}".format(output_dir))
    return output_dir


def load_saved_models(output_dir, model, opt):
    """
    Find the last saved model in the output_dir and load it.
    Load also the best_model
    If no model found in the output_dir or if the dir does not exists
    initialize the model randomly.
    Sets self.model, self.best_model
    """
    try:
        content = torch.load(os.path.join(output_dir, "last_model.chkpnt"))
        model.load_state_dict(content["checkpoint"])
        model.epoch = content["epoch"]
        model.epoch.requires_grad = False
        opt.load_state_dict(torch.load(os.path.join(output_dir, "last_opt.chkpnt")))
        print("*** LOADING FROM EPOCH {} ****".format(model.epoch.cpu().item()))
    except:
        print(
            "*** NO SAVED MODEL FOUND in {}. LOADING FROM SCRATCH ****".format(
                output_dir
            )
        )
        # Initilize the model

    try:
        # Load the best model
        best_model_path = os.path.join(output_dir, "best_model.chkpnt")
        best_model = copy.deepcopy(model)
        content = torch.load(best_model_path)
        best_model.load_state_dict(content["checkpoint"])
        best_model.epoch = content["epoch"]
        best_model.mrr = content["mrr"]
        print(
            "*** Loading the model with the best MRR {} from epoch {} ****.".format(
                best_model.mrr.cpu().item(), best_model.epoch.cpu().item()
            )
        )
    except:
        print("*** NO BEST MODEL FOUND in {} ****".format(output_dir))
        best_model = None

    return model, opt, best_model


def save_model(model, opt, output_dir, is_best_model=False):
    """
    Save the model state to the output folder.
    If is_best_model is True, then save the model also as best_model.chkpnt
    """
    if is_best_model:
        torch.save(
            {"checkpoint": model.state_dict(), "epoch": model.epoch, "mrr": model.mrr},
            os.path.join(output_dir, "best_model.chkpnt"),
        )
        print("### Saving the BEST MODEL ###")
    else:
        model_name = "last_model.chkpnt"
        opt_name = "last_opt.chkpnt"
        # print("### Saving the model and optimizer from epoch {} ###".format(model.epoch.cpu().item()))

        torch.save(
            {"checkpoint": model.state_dict(), "epoch": model.epoch},
            os.path.join(output_dir, model_name),
        )
        torch.save(opt.state_dict(), os.path.join(output_dir, opt_name))
