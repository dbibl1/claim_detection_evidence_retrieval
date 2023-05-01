import random

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from objective import objective
import config


def train_cb(lr_range,
             optimizer_list,
             epoch_range,
             model_list,
             batch_size_list,
             gamma_range,
             scheduler_list,
             optuna_seed,
             df_train,
             num_trials):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    study = optuna.create_study(study_name="claim_detection",
                                direction="maximize",
                                sampler=TPESampler(seed=optuna_seed))

    func = lambda trial: objective(trial,
                                   lr_range,
                                   optimizer_list,
                                   epoch_range,
                                   model_list,
                                   batch_size_list,
                                   gamma_range,
                                   scheduler_list,
                                   df_train)

    study.optimize(func, n_trials=num_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == 'main':
    lr_range = (1e-6, 1e-3)
    optimizer_list = ["Adam", "AdamW"]
    epoch_range = (1, 5)
    model_list = ['bert', 'distilbert', 'roberta', 'albert']
    batch_size_list = [128]
    gamma_range = (0.1, 0.9)
    scheduler_list = ['step', 'exponential']
    num_trials = 20
    optuna_seed = 2

    claimbuster = Claimbuster()
    df_train = claimbuster.get_df()

    train_cb()
