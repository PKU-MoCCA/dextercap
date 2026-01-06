import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from torcheval.metrics.functional import binary_f1_score

from models import UNet
from datasets import MarkerDataset

import numpy as np
import os
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = UNet(output_channel=2).to(device)
    
    dataset_folder, dataset_file = os.path.join(''), 'dataset/all_in_one_data.json'

    train_loader = DataLoader(
        MarkerDataset(dataset_folder, dataset_file, size=128*500, train=True, max_num_markers=100, 
                      output_mask_image=True, output_line_mask_image=True, augment_image=True), 
        batch_size=1024, shuffle=True, num_workers=4, persistent_workers=True
    )

    val_loader = DataLoader(
        MarkerDataset(dataset_folder, dataset_file, size=1280, train=False, max_num_markers=100, 
                      output_mask_image=True, output_line_mask_image=True, augment_image=False), 
        batch_size=1024, shuffle=False, num_workers=4, persistent_workers=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    def loss_func(input: torch.Tensor, target: torch.Tensor):
        heatmap_loss = nn.functional.mse_loss(input, target*10)
        return heatmap_loss

    trainer = create_supervised_trainer(model, optimizer, loss_func, device)

    def accuracy_func(input: torch.Tensor, target: torch.Tensor):
        pred = input > 5
        targ = target > 0.5
        f1 = binary_f1_score(pred.view(-1), targ.view(-1))        
        return f1
    
    val_metrics = {
        "accuracy": Loss(accuracy_func),
        "loss": Loss(loss_func)
    }

    # train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_interval = 1
    tb_logger = TensorboardLogger(log_dir="logger/conv")
        
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
        
        tb_logger.writer.flush()


    def score_function(engine):
        return engine.state.metrics["accuracy"]


    model_checkpoint = ModelCheckpoint(
        "ckpts/0515/conv",
        n_saved=1,
        # filename_prefix="best",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    best_checkpoint = ModelCheckpoint(
        "ckpts/0515/conv",
        n_saved=3,
        filename_prefix="best",
        score_function=score_function,
        score_name="acc",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )
    
    val_evaluator.add_event_handler(Events.COMPLETED, best_checkpoint, {"model": model})
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, 
                                    {'model': model, 'optimizer': optimizer, 'trainer': trainer})


    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [
        # ("training", train_evaluator), 
        ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )
        
    # resume
    # if True:
    #     checkpoint_fp = os.path.join('checkpoint-conv', "checkpoint_100.pt")
    #     checkpoint = torch.load(checkpoint_fp, map_location=device) 
    #     Checkpoint.load_objects(to_load={
    #         'model': model, 
    #         'optimizer': optimizer, 
    #         'trainer': trainer
    #         }, checkpoint=checkpoint) 

    trainer.run(train_loader, max_epochs=400)

    tb_logger.close()