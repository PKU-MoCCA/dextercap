import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from models import BlockNet
from datasets import BlockCodeDataset

import numpy as np

from typing import Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset_folder, dataset_file = os.path.join(''), 'dataset/mocap0515/label_files.json'
    
    dataset = BlockCodeDataset(dataset_folder, dataset_file, size=128*500, train=True, augment_image=True)

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8, persistent_workers=True)

    val_loader = DataLoader(
        BlockCodeDataset(dataset_folder, dataset_file, size=1280, train=False, augment_image=False), 
        batch_size=512, shuffle=False, num_workers=4, persistent_workers=True
    )
    
    model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    def loss_func(input: Tuple[torch.Tensor], target: Tuple[torch.Tensor]):
        label_0, label_1, blk_dir = input
        label_0_gt, label_1_gt, blk_dir_gt = target        
        label_0_loss = nn.functional.cross_entropy(label_0, label_0_gt)
        label_1_loss = nn.functional.cross_entropy(label_1, label_1_gt)
        blk_dir_loss = nn.functional.cross_entropy(blk_dir, blk_dir_gt)
        
        return label_0_loss + label_1_loss + blk_dir_loss
    
    def accuracy_func(input: torch.Tensor, target: torch.Tensor):
        pred = input.argmax(dim=-1)
        targ = target.argmax(dim=-1)
        
        return (pred == targ).sum() / pred.numel()

    # criterion = nn.MSELoss()
    criterion = loss_func

    trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {
        "accuracy": Loss(accuracy_func),
        "loss": Loss(nn.functional.cross_entropy)
    }

    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_interval = 1
    tb_logger = TensorboardLogger(log_dir="logger/block")
        
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     train_evaluator.run(train_loader)
    #     metrics = train_evaluator.state.metrics
    #     print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")        
    #     tb_logger.writer.flush()


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")
        
        tb_logger.writer.flush()


    def score_function(engine):
        return engine.state.metrics["accuracy"]


    model_checkpoint = ModelCheckpoint(
        "ckpts/0515/block",
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    best_checkpoint = ModelCheckpoint(
        "ckpts/0515/block",
        n_saved=3,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
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
    #     checkpoint_fp = os.path.join('checkpoint-block', "checkpoint_100.pt")
    #     checkpoint = torch.load(checkpoint_fp, map_location=device) 
    #     Checkpoint.load_objects(to_load={
    #         'model': model, 
    #         'optimizer': optimizer, 
    #         'trainer': trainer
    #         }, checkpoint=checkpoint) 

    trainer.run(train_loader, max_epochs=300)

    tb_logger.close()