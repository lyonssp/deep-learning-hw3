import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.metrics import AccuracyMetric

from .models import load_model, save_model
from .datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    model_name = "classifier"
    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "classification_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="aug"
    )
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    start = time.perf_counter()
    total_training_time = 0

    train_accuracy = AccuracyMetric()
    val_accuracy = AccuracyMetric()

    # training loop
    for epoch in range(num_epoch):
        train_accuracy.reset()
        val_accuracy.reset()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(img)
            train_accuracy.add(pred, label)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                pred = model(img)
                val_accuracy.add(pred, label)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(train_accuracy.compute()["accuracy"])
        epoch_val_acc = torch.as_tensor(val_accuracy.compute()["accuracy"])

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
        elapsed = time.perf_counter() - start
        total_training_time += elapsed
        print(f"Average time per epoch: {elapsed / (epoch + 1):.2f} seconds")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))