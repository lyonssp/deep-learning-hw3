import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.metrics import DetectionMetric

from .models import load_model, save_model
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    model_name = "detector"

    if torch.cuda.is_available():
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
        "road_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        transform_pipeline="default"
    )
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    training_metrics = DetectionMetric()
    validation_metrics = DetectionMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        training_metrics.reset()

        model.train()

        for batch in train_data:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            img = batch["image"]
            track = batch["track"]
            depth = batch["depth"]

            optimizer.zero_grad()
            pred, pred_depth = model(img)
            pred_labels = pred.argmax(dim=1)

            # expand the (b, h, w) track labels to (b, 3, h, w) logits for loss calculation
            track_logits = torch.nn.functional.one_hot(track, num_classes=3).permute(0, 3, 1, 2).float()

            # expand the (b, h, w) depth labels to (b, 1, h, w) logits for loss calculation
            depth_logits = depth.softmax(dim=1).unsqueeze(1)

            # print({
            #     "img": img.shape, 
            #     "depth": depth.shape, 
            #     "depth_logits": depth.shape,
            #     "track": track.shape, 
            #     "track_logits": track_logits.shape, 
            #     "predictions": pred.shape, 
            #     "depth_predictions": pred_depth.shape
            # })
            training_metrics.add(pred_labels, track, pred_depth, depth)

            loss = .3 * ce_loss(pred, track_logits) + .7 * mse_loss(pred_depth, depth_logits)
            loss.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in train_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                img = batch["image"]
                track = batch["track"]
                depth = batch["depth"]

                pred, pred_depth = model(img)
                pred_labels = pred.argmax(dim=1)
                validation_metrics.add(pred_labels, track, pred_depth, depth) 

        # log average train and val accuracy to tensorboard
        computed_validation_metrics = validation_metrics.compute()
        epoch_train_acc = torch.as_tensor(training_metrics.compute()["accuracy"])
        epoch_val_acc = torch.as_tensor(computed_validation_metrics["accuracy"])
        epoch_iou = torch.as_tensor(computed_validation_metrics["iou"])
        epoch_abs_depth_error = torch.as_tensor(computed_validation_metrics["abs_depth_error"])
        epoch_tp_depth_error = torch.as_tensor(computed_validation_metrics["tp_depth_error"])

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)
        logger.add_scalar("accuracy", computed_validation_metrics["accuracy"], global_step)
        logger.add_scalar("iou", computed_validation_metrics["iou"], global_step)
        logger.add_scalar("abs_depth_error", computed_validation_metrics["abs_depth_error"])
        logger.add_scalar("tp_depth_error", computed_validation_metrics["tp_depth_error"])

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"accuracy={epoch_val_acc:.4f} "
                f"iou={epoch_iou:.4f} "
                f"abs_depth_error={epoch_abs_depth_error:.4f} "
                f"tp_depth_error={epoch_tp_depth_error:.4f} "
            )

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