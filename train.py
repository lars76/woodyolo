import argparse
import logging
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model.detection_model import DetectionModel
from loss import ObjectDetectionLoss
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm
from datasets import DetectionDataset, collate_fn
from collections import defaultdict

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer, warmup_epochs, train_epochs, train_loader, last_epoch=-1
    ):
        self.warmup_steps = warmup_epochs * len(train_loader)
        self.t_total = train_epochs * len(train_loader)
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transform_pipeline(config, is_training=True):
    base_transforms = [
        v2.ToImage(),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(),
        v2.Resize(
            size=(config.image_height, config.image_width),
            antialias=True,
            interpolation=InterpolationMode.BILINEAR,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if is_training:
        augmentation_transforms = [
            v2.RandomAffine(
                degrees=config.degrees, translate=(config.translate, config.translate)
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
            ),
        ]
        return v2.Compose(base_transforms + augmentation_transforms)

    return v2.Compose(base_transforms)


def create_experiment_directory(base_dir="experiments"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def compute_metrics(predictions, ground_truths, iou_threshold=0.5, beta=2):
    """
    Compute detection metrics

    Args:
        predictions: List of dicts with keys: image_id, x0, y0, x1, y1, conf
        ground_truths: List of dicts with keys: image_id, x0, y0, x1, y1
        iou_threshold: Minimum IoU for a match between prediction and ground truth
        beta: Beta value for F-score calculation

    Returns:
        dict: Dictionary containing metrics:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f_score: F-beta score
            - true_positives: Number of correct detections
            - false_positives: Number of incorrect detections
            - false_negatives: Number of missed detections
            - num_ground_truths: Total number of ground truth boxes
    """

    def calculate_iou_matrix(pred_boxes, gt_boxes):
        # Compute IoU matrix between pred_boxes and gt_boxes
        x0_inter = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
        y0_inter = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
        x1_inter = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
        y1_inter = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

        inter_area = np.maximum(0, x1_inter - x0_inter) * np.maximum(
            0, y1_inter - y0_inter
        )
        pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        )
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        union_area = pred_areas[:, None] + gt_areas[None, :] - inter_area
        return inter_area / (union_area + 1e-6)

    pred_boxes_by_image = defaultdict(list)
    gt_boxes_by_image = defaultdict(list)

    # Convert predictions to arrays and group by image
    for pred in predictions:
        pred_boxes_by_image[pred["image_id"]].append(
            [pred["x0"], pred["y0"], pred["x1"], pred["y1"], pred["conf"]]
        )

    # Convert ground truths to arrays and group by image
    for gt in ground_truths:
        gt_boxes_by_image[gt["image_id"]].append(
            [gt["x0"], gt["y0"], gt["x1"], gt["y1"]]
        )

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    num_ground_truths = len(ground_truths)

    # Process each image
    for image_id in tqdm(
        set(pred_boxes_by_image.keys()) | set(gt_boxes_by_image.keys())
    ):
        if image_id not in pred_boxes_by_image:
            # No predictions for this image, all ground truths are false negatives
            if image_id in gt_boxes_by_image:
                false_negatives += len(gt_boxes_by_image[image_id])
            continue

        preds = np.array(pred_boxes_by_image[image_id])
        if image_id not in gt_boxes_by_image:
            false_positives += len(preds)
            continue

        # Sort predictions by confidence
        sorted_indices = np.argsort(-preds[:, 4])
        preds = preds[sorted_indices]

        gt_boxes = np.array(gt_boxes_by_image[image_id])
        num_preds, num_gts = len(preds), len(gt_boxes)

        # Calculate IoU matrix for all predictions against all ground truths
        iou_matrix = calculate_iou_matrix(preds[:, :4], gt_boxes)

        # Track matched ground truths
        gt_matched = np.zeros(num_gts, dtype=bool)

        # Process each prediction in order of confidence
        for pred_idx in range(num_preds):
            # Find best unmatched ground truth for this prediction
            unmatched_ious = iou_matrix[pred_idx, ~gt_matched]

            if len(unmatched_ious) == 0:
                false_positives += 1
                continue

            best_iou = np.max(unmatched_ious)
            if best_iou >= iou_threshold:
                best_gt_idx = np.where(~gt_matched)[0][np.argmax(unmatched_ious)]
                gt_matched[best_gt_idx] = True
                true_positives += 1
            else:
                false_positives += 1

        # Count unmatched ground truths as false negatives
        false_negatives += np.sum(~gt_matched).item()

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (num_ground_truths + 1e-6)

    # Calculate F-beta score
    beta_squared = beta**2
    f_score = ((1 + beta_squared) * precision * recall) / (
        beta_squared * precision + recall + 1e-6
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_ground_truths": num_ground_truths,
    }


def validate_epoch(model, data_loader, config, device):
    model.eval()
    predictions = []
    ground_truths = []
    image_id = 0

    with torch.no_grad():
        for images, boxes in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            batch_size = images.shape[0]

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                output = model(images)
                output = model.postprocess(
                    output,
                    conf_thres=config.conf_thres,
                    nms_thres=config.nms_thres,
                    max_detections=config.max_detections,
                )

            batch_image_ids = range(image_id, image_id + batch_size)

            # Process ground truth boxes
            for idx, img_id in enumerate(batch_image_ids):
                mask = boxes[:, 0] == idx
                if mask.any():
                    gt_boxes = boxes[mask, 1:]
                    for box in gt_boxes:
                        ground_truths.append(
                            {
                                "image_id": img_id,
                                "x0": box[0].item() * config.image_width,
                                "y0": box[1].item() * config.image_height,
                                "x1": box[2].item() * config.image_width,
                                "y1": box[3].item() * config.image_height,
                            }
                        )

            # Process predictions
            for idx, pred in enumerate(output):
                if len(pred):
                    for x0, y0, x1, y1, conf in pred.cpu():
                        predictions.append(
                            {
                                "image_id": batch_image_ids[idx],
                                "x0": x0.item(),
                                "y0": y0.item(),
                                "x1": x1.item(),
                                "y1": y1.item(),
                                "conf": conf.item(),
                            }
                        )

            image_id += batch_size

    metrics = compute_metrics(
        predictions,
        ground_truths,
        iou_threshold=config.eval_iou_threshold,
        beta=config.eval_beta,
    )
    return metrics


def train_epoch(model, train_loader, optimizer, scaler, scheduler, loss_func, device):
    model.train()
    box_loss_sum = obj_loss_sum = 0

    for images, boxes in tqdm(train_loader, desc="Training"):
        images, boxes = images.to(device), boxes.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(images)
            box_loss, obj_loss = loss_func(output, boxes)
            loss = box_loss + obj_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        box_loss_sum += box_loss.item()
        obj_loss_sum += obj_loss.item()
        scheduler.step()

    num_batches = len(train_loader)
    return box_loss_sum / num_batches, obj_loss_sum / num_batches


def train_model(train_df, val_df, config, exp_dir):
    try:
        device = setup_device(config.device)
    except RuntimeError as e:
        logger.error(f"Device setup failed: {e}")
        raise
    start_time = time.time()

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)

    model = DetectionModel(
        backbone=config.backbone,
        neck=config.neck,
        act=config.act,
        max_height=config.max_relative_height,
        max_width=config.max_relative_width,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
    ).to(device)

    # Prepare datasets and dataloaders
    train_transform = build_transform_pipeline(config, is_training=True)
    val_transform = build_transform_pipeline(config, is_training=False)

    train_dataset = DetectionDataset(
        train_df,
        transform=train_transform,
        mosaic_prob=config.mosaic_prob,
        load_into_memory=not config.no_memory_load,
    )
    val_dataset = DetectionDataset(
        val_df,
        transform=val_transform,
        mosaic_prob=0.0,
        load_into_memory=not config.no_memory_load,
    )

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Initialize training components
    loss_func = ObjectDetectionLoss(
        num_neighbors=config.num_neighbors, iou_type=config.iou_type
    )
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.wdecay
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        train_epochs=config.epochs,
        train_loader=train_loader,
    )

    # Training loop
    best_f_score = 0
    metrics_log = []

    logger.info(
        f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{config.epochs}")

        # Train and evaluate
        box_loss, obj_loss = train_epoch(
            model, train_loader, optimizer, scaler, scheduler, loss_func, device
        )

        metrics = validate_epoch(model, val_loader, config, device)

        # Save checkpoint if best
        if metrics["f_score"] > best_f_score:
            best_f_score = metrics["f_score"]
            logger.info(f"New best F{config.eval_beta}-score: {best_f_score:.4f}")
            torch.save(model.state_dict(), exp_dir / "model.pt")

        # Log metrics
        epoch_time = time.time() - epoch_start
        metrics_log.append(
            {
                "epoch": epoch,
                "train_box_loss": box_loss,
                "train_obj_loss": obj_loss,
                **metrics,
                "lr": scheduler.get_last_lr()[0],
                "time": epoch_time,
            }
        )

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(exp_dir / "metrics.csv", index=False)

        logger.info(metrics_log[-1])

    total_time = (time.time() - start_time) / 60
    return best_f_score, total_time


def setup_device(device_str=None):
    # If no device specified, use CUDA if available, otherwise CPU
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle different GPU specification formats
    if device_str.startswith("cuda"):
        # Check if CUDA is available at all
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")

        # Handle specific GPU index if provided (e.g., 'cuda:1')
        if ":" in device_str:
            device_idx = int(device_str.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU index {device_idx} requested but only "
                    f"{torch.cuda.device_count()} GPUs available"
                )

    device = torch.device(device_str)

    # Log device information
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB"
        )
    else:
        logger.info("Using CPU for training")

    return device


def parse_args():
    """Parse command line arguments for object detection training.

    Returns:
        argparse.Namespace: Parsed command line arguments with the following groups:
            - Model Architecture: Network configuration and parameters
            - Training Parameters: Basic training loop settings
            - Data Configuration: Input/output paths and data processing settings
            - Evaluation Parameters: Metrics and evaluation thresholds
            - Augmentation Parameters: Image augmentation settings
            - Hardware Configuration: Device and parallelization settings
    """
    parser = argparse.ArgumentParser(description="Object Detection Training")

    # Model Architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--backbone",
        type=str,
        default="vgg11_bn",
        help="Backbone architecture for feature extraction (default: vgg11_bn)",
    )
    model_group.add_argument(
        "--neck",
        type=str,
        default="yolov7",
        help="Neck architecture for feature aggregation (default: yolov7)",
    )
    model_group.add_argument(
        "--act",
        type=str,
        default="ReLU",
        help="Activation function to use throughout the network (default: ReLU)",
    )
    model_group.add_argument(
        "--num_neighbors",
        type=int,
        default=2,
        choices=[0, 2, 4],
        help="Number of neighboring grid cells to consider (0, 2, or 4)",
    )
    model_group.add_argument(
        "--iou_type",
        type=str,
        default="ciou",
        choices=["iou", "giou", "diou", "ciou"],
        help="Type of IOU calculation for loss function (default: ciou)",
    )

    # Model Parameters
    params_group = parser.add_argument_group("Model Parameters")
    params_group.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Flexibility factor for grid boundaries (default: 2.0)",
    )
    params_group.add_argument(
        "--beta",
        type=float,
        default=-0.5,
        help="Shifting factor for coordinate range (default: -0.5)",
    )
    params_group.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Object size bias for width/height predictions (default: 2.0)",
    )
    params_group.add_argument(
        "--max_relative_width",
        type=float,
        default=0.1,
        help="Maximum allowed relative width of detection boxes (default: 0.1)",
    )
    params_group.add_argument(
        "--max_relative_height",
        type=float,
        default=0.1,
        help="Maximum allowed relative height of detection boxes (default: 0.1)",
    )

    # Training Parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)",
    )
    train_group.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for learning rate scheduler (default: 5)",
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)"
    )
    train_group.add_argument(
        "--wdecay",
        type=float,
        default=5e-4,
        help="Weight decay for AdamW optimizer (default: 5e-4)",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size (default: 8)"
    )
    train_group.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")

    # Data Configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--training_file",
        type=str,
        default="folds/train_fold_0.csv",
        help="Path to training annotations CSV file",
    )
    data_group.add_argument(
        "--validation_file",
        type=str,
        default="folds/val_fold_0.csv",
        help="Path to validation annotations CSV file",
    )
    data_group.add_argument(
        "--image_height",
        type=int,
        default=2048,
        help="Input image height (default: 2048)",
    )
    data_group.add_argument(
        "--image_width",
        type=int,
        default=2048,
        help="Input image width (default: 2048)",
    )

    # Evaluation Parameters
    eval_group = parser.add_argument_group("Evaluation Parameters")
    eval_group.add_argument(
        "--eval_beta",
        type=int,
        default=2,
        help="Beta value for F-score calculation (default: 2)",
    )
    eval_group.add_argument(
        "--eval_iou_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for evaluation metrics (default: 0.3)",
    )
    eval_group.add_argument(
        "--conf_thres",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions (default: 0.3)",
    )
    eval_group.add_argument(
        "--nms_thres",
        type=float,
        default=0.3,
        help="Non-maximum suppression IoU threshold (default: 0.3)",
    )
    eval_group.add_argument(
        "--max_detections",
        type=int,
        default=1000,
        help="Maximum number of detections per image (default: 1000)",
    )

    # Augmentation Parameters
    aug_group = parser.add_argument_group("Augmentation Parameters")
    aug_group.add_argument(
        "--brightness",
        type=float,
        default=0.0,
        help="Brightness augmentation factor (default: 0.0)",
    )
    aug_group.add_argument(
        "--contrast",
        type=float,
        default=0.0,
        help="Contrast augmentation factor (default: 0.0)",
    )
    aug_group.add_argument(
        "--saturation",
        type=float,
        default=0.0,
        help="Saturation augmentation factor (default: 0.0)",
    )
    aug_group.add_argument(
        "--hue", type=float, default=0.0, help="Hue augmentation factor (default: 0.0)"
    )
    aug_group.add_argument(
        "--degrees",
        type=int,
        default=0,
        help="Maximum rotation degrees for augmentation (default: 0)",
    )
    aug_group.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Maximum translation factor for augmentation (default: 0.1)",
    )
    aug_group.add_argument(
        "--mosaic_prob",
        type=float,
        default=0.0,
        help="Probability of applying mosaic augmentation (default: 0.0)",
    )

    # Hardware Configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device to use for training (e.g., 'cuda:0', 'cpu'). "
        "If not specified, will use CUDA if available, otherwise CPU.",
    )
    hw_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    hw_group.add_argument(
        "--no_memory_load",
        action="store_true",
        help="Disable loading images into memory. If set, images will be loaded on-demand "
        "during training (reduces memory usage but may be slower)",
    )
    hw_group.add_argument(
        "--no_deterministic",
        action="store_false",
        dest="deterministic",
        help="Disable deterministic algorithms. By default, torch.use_deterministic_algorithms(True) "
        "is enabled for reproducibility. Use this flag to disable it for better performance.",
    )

    args = parser.parse_args()
    return args


def main():
    config = parse_args()
    exp_dir = create_experiment_directory()

    # Configure logging to file
    fh = logging.FileHandler(exp_dir / "training.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Starting experiment in {exp_dir}")
    logger.info(f"Configuration: {vars(config)}")

    # Load data
    train_df = pd.read_csv(config.training_file)
    val_df = pd.read_csv(config.validation_file)

    # Train model
    f_score, runtime = train_model(train_df, val_df, config, exp_dir)

    logger.info(f"Training completed in {runtime:.2f} minutes")
    logger.info(f"Best F{config.eval_beta} score: {f_score:.4f}")


if __name__ == "__main__":
    main()
