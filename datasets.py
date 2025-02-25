from torch.utils.data import Dataset
import numpy as np
from torchvision import tv_tensors
import random
import torch
from PIL import Image
from tqdm import tqdm


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)

    image_ids = torch.cat(
        [torch.full((len(b["boxes"]), 1), i) for i, b in enumerate(batch)], dim=0
    )
    boxes = torch.cat([b["boxes"] for b in batch], dim=0)

    out = torch.cat((image_ids, boxes), dim=1)

    return images, out


class DetectionDataset(Dataset):
    def __init__(self, df, transform, mosaic_prob=1.0, load_into_memory=False):
        """Initialize the detection dataset.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and annotations
            transform: Transforms to apply to images and annotations
            mosaic_prob (float): Probability of applying mosaic augmentation (0.0-1.0)
            load_into_memory (bool): If True, loads all images into memory during initialization
        """
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.pad_colour = (144, 144, 144)
        self.df = []
        self.load_into_memory = load_into_memory
        self.images = []

        desc = (
            "Loading images into memory"
            if self.load_into_memory
            else "Processing annotations"
        )

        # Group annotations by image path
        image_paths = sorted(df["image_path"].unique())
        for image_path in tqdm(image_paths, desc=desc):
            rows = df[df["image_path"] == image_path]
            self.df.append(
                {
                    "image_path": image_path,
                    "bounding_boxes": rows[["x0", "y0", "x1", "y1"]].values.astype(
                        np.float32
                    ),
                    "class_ids": rows["class_id"].values.astype(np.float32),
                }
            )

            # Load images into memory if requested
            if self.load_into_memory:
                with Image.open(image_path) as img:
                    # Convert to numpy array to ensure image file is closed
                    self.images.append(np.array(img))

    def __len__(self):
        return len(self.df)

    def create_mosaic(self, images, boxes, classes):
        mosaic_width = max(
            images[0].shape[1] + images[1].shape[1],
            images[2].shape[1] + images[3].shape[1],
        )
        mosaic_height = max(
            images[0].shape[0] + images[2].shape[0],
            images[1].shape[0] + images[3].shape[0],
        )

        mosaic_image = np.full(
            (mosaic_height, mosaic_width, 3),
            self.pad_colour,
            dtype=np.uint8,
        )

        centre_x, centre_y = self._get_mosaic_centre(mosaic_height, mosaic_width)

        mosaic_labels = []

        for mosaic_position, (image, image_boxes, image_classes) in enumerate(
            zip(images, boxes, classes)
        ):
            # concat boxes and classes for easier processing
            if len(image_boxes) > 0:
                _labels = np.concatenate((image_boxes, image_classes[None].T), axis=1)
            else:
                _labels = np.array([])

            # generate output mosaic image
            (image_height, image_width, c) = image.shape[:3]

            (
                (mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2),
                (
                    image_x1,
                    image_y1,
                    image_x2,
                    image_y2,
                ),
            ) = self._get_mosaic_coordinates(
                mosaic_position,
                centre_x,
                centre_y,
                image_height,
                image_width,
                mosaic_height,
                mosaic_width,
            )

            mosaic_image[mosaic_y1:mosaic_y2, mosaic_x1:mosaic_x2] = image[
                image_y1:image_y2, image_x1:image_x2
            ]

            shift_x = mosaic_x1 - image_x1
            shift_y = mosaic_y1 - image_y1

            labels = _labels.copy()
            if labels.size > 0:
                labels = self.apply_shift_to_labels(labels, shift_x, shift_y)
            mosaic_labels.append(labels)

        # filter empty labels
        mosaic_labels = [labels for labels in mosaic_labels if len(labels) > 0]
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            self.clip_labels_inplace(
                mosaic_labels,
                output_height=mosaic_height,
                output_width=mosaic_width,
            )

            valid_boxes = (mosaic_labels[:, 2] > mosaic_labels[:, 0]) & (
                mosaic_labels[:, 3] > mosaic_labels[:, 1]
            )
            mosaic_labels = mosaic_labels[valid_boxes]
            mosaic_boxes = mosaic_labels[:, :4]
            mosaic_classes = mosaic_labels[:, 4]

        if len(mosaic_labels) == 0:
            mosaic_boxes = np.array([])
            mosaic_classes = np.array([])

        return (
            mosaic_image,
            mosaic_boxes,
            mosaic_classes,
        )  # , (mosaic_height, mosaic_width)

    def clip_labels_inplace(self, labels, output_height, output_width):
        np.clip(labels[:, 0], 0, output_width, out=labels[:, 0])
        np.clip(labels[:, 1], 0, output_height, out=labels[:, 1])
        np.clip(labels[:, 2], 0, output_width, out=labels[:, 2])
        np.clip(labels[:, 3], 0, output_height, out=labels[:, 3])

    def apply_shift_to_labels(self, labels, shift_x, shift_y):
        labels_out = labels.copy()
        labels_out[:, 0] = labels[:, 0] + shift_x
        labels_out[:, 1] = labels[:, 1] + shift_y
        labels_out[:, 2] = labels[:, 2] + shift_x
        labels_out[:, 3] = labels[:, 3] + shift_y
        return labels_out

    def _get_mosaic_centre(self, mosaic_height, mosaic_width):
        centre_x = mosaic_width // 2
        centre_y = mosaic_height // 2
        return centre_x, centre_y

    def _get_mosaic_coordinates(
        self,
        position_idx,
        centre_x,
        centre_y,
        image_height,
        image_width,
        mosaic_height,
        mosaic_width,
    ):
        if position_idx == 0:  # top left
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                max(centre_x - image_width, 0),
                max(centre_y - image_height, 0),
                centre_x,
                centre_y,
            )
            image_x1, image_y1, image_x2, image_y2 = (
                image_width - (mosaic_x2 - mosaic_x1),
                image_height - (mosaic_y2 - mosaic_y1),
                image_width,
                image_height,
            )

        elif position_idx == 1:  # top right
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                centre_x,
                max(centre_y - image_height, 0),
                min(centre_x + image_width, mosaic_width),
                centre_y,
            )
            image_x1, image_y1, image_x2, image_y2 = (
                0,
                image_height - (mosaic_y2 - mosaic_y1),
                min(image_width, mosaic_x2 - mosaic_x1),
                image_height,
            )

        elif position_idx == 2:  # bottom left
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                max(centre_x - image_width, 0),
                centre_y,
                centre_x,
                min(mosaic_height, centre_y + image_height),
            )
            image_x1, image_y1, image_x2, image_y2 = (
                image_width - (mosaic_x2 - mosaic_x1),
                0,
                image_width,
                min(mosaic_y2 - mosaic_y1, image_height),
            )

        elif position_idx == 3:  # bottom right
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                centre_x,
                centre_y,
                min(centre_x + image_width, mosaic_width),
                min(mosaic_height, centre_y + image_height),
            )
            image_x1, image_y1, image_x2, image_y2 = (
                0,
                0,
                min(image_width, mosaic_x2 - mosaic_x1),
                min(mosaic_y2 - mosaic_y1, image_height),
            )

        else:
            raise ValueError("Incorrect index given, the accepted range is [0, 3]")

        return (mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2), (
            image_x1,
            image_y1,
            image_x2,
            image_y2,
        )

    def load_from_dataset(self, idx):
        """Load an image and its annotations from the dataset.

        Args:
            idx (int): Index of the image to load

        Returns:
            tuple: (image array, bounding boxes, class IDs)
        """
        rows = self.df[idx]

        if self.load_into_memory:
            img = self.images[idx]
        else:
            with Image.open(rows["image_path"]) as img:
                img = np.array(img)

        return img, rows["bounding_boxes"], rows["class_ids"]

    def __getitem__(self, idx):
        indices = [idx]
        if random.random() <= self.mosaic_prob:
            indices += torch.randint(low=0, high=len(self.df), size=(3,)).tolist()
            random.shuffle(indices)

            mosaic_img, mosaic_boxes, mosaic_classes = zip(
                *[self.load_from_dataset(ds_index) for ds_index in indices]
            )
            img, boxes, classes = self.create_mosaic(
                mosaic_img, mosaic_boxes, mosaic_classes
            )
        else:
            img, boxes, classes = self.load_from_dataset(indices[0])

        img = Image.fromarray(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.float32)

        h, w = img.height, img.width
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))

        transformed = self.transform({"image": img, "boxes": boxes, "labels": classes})
        transformed["boxes"][:, [1, 3]] /= transformed["image"].shape[1]
        transformed["boxes"][:, [0, 2]] /= transformed["image"].shape[2]

        return transformed
