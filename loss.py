import math

import torch
import torchvision
import torch.nn as nn
from typing import List, Tuple


class ObjectDetectionLoss:
    """
    Object Detection Loss function that combines bounding box regression and objectness prediction.

    This loss function is designed for object detection networks that predict bounding boxes
    and objectness scores across multiple grid scales. It supports various IOU calculations
    and neighboring cell assignments.

    Args:
        num_neighbors (int): Number of neighboring grid cells to consider. Valid options:
            - 0: Only center cell
            - 2: Center cell plus closest horizontal and vertical neighbors
            - 4: Center cell plus all adjacent cells
        iou_type (str): Type of IOU calculation to use. Options:
            - "iou": Standard Intersection over Union
            - "giou": Generalized IOU (adds penalty for empty area)
            - "diou": Distance IOU (adds center point distance term)
            - "ciou": Complete IOU (adds aspect ratio consistency)

    Example:
        >>> # Initialize loss function with 2 neighbors and GIOU
        >>> loss_fn = ObjectDetectionLoss(num_neighbors=2, iou_type="giou")
        >>>
        >>> # Example predictions (batch_size=1, grid=13x13, 5 values per cell)
        >>> pred = torch.rand(1, 13, 13, 5)  # [cx, cy, w, h, obj]
        >>>
        >>> # Example targets (2 ground truth boxes)
        >>> target = torch.tensor([[0, 1, 0.2, 0.3, 0.4, 0.5],
        ...                       [0, 2, 0.6, 0.7, 0.8, 0.9]])
        >>>
        >>> # Compute loss
        >>> box_loss, obj_loss = loss_fn([pred], target)

    Raises:
        ValueError: If num_neighbors is not 0, 2, or 4
        ValueError: If iou_type is not one of the supported types
        TypeError: If input parameters are of incorrect type
    """

    VALID_NEIGHBOR_COUNTS = {0, 2, 4}
    VALID_IOU_TYPES = {"iou", "giou", "diou", "ciou"}

    def __init__(self, num_neighbors: int = 2, iou_type: str = "ciou") -> None:
        # Type checking
        if not isinstance(num_neighbors, int):
            raise TypeError(
                f"num_neighbors must be an integer, got {type(num_neighbors)}"
            )
        if not isinstance(iou_type, str):
            raise TypeError(f"iou_type must be a string, got {type(iou_type)}")

        # Value validation
        if num_neighbors not in self.VALID_NEIGHBOR_COUNTS:
            raise ValueError(
                f"num_neighbors must be one of {self.VALID_NEIGHBOR_COUNTS}, "
                f"got {num_neighbors}"
            )

        iou_type = iou_type.lower()
        if iou_type not in self.VALID_IOU_TYPES:
            raise ValueError(
                f"iou_type must be one of {self.VALID_IOU_TYPES}, got '{iou_type}'"
            )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.num_neighbors = num_neighbors
        self.iou_type = iou_type

    def __call__(
        self,
        all_preds: List[torch.Tensor],
        all_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute object detection loss combining bounding box regression and objectness prediction.

        Args:
            all_preds (List[torch.Tensor]): List of prediction tensors at different scales.
                Each tensor has shape: [batch_size, grid_height, grid_width, 5] where:
                - batch_size: Number of images in batch
                - grid_height: Height of prediction grid at this scale
                - grid_width: Width of prediction grid at this scale
                - 5: Number of values predicted per cell [center_x, center_y, width, height, objectness_logit]
                All spatial values should be normalized to [0, 1] relative to grid size.

            all_targets (torch.Tensor): Ground truth boxes tensor with shape [num_boxes, 5] where:
                - num_boxes: Total number of ground truth boxes
                - 5: Values per box [image_index, x1, y1, x2, y2]
                Box coordinates should be in XYXY format normalized to [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - box_loss (torch.Tensor): Scalar tensor with bounding box regression loss
                - obj_loss (torch.Tensor): Scalar tensor with objectness prediction loss
        """
        # Validate inputs and initialize device-aware tensors
        self._validate_inputs(all_preds, all_targets)
        device = all_preds[0].device
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)

        # Special handling for empty targets case
        if all_targets.shape[0] == 0:
            for pred_boxes in all_preds:
                # When no targets exist, all locations should predict no object
                pred_objectness = pred_boxes[..., 4]
                target_objectness = torch.zeros_like(pred_objectness)
                obj_loss += self.bce_loss(pred_objectness, target_objectness)
            return box_loss, obj_loss

        # Convert target boxes from XYXY to CXCYWH format for grid assignment
        cxcywh_boxes = torchvision.ops.box_convert(
            all_targets[:, 1:5], "xyxy", "cxcywh"
        )
        all_targets = torch.cat((all_targets[:, :1], cxcywh_boxes), dim=-1)

        # Process each prediction scale
        for pred_boxes in all_preds:
            # Get grid dimensions and find responsible cells
            grid_height, grid_width = pred_boxes.shape[1:3]
            gt_idx_neighbors, gt_val_neighbors = self._get_neighbors(
                all_targets.clone(), grid_height, grid_width
            )

            # Extract predictions for responsible cells
            pred_boxes_neighbors = pred_boxes[*gt_idx_neighbors]

            # Convert boxes to XYXY format for IOU computation
            pred_boxes_xyxy = torchvision.ops.box_convert(
                pred_boxes_neighbors[..., :4], "cxcywh", "xyxy"
            )
            gt_boxes_xyxy = torchvision.ops.box_convert(
                gt_val_neighbors[..., :4], "cxcywh", "xyxy"
            )

            # Compute box loss using IOU
            iou = self.bbox_iou(pred_boxes_xyxy, gt_boxes_xyxy)
            box_loss += (1.0 - iou).mean()

            # Compute objectness loss
            pred_objectness = pred_boxes[..., 4]
            target_objectness = torch.zeros_like(pred_objectness, device=device)
            target_objectness[*gt_idx_neighbors] = iou.detach().clamp(0, 1)
            obj_loss += self.bce_loss(pred_objectness, target_objectness)

        return box_loss, obj_loss

    def _validate_inputs(
        self,
        all_preds: List[torch.Tensor],
        all_targets: torch.Tensor,
    ) -> None:
        """
        Validates the shape, format, and content of prediction and target tensors.

        Args:
            all_preds (List[torch.Tensor]): List of prediction tensors to validate.
                Each tensor should have shape [batch_size, height, width, 5].
            all_targets (torch.Tensor): Target tensor to validate with shape [num_boxes, 5].

        Raises:
            ValueError: If predictions list is empty
            ValueError: If any prediction tensor has incorrect dimensions
            ValueError: If prediction channels != 5
            ValueError: If targets tensor has incorrect dimensions
            ValueError: If target columns != 5
            TypeError: If inputs are of incorrect type
        """
        if not isinstance(all_preds, list):
            raise ValueError("Predictions must be provided as a list of tensors")

        for i, pred in enumerate(all_preds):
            if not isinstance(pred, torch.Tensor):
                raise ValueError(f"Prediction at index {i} must be a tensor")

            if len(pred.shape) != 4:
                raise ValueError(
                    f"Prediction tensor at index {i} must have 4 dimensions "
                    f"(batch_size, height, width, channels), got shape {pred.shape}"
                )

            if pred.shape[-1] != 5:  # x, y, w, h, objectness
                raise ValueError(
                    f"Prediction tensor at index {i} must have 5 channels, got {pred.shape[-1]}"
                )

        if not isinstance(all_targets, torch.Tensor):
            raise ValueError("Targets must be provided as a tensor")

        if len(all_targets.shape) != 2:
            raise ValueError(
                f"Target tensor must have 2 dimensions (num_boxes, 5), got shape {all_targets.shape}"
            )

        if all_targets.shape[1] != 5:  # image_idx, x1, y1, x2, y2
            raise ValueError(
                f"Target tensor must have 5 columns, got {all_targets.shape[1]}"
            )

    def _get_neighbors(
        self, gt_boxes: torch.Tensor, grid_height: int, grid_width: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Find neighboring grid cells for ground truth boxes based on their positions.

        Args:
            gt_boxes (torch.Tensor): Ground truth boxes tensor with shape [N, 5] containing:
                [image_idx, center_x, center_y, width, height]
                All spatial values should be normalized to [0, 1].
            grid_height (int): Height of the prediction grid
            grid_width (int): Width of the prediction grid

        Returns:
            Tuple containing:
                - grid_indices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                    Indices of selected grid cells as (image_idx, y_idx, x_idx)
                - box_values (torch.Tensor): Box coordinates normalized to grid size
                    with shape [num_selected_cells, 4] containing [cx, cy, w, h]
        """
        IMAGE_IDX = 0
        X_COORD = 1
        Y_COORD = 2
        WIDTH = 3
        HEIGHT = 4

        # gt_boxes (num_boxes, 5)
        # scale to size of grid
        gt_boxes[:, [X_COORD, WIDTH]] *= grid_width
        gt_boxes[:, [Y_COORD, HEIGHT]] *= grid_height

        box_cx = gt_boxes[:, X_COORD]
        box_cy = gt_boxes[:, Y_COORD]
        box_cx_inv = grid_width - gt_boxes[:, X_COORD]
        box_cy_inv = grid_height - gt_boxes[:, Y_COORD]

        get_center = torch.ones(
            gt_boxes.shape[0], device=gt_boxes.device, dtype=torch.bool
        )

        # Get all surrounding grid cells (right, left, top, down)
        max_offset = 1
        if self.num_neighbors == 2 or self.num_neighbors == 4:
            # at the edges there are no neighbors
            get_left = box_cx > 1.0
            get_up = box_cy > 1.0
            get_right = box_cx_inv > 1.0
            get_down = box_cy_inv > 1.0

        # Get closest grid cells (right or left, and top or down)
        if self.num_neighbors == 2:
            max_offset = 0.5

            # returns the fractional part
            box_cx_offset = box_cx % 1.0
            box_cy_offset = box_cy % 1.0
            box_cx_offset_inv = box_cx_inv % 1.0
            box_cy_offset_inv = box_cy_inv % 1.0

            get_left &= box_cx_offset < 0.5
            get_up &= box_cy_offset < 0.5
            get_right &= box_cx_offset_inv < 0.5
            get_down &= box_cy_offset_inv < 0.5

        directions = [get_center]
        directions_offsets_x = [0]
        directions_offsets_y = [0]
        if self.num_neighbors > 0:
            directions += [get_left, get_up, get_right, get_down]
            directions_offsets_x += [1, 0, -1, 0]
            directions_offsets_y += [0, 1, 0, -1]

        # boolean tensor shape: (num_directions, num_boxes) where num_directions \in {1,5}
        directions_mask = torch.stack(directions, dim=0)

        # tensor shape: (num_boxes, 5) -> (num_directions, num_boxes, 5)
        box_neighbors = gt_boxes.repeat((directions_mask.shape[0], 1, 1))

        # apply mask (num_directions, num_boxes) to (num_directions, num_boxes, 5)
        # result: (num_boxes * num_directions_true, 5)
        box_neighbors = box_neighbors[directions_mask]

        # tensor shape: (num_directions, num_boxes) -> (num_boxes * num_directions_true,)
        directions_offsets_x = max_offset * torch.tensor(
            directions_offsets_x, device=directions_mask.device
        ).view(-1, 1)
        directions_offsets_x = directions_offsets_x.repeat((1, gt_boxes.shape[0]))
        directions_offsets_x = directions_offsets_x[directions_mask]

        directions_offsets_y = max_offset * torch.tensor(
            directions_offsets_y, device=directions_mask.device
        ).view(-1, 1)
        directions_offsets_y = directions_offsets_y.repeat((1, gt_boxes.shape[0]))
        directions_offsets_y = directions_offsets_y[directions_mask]

        box_cx_neighbors = box_neighbors[..., X_COORD] - directions_offsets_x
        box_cx_neighbors.clamp_(0, grid_width - 1)
        box_cy_neighbors = box_neighbors[..., Y_COORD] - directions_offsets_y
        box_cy_neighbors.clamp_(0, grid_height - 1)

        box_cx_int_neighbors = box_cx_neighbors.long()
        box_cy_int_neighbors = box_cy_neighbors.long()

        box_image_id_neighbors = box_neighbors[..., IMAGE_IDX].long()
        box_width_neighbors = box_neighbors[..., WIDTH]
        box_height_neighbors = box_neighbors[..., HEIGHT]

        gt_box_idx_neighbors = (
            box_image_id_neighbors,
            box_cy_int_neighbors,
            box_cx_int_neighbors,
        )
        gt_box_val_neighbors = torch.stack(
            (
                box_neighbors[..., X_COORD],
                box_neighbors[..., Y_COORD],
                box_width_neighbors,
                box_height_neighbors,
            ),
            dim=-1,
        )
        gt_box_val_neighbors[..., [0, 2]] /= grid_width
        gt_box_val_neighbors[..., [1, 3]] /= grid_height

        return gt_box_idx_neighbors, gt_box_val_neighbors

    def bbox_iou(
        self, box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Calculate IoU (Intersection over Union) between two sets of bounding boxes.

        Supports multiple IoU variants:
        - Standard IoU: Intersection area / Union area
        - GIoU: Adds penalty based on smallest enclosing box
        - DIoU: Adds penalty based on center point distance
        - CIoU: Adds penalties for both center distance and aspect ratio

        Args:
            box1 (torch.Tensor): First set of boxes in XYXY format with shape [..., 4]
            box2 (torch.Tensor): Second set of boxes in XYXY format with shape [..., 4]
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-7.

        Returns:
            torch.Tensor: IoU scores with same shape as input except last dimension.
                Values are in range [0, 1] for standard IoU, or [-1, 1] for other variants.
        """
        # Extract coordinates for both boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[..., 0],
            box1[..., 1],
            box1[..., 2],
            box1[..., 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[..., 0],
            box2[..., 1],
            box2[..., 2],
            box2[..., 3],
        )

        # Calculate intersection area
        # max of xs/ys for top-left corner, min of xs/ys for bottom-right corner
        intersection_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        intersection_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        intersection = intersection_w * intersection_h

        # Calculate areas of both boxes
        # Add eps to prevent division by zero later
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calculate union area and IoU
        union = box1_area + box2_area - intersection + eps
        iou = intersection / union

        # Return standard IoU if no other metric specified
        if self.iou_type == "iou":
            return iou

        # Calculate the coordinates of the smallest enclosing box (convex hull)
        convex_width = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        convex_height = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if self.iou_type in ["diou", "ciou"]:
            # Calculate squared diagonal of smallest enclosing box
            convex_diagonal_squared = convex_width**2 + convex_height**2 + eps

            # Calculate squared euclidean distance between centers
            center_distance_squared = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4

            if self.iou_type == "diou":
                # DIoU adds a normalized distance term to standard IoU
                return iou - center_distance_squared / convex_diagonal_squared

            else:  # CIoU
                # Calculate aspect ratio consistency term
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                # Calculate alpha (trade-off parameter) without gradient calculation
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # CIoU adds both distance and aspect ratio terms to standard IoU
                return iou - (
                    center_distance_squared / convex_diagonal_squared + v * alpha
                )

        else:  # GIoU
            # Calculate area of smallest enclosing box
            convex_area = convex_width * convex_height + eps
            # GIoU adds a penalty based on the ratio of empty area in the convex hull
            return iou - (convex_area - union) / convex_area
