from torch import nn
import timm
import torch
import torchvision
from model.necks.yolov7 import Yolov7Neck
from model.necks.yolox import YoloXNeck
from model.heads.yolov7 import Yolov7Head
from model.backbones.yolov7 import yolov7_tiny


class DetectionModel(nn.Module):
    def __init__(
        self,
        backbone="yolov7_tiny",
        neck="yolov7",
        act="ReLU",
        max_height=0.1,
        max_width=0.1,
        alpha=2.0,
        beta=-0.5,
        gamma=2,
        num_layers=3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.max_height = max_height
        self.max_width = max_width
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if backbone == "yolov7_tiny":
            feature_channels = [128, 256, 512]

            self.backbone = yolov7_tiny(act=act)
        else:
            self.backbone = timm.create_model(
                backbone,
                features_only=True,
                pretrained=True,
            )
            feature_channels = self.backbone.feature_info.channels()[-self.num_layers :]

        if neck == "yolov7":
            self.neck = Yolov7Neck(
                num_channels_in=feature_channels,
                num_channels_out=[128, 256, 512],
                act=act,
            )
        elif neck == "yolox":
            self.neck = YoloXNeck(
                num_channels_in=feature_channels,
                num_channels_out=[128, 256, 512],
                act=act,
            )

        self.head = Yolov7Head(
            in_channels=[128, 256, 512],
            act=act,
        )

    def forward(self, x):
        image_height, image_width = x.shape[2:]
        feats = self.backbone(x)[-self.num_layers :]
        feats = self.neck(feats)
        feats = self.head(feats)

        outs = []
        for layer_idx, out in enumerate(feats):
            out = out.permute(0, 2, 3, 1)
            batch_size, grid_height, grid_width, _ = out.shape

            # inplace operations are not possible here!
            preds_xy = out[..., 0:2].sigmoid() * self.alpha + self.beta
            preds_w = (
                out[..., 2:3].sigmoid() ** self.gamma * grid_width * self.max_width
            )
            preds_h = (
                out[..., 3:4].sigmoid() ** self.gamma * grid_height * self.max_height
            )
            preds_obj = out[..., 4:5]

            meshgrid = torch.meshgrid(
                [
                    torch.arange(grid_height, device=out.device),
                    torch.arange(grid_width, device=out.device),
                ],
                indexing="ij",
            )
            grid = torch.stack((meshgrid[1], meshgrid[0]), dim=2)[None].float()
            preds_xy += grid

            preds_xy[..., 0] *= image_width / grid_width
            preds_xy[..., 1] *= image_height / grid_height
            preds_w *= image_width / grid_width
            preds_h *= image_height / grid_height

            if not self.training:
                preds_obj = torch.sigmoid(preds_obj)

                out = torch.cat([preds_xy, preds_w, preds_h, preds_obj], dim=-1)
                outs.append(out.reshape(batch_size, -1, out.shape[-1]))
            else:
                out = torch.cat([preds_xy, preds_w, preds_h, preds_obj], dim=-1)
                out[..., [0, 2]] /= image_width
                out[..., [1, 3]] /= image_height
                outs.append(out)

        if not self.training:
            return torch.cat(outs, dim=1)

        return outs

    def postprocess(
        self,
        preds: torch.Tensor,
        conf_thres: float = 0.3,
        nms_thres: float = 0.3,
        max_detections: int = 1000,
    ) -> list:
        device = preds.device
        batch_size = preds.shape[0]

        conf_mask = preds[..., 4] >= conf_thres

        formatted_preds = []
        for image_idx in range(batch_size):
            image_mask = conf_mask[image_idx]
            if not image_mask.any():
                formatted_preds.append(torch.zeros((0, 5), device=device))
                continue

            image_dets = preds[image_idx][image_mask]

            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
            boxes = torchvision.ops.box_convert(image_dets[:, :4], "cxcywh", "xyxy")

            # Apply NMS
            nms_indices = torchvision.ops.nms(
                boxes=boxes, scores=image_dets[:, 4], iou_threshold=nms_thres
            )

            # Take top-k detections by confidence after NMS
            if len(nms_indices) > max_detections:
                scores = image_dets[nms_indices, 4]
                _, top_k_indices = torch.topk(scores, max_detections)
                nms_indices = nms_indices[top_k_indices]

            # Store results
            formatted_preds.append(
                torch.cat([boxes[nms_indices], image_dets[nms_indices, 4:5]], dim=1)
            )

        return formatted_preds
