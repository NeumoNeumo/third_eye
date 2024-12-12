import torch.nn as nn
import torch
import torchvision

class DetectionLoss(nn.Module):
    def __init__(self, alpha1=1.0, alpha2=5.0, iou_thresh=0.5):
        super(DetectionLoss, self).__init__()
        self.alpha1 = alpha1  # Weight for objectness loss
        self.alpha2 = alpha2  # Weight for bbox regression loss
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # Classification loss
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # Objectness loss
        self.iou_thresh = iou_thresh

    def eiou_loss(self, pred_boxes, target_boxes):
        pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x_min, target_y_min, target_x_max, target_y_max = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # Compute IoU
        inter_x_min = torch.max(pred_x_min, target_x_min)
        inter_y_min = torch.max(pred_y_min, target_y_min)
        inter_x_max = torch.min(pred_x_max, target_x_max)
        inter_y_max = torch.min(pred_y_max, target_y_max)
        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)
        pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
        target_area = (target_x_max - target_x_min) * (target_y_max - target_y_min)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / union_area

        # Compute center and aspect differences for EIoU
        pred_center_x = (pred_x_min + pred_x_max) / 2
        pred_center_y = (pred_y_min + pred_y_max) / 2
        target_center_x = (target_x_min + target_x_max) / 2
        target_center_y = (target_y_min + target_y_max) / 2
        center_dist = (pred_center_x - target_center_x).pow(2) + (pred_center_y - target_center_y).pow(2)

        pred_diag = (pred_x_max - pred_x_min).pow(2) + (pred_y_max - pred_y_min).pow(2)
        target_diag = (target_x_max - target_x_min).pow(2) + (target_y_max - target_y_min).pow(2)

        eiou = 1 - iou + center_dist / (pred_diag + 1e-7) + (pred_diag - target_diag).pow(2) / (pred_diag + target_diag + 1e-7)
        return eiou.mean()

    def forward(self, outputs, targets):
        cls_preds, obj_preds, bbox_preds = outputs
        total_loss = 0

        batch_size = cls_preds.size(0)
        for i in range(batch_size):
            cls_pred = cls_preds[i]  # Shape: [num_preds, num_classes]
            obj_pred = obj_preds[i]  # Shape: [num_preds, 1]
            bbox_pred = bbox_preds[i]  # Shape: [num_preds, 4]
            # x_min, y_min, h, w -> x_min, y_min, x_max, y_max
            bbox_pred[:, 2:] += bbox_pred[:, :2]
            

            if targets[i].size(0) == 0:
                obj_target = torch.zeros_like(obj_pred).to(obj_pred.device)
                cls_target = torch.zeros_like(cls_pred).to(cls_pred.device)
                bbox_loss = torch.tensor(0.0, device=obj_pred.device)
            else:
                gt_boxes = targets[i][:, :4]

                iou_matrix = torchvision.ops.box_iou(gt_boxes, bbox_pred)
                max_ious, max_indices = iou_matrix.max(dim=0)

                obj_target = (max_ious > self.iou_thresh).float().unsqueeze(-1)
                cls_target = torch.zeros_like(cls_pred)
                cls_target[max_indices[max_ious > self.iou_thresh], 0] = 1  # Assume only one class

                pos_mask = max_ious > self.iou_thresh
                if pos_mask.sum() > 0:
                    bbox_loss = self.eiou_loss(bbox_pred[pos_mask], gt_boxes[max_indices[pos_mask]])
                else:
                    bbox_loss = torch.tensor(0.0, device=obj_pred.device)

            cls_loss = self.cls_loss_fn(cls_pred, cls_target)
            obj_loss = self.obj_loss_fn(obj_pred, obj_target)

            total_loss += cls_loss + self.alpha1 * obj_loss + self.alpha2 * bbox_loss

        return total_loss / batch_size