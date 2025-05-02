import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, dist2bbox

class DFLoss(nn.Module):
    def _init_(self, reg_max=16):
        super()._init_()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
            F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr
        )
        return loss.mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    def _init_(self, reg_max=16, iou_type='ciou'):
        super()._init_()
        self.dfl_loss = DFLoss(reg_max)
        self.iou_type = iou_type

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou_args = {'xywh': False, self.iou_type.upper(): True}
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], iou_args)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

class CustomYOLOv8Loss:
    def _init_(self, model, tal_topk=10):
        device = next(model.parameters()).device
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = model.args
        self.stride = m.stride
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.no = self.nc + self.reg_max * 4
        self.device = device
        self.use_dfl = self.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max, iou_type=self.hyp.get('iou_type', 'ciou')).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def _call_(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([
            x.view(feats[0].shape[0], self.no, -1) for x in feats
        ], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        gt_labels, gt_bboxes = targets[:, 1].long(), targets[:, 2:]
        mask_gt = gt_bboxes.sum(1, keepdim=True).gt(0.0)

        pred_bboxes = dist2bbox(
            pred_distri.view(batch_size, -1, 4, self.reg_max).softmax(-1).matmul(self.proj),
            anchor_points, xywh=False
        )

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels.unsqueeze(-1),
            gt_bboxes.unsqueeze(0),
            mask_gt.unsqueeze(0)
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()
