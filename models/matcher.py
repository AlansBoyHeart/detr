# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):    # 1   5   2
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class   #1
        self.cost_bbox = cost_bbox     #5
        self.cost_giou = cost_giou     #2
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]   #[2,100,92]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]  [200,92]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]   [200,4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])   # [18]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])   # [18,4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]   #[200,18]
        # print(1,cost_class)   #[[-0.0025, -0.0017, -0.0017, -0.0017]]  如果tgt中同一个类别出现多个，比如多个人，对应的概率也就会被多次选中。
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)   #求两个向量的l1距离， [2,3,4]*[2,5,4]=[2,3,5]   [200,4]*[18,4]=[200,18]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))   #[200,4]  [18,4]
        #Lgiou = 1-GIOU = 1- ( iou - (area-union)/area)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou   #[200,18]
        C = C.view(bs, num_queries, -1).cpu()   #[2,100,18]

        sizes = [len(v["boxes"]) for v in targets]   #[14  4]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]   #切出来两个矩阵[2,100,14] [2,100,4]
        #linear_sum_assignment每一行取一个值，相加的和最小，列也不能重复，[100,14]只能取14个值,返回值是行和列的索引
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        #返回值是匹配的索引[100,14]，[100,4] 返回[([14],[14]), ([4],[4])]分别是行和列的索引


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)   # 1   5   2


if __name__ == '__main__':
    # a = torch.ones((1,2,5)) *2
    # b = torch.zeros((1,3,5))
    # c = torch.cdist(a,b,p=2)
    # print(c)

    import numpy as np
    cost = np.array([[4, 1, 3], [2, 0, 5], [6,7,8],[1,2,3]])
    row_ind, col_ind = linear_sum_assignment(cost)
    print(row_ind)  # 开销矩阵对应的行索引
    print(col_ind)  # 对应行索引的最优指派的列索引
    print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
    print(cost[row_ind, col_ind].sum())  # 数组求和

