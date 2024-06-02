import torch.nn as nn
import torch_scatter

from scorer.ptv3 import model

class Classifier(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.backbone = model.PointTransformerV3()
        self.num_classes = cfg.num_classes
        self.backbone_embed_dim = cfg.backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(self.backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, input_dict):
        point = model.Point(input_dict)
        point = self.backbone(point) 
        if isinstance(point, model.Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="max",
            )
            feat = point.feat #(b,512)
        else:
            feat = point
        cls_logits = self.cls_head(feat) #(b,1)
        return (cls_logits)
