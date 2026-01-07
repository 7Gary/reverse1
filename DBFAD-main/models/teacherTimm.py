import timm
import torch
import torch.nn as nn
import os

class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet34",
        out_indices=[0, 1, 2, 3],
        weight_path="/data/c425/cx/DBFAD/resnet34-b627a593.pth"
    ):
        super(teacherTimm, self).__init__()

        # 1. 创建模型（禁止任何在线下载）
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=False,          # ★ 关键：一定是 False
            features_only=True,
            out_indices=out_indices
        )

        # 2. 加载本地 ImageNet 预训练权重
        if weight_path is not None:
            assert os.path.exists(weight_path), f"Weight file not found: {weight_path}"
            state_dict = torch.load(weight_path, map_location="cpu")

            # timm 的 resnet 权重是标准格式，可直接 load
            missing, unexpected = self.feature_extractor.load_state_dict(
                state_dict, strict=False
            )

            if len(missing) > 0:
                print("[INFO] Missing keys when loading teacher:", missing)
            if len(unexpected) > 0:
                print("[INFO] Unexpected keys when loading teacher:", unexpected)

            print(f"[INFO] Loaded pretrained weights from {weight_path}")

        # 3. Teacher 冻结
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        features_t = self.feature_extractor(x)
        features_t[0] = self.pool(features_t[0])
        return features_t
