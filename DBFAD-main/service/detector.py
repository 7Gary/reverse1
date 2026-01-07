from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from datasets.mvtec import MVTecDataset
from models.reverse_Residual_ResNet import reverse_student18
from models.teacherTimm import teacherTimm
from utils.functions import cal_anomaly_maps


@dataclass
class DetectorConfig:
    data_path: str = "dataset/MVTEC"
    obj: str = "nuser3"
    model_dir: str = "results/models"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_resize: int = 256
    img_cropsize: int = 256
    threshold_percentile: float = 99.5
    calibration_file: str = "calibration.json"


@dataclass
class CalibrationStats:
    min_score: float
    max_score: float
    threshold: float

    @property
    def span(self) -> float:
        return max(self.max_score - self.min_score, 1e-8)

    def normalize(self, raw_score: float) -> float:
        return float((raw_score - self.min_score) / self.span)


class FabricAnomalyDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.device = config.device
        self.model_t = None
        self.model_s = None
        self.transform = T.Compose(
            [
                T.Resize(config.img_resize),
                T.CenterCrop(config.img_cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.calibration: Optional[CalibrationStats] = None
        self._load_models()
        self._load_or_calibrate()

    def _load_models(self) -> None:
        self.model_t = teacherTimm(backbone_name="resnet34", out_indices=[0, 1, 2, 3]).to(self.device)
        self.model_s = reverse_student18(DG=False).to(self.device)

        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()

        model_path = os.path.join(
            self.config.model_dir,
            f"{self.config.obj}_Reverse",
            "model_s.pth",
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_s.load_state_dict(checkpoint["model"])
        self.model_s.eval()
        for param in self.model_s.parameters():
            param.requires_grad = False

    def _load_or_calibrate(self) -> None:
        calibration_path = self._calibration_path()
        if os.path.exists(calibration_path):
            with open(calibration_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.calibration = CalibrationStats(
                min_score=float(payload["min_score"]),
                max_score=float(payload["max_score"]),
                threshold=float(payload["threshold"]),
            )
            return

        self.calibration = self.calibrate()
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)
        with open(calibration_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "min_score": self.calibration.min_score,
                    "max_score": self.calibration.max_score,
                    "threshold": self.calibration.threshold,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _calibration_path(self) -> str:
        return os.path.join(
            self.config.model_dir,
            f"{self.config.obj}_Reverse",
            self.config.calibration_file,
        )

    def _score_tensor(self, tensor: torch.Tensor) -> float:
        with torch.no_grad():
            features_t = self.model_t(tensor)
            features_s = self.model_s(features_t)
            anomaly_map = cal_anomaly_maps(features_s, features_t, self.config.img_cropsize)
        return float(np.max(anomaly_map))

    def _iter_training_images(self) -> Iterable[Image.Image]:
        dataset = MVTecDataset(
            self.config.data_path,
            class_name=self.config.obj,
            is_train=True,
            resize=self.config.img_resize,
            cropsize=self.config.img_cropsize,
            blur=False,
            vis=False,
        )
        for idx in range(len(dataset)):
            image_path = dataset.x[idx]
            yield Image.open(image_path).convert("RGB")

    def calibrate(self) -> CalibrationStats:
        raw_scores = []
        for image in self._iter_training_images():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            raw_scores.append(self._score_tensor(tensor))

        if not raw_scores:
            raise RuntimeError("训练集为空，无法计算阈值。请检查数据集路径。")

        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        min_score = float(raw_scores.min())
        max_score = float(raw_scores.max())
        normalized = (raw_scores - min_score) / max(max_score - min_score, 1e-8)
        threshold = float(np.percentile(normalized, self.config.threshold_percentile))
        return CalibrationStats(min_score=min_score, max_score=max_score, threshold=threshold)

    def score_image(self, image: Image.Image) -> Tuple[float, float, bool]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        raw_score = self._score_tensor(tensor)
        if self.calibration is None:
            raise RuntimeError("模型尚未完成阈值校准。")
        score = self.calibration.normalize(raw_score)
        is_defect = score >= self.calibration.threshold
        return raw_score, score, is_defect