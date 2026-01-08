import argparse
import io

import torch

from flask import Flask, jsonify, request
from PIL import Image

from service.detector import DetectorConfig, FabricAnomalyDetector


def create_app(config: DetectorConfig) -> Flask:
    app = Flask(__name__)
    detector = FabricAnomalyDetector(config)

    @app.get("/health")
    def health():
        return jsonify(
            ok=True,
            model=config.obj,
            threshold=detector.calibration.threshold if detector.calibration else None,
        )

    @app.post("/detect")
    def detect():
        if "image" not in request.files:
            return jsonify(ok=False, error="缺少 image 文件字段"), 400

        file_storage = request.files["image"]
        if not file_storage.filename:
            return jsonify(ok=False, error="未提供图片文件"), 400

        image_bytes = file_storage.read()
        if not image_bytes:
            return jsonify(ok=False, error="图片内容为空"), 400

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            return jsonify(ok=False, error=f"图片解析失败: {exc}"), 400

        try:
            raw_score, score, is_defect = detector.score_image(image)
        except Exception as exc:
            return jsonify(ok=False, error=f"检测失败: {exc}"), 500

        return jsonify(
            ok=True,
            raw_score=raw_score,
            score=score,
            is_defect=is_defect,
            threshold=detector.calibration.threshold if detector.calibration else None,
            min_score=detector.calibration.min_score if detector.calibration else None,
            max_score=detector.calibration.max_score if detector.calibration else None,
        )

    return app


def parse_args() -> DetectorConfig:
    parser = argparse.ArgumentParser(description="DBFAD 推理服务")
    parser.add_argument("--data-path", default="dataset/MVTEC")
    parser.add_argument("--obj", default="nuser3")
    parser.add_argument("--model-dir", default="results/models")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--img-resize", type=int, default=256)
    parser.add_argument("--img-cropsize", type=int, default=256)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--calibration-file", default="calibration.json")
    args = parser.parse_args()
    return DetectorConfig(
        data_path=args.data_path,
        obj=args.obj,
        model_dir=args.model_dir,
        device=args.device,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
        threshold_percentile=args.threshold_percentile,
        calibration_file=args.calibration_file,
    )


if __name__ == "__main__":
    cfg = parse_args()
    app = create_app(cfg)
    app.run(host="0.0.0.0", port=5000)
