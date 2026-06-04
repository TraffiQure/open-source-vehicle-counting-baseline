"""Stage-2 cascade inference for FHWA 13-class vehicle labels."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_b0

LAYER2_ART_TO_FHWA = {"8": 8, "9": 9, "10": 10, "multi_trailer": 13}
LAYER2_SU_TO_FHWA = {"5": 5, "6": 6, "7": 7}

_IMG_SIZE = 224


def _build_layer2_transform():
    # Matches train_stage2_axle.make_transforms()[1].
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose(
        [
            T.Resize((_IMG_SIZE, _IMG_SIZE)),
            T.ToTensor(),
            normalize,
        ]
    )


def _build_layer2_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_idx_to_label(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid classes.json at {path}: expected non-empty object")

    key_indices = [int(k) for k in raw if str(k).isdigit()]
    value_indices = [int(v) for v in raw.values() if str(v).isdigit()]
    expected_indices = list(range(len(raw)))
    keys_are_indices = len(key_indices) == len(raw) and sorted(key_indices) == expected_indices
    values_are_indices = (
        len(value_indices) == len(raw) and sorted(value_indices) == expected_indices
    )

    if keys_are_indices:
        idx_to_label = {int(k): str(v) for k, v in raw.items()}
    elif values_are_indices:
        idx_to_label = {int(v): str(k) for k, v in raw.items()}
    else:
        raise ValueError(
            f"Invalid classes.json at {path}: expected one side to be a contiguous "
            f"0..{len(raw) - 1} index mapping"
        )

    if len(idx_to_label) != len(raw):
        raise ValueError(f"Invalid classes.json at {path}: duplicate class indices")
    return idx_to_label


def _validate_labels(
    *,
    layer_name: str,
    idx_to_label: dict[int, str],
    valid_labels: set[str],
) -> None:
    unexpected = sorted({label for label in idx_to_label.values() if label not in valid_labels})
    if unexpected:
        raise ValueError(
            f"{layer_name} classes.json has unsupported labels: {unexpected}. "
            f"Expected labels drawn from: {sorted(valid_labels)}"
        )


def _load_checkpoint_model(
    checkpoint_path: Path,
    build_model,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing a 'model' state_dict entry"
        )

    model = build_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_label(
    model: nn.Module,
    transform,
    idx_to_label: dict[int, str],
    crop_bgr_np: np.ndarray,
    device: torch.device,
) -> str:
    if crop_bgr_np.ndim != 3 or crop_bgr_np.shape[2] != 3:
        raise ValueError(
            f"Expected crop with shape HxWx3, got {tuple(crop_bgr_np.shape)}"
        )
    if crop_bgr_np.dtype != np.uint8:
        crop_bgr_np = np.ascontiguousarray(crop_bgr_np).astype(np.uint8)

    crop_rgb = cv2.cvtColor(crop_bgr_np, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_idx = int(model(tensor).argmax(dim=1).item())
    try:
        return idx_to_label[pred_idx]
    except KeyError as exc:
        raise RuntimeError(f"Predicted unknown class index {pred_idx}") from exc


class Stage2Cascade:
    def __init__(
        self,
        layer2_art_ckpt: Path | str,
        layer2_su_ckpt: Path | str,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)

        self.layer2_art_ckpt = Path(layer2_art_ckpt)
        self.layer2_su_ckpt = Path(layer2_su_ckpt)

        self.layer2_art_idx_to_label = _load_idx_to_label(
            self.layer2_art_ckpt.parent / "classes.json"
        )
        self.layer2_su_idx_to_label = _load_idx_to_label(
            self.layer2_su_ckpt.parent / "classes.json"
        )

        _validate_labels(
            layer_name="Layer 2 articulated",
            idx_to_label=self.layer2_art_idx_to_label,
            valid_labels=set(LAYER2_ART_TO_FHWA),
        )
        _validate_labels(
            layer_name="Layer 2 single-unit",
            idx_to_label=self.layer2_su_idx_to_label,
            valid_labels=set(LAYER2_SU_TO_FHWA),
        )

        self.layer2_art_model = _load_checkpoint_model(
            self.layer2_art_ckpt,
            _build_layer2_model,
            len(self.layer2_art_idx_to_label),
            self.device,
        )
        self.layer2_su_model = _load_checkpoint_model(
            self.layer2_su_ckpt,
            _build_layer2_model,
            len(self.layer2_su_idx_to_label),
            self.device,
        )

        self.layer2_transform = _build_layer2_transform()

    def classify(self, routing: str, crop_bgr_np: np.ndarray) -> int:
        if routing == "axle_single_unit":
            layer2_label = _predict_label(
                self.layer2_su_model,
                self.layer2_transform,
                self.layer2_su_idx_to_label,
                crop_bgr_np,
                self.device,
            )
            return LAYER2_SU_TO_FHWA[layer2_label]
        if routing == "axle_articulated":
            layer2_label = _predict_label(
                self.layer2_art_model,
                self.layer2_transform,
                self.layer2_art_idx_to_label,
                crop_bgr_np,
                self.device,
            )
            return LAYER2_ART_TO_FHWA[layer2_label]
        raise ValueError(f"Unsupported routing: {routing!r}")
