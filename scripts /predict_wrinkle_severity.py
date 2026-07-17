"""Run multimodal inference using a skin image and sensor measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from models.fusion_model import FusionModel


SENSOR_DIM = 10
NUM_CLASSES = 3

CLASS_NAMES = {
    0: "Mild Wrinkles",
    1: "Moderate Wrinkles",
    2: "Severe Wrinkles",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model(weights_path: Path) -> FusionModel:
    """Load trained model weights and prepare the model for inference."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = FusionModel(
        sensor_dim=SENSOR_DIM,
        num_classes=NUM_CLASSES,
    )

    try:
        state_dict = torch.load(
            weights_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model weights from {weights_path}: {exc}"
        ) from exc

    model.to(DEVICE)
    model.eval()

    return model


def load_image(image_path: Path) -> torch.Tensor:
    """Load and preprocess one RGB image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid or unsupported image: {image_path}") from exc


def load_sensor_data(sensor_path: Path) -> torch.Tensor:
    """Load and validate sensor measurements from a NumPy file."""
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Sensor file not found: {sensor_path}")

    try:
        sensor_values = np.load(sensor_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Could not load sensor data from {sensor_path}: {exc}"
        ) from exc

    sensor_values = np.asarray(sensor_values, dtype=np.float32).reshape(-1)

    if sensor_values.size != SENSOR_DIM:
        raise ValueError(
            f"Expected {SENSOR_DIM} sensor values, "
            f"but received {sensor_values.size}."
        )

    if not np.all(np.isfinite(sensor_values)):
        raise ValueError("Sensor data contains NaN or infinite values.")

    return (
        torch.from_numpy(sensor_values)
        .reshape(1, SENSOR_DIM)
        .to(DEVICE)
    )


def run_inference(
    model: FusionModel,
    image_path: Path,
    sensor_path: Path,
) -> dict[str, Any]:
    """Run multimodal inference and return structured prediction results."""
    image_tensor = load_image(image_path)
    sensor_tensor = load_sensor_data(sensor_path)

    with torch.inference_mode():
        logits = model(image_tensor, sensor_tensor)

        if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
            raise RuntimeError(
                "Unexpected model output shape. "
                f"Expected (batch_size, {NUM_CLASSES}), "
                f"received {tuple(logits.shape)}."
            )

        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    return {
        "predicted_class": predicted_index,
        "predicted_label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            CLASS_NAMES[index]: float(probability.item())
            for index, probability in enumerate(probabilities)
        },
        "device": str(DEVICE),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multimodal wrinkle-severity inference using "
            "a skin image and sensor measurements."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input skin image.",
    )
    parser.add_argument(
        "--sensors",
        type=Path,
        required=True,
        help="Path to a NumPy .npy file containing sensor values.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the trained model state dictionary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return the prediction as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line inference workflow."""
    args = parse_arguments()

    try:
        model = load_model(args.weights)
        result = run_inference(
            model=model,
            image_path=args.image,
            sensor_path=args.sensors,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.json:
        print(json.dumps(result, indent=4))
    else:
        print(f"Predicted class: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Device: {result['device']}")

        print("\nClass probabilities:")
        for label, probability in result["probabilities"].items():
            print(f"  {label}: {probability:.2%}")


if __name__ == "__main__":
    main()
