# Copyright 2025 Paras (PR00T - Paras Robotics 00 Technology)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


class VisionModel:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        """
        Initializes the VisionModel and feature extractor.
        """
        print(f"[ViTModel] Loading model: {model_name}")
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model.eval()
        print("[ViTModel] Model loaded successfully.")

    def predict(self, image_path: str) -> str:
        """
        Predicts the class label of an input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Predicted class label.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = self.model.config.id2label[predicted_class_idx]

        print(f"[ViTModel] Prediction: {label}")
        return label
