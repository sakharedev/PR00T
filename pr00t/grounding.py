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
from transformers import AutoProcessor, GroundingDinoModel
from PIL import Image


class Grounding:
    def __init__(self, model_name="IDEA-Research/grounding-dino-tiny"):
        print("[Grounding] Loading model:", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoModel.from_pretrained(model_name).to(self.device)
        print("[Grounding] Ready")

    def detect(self, image_path, text_prompt, box_thresh=0.35, text_thresh=0.25):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)

        # Use official predict helper
        from groundingdino.util.inference import predict, annotate

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )
        return boxes, logits, phrases
