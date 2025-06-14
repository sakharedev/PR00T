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

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch


class LlavaInterface:
    def __init__(self, model_name="bczhou/tiny-llava-v1-hf"):
        print(f"[TinyLLaVA] Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # low_cpu_mem_usage=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        print("[TinyLLaVA] Model and processor ready.")

    def ask(self, image_path: str, question: str, max_tokens=200) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        inputs = self.processor(prompt, image, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        # format from README: skip first tokens to drop the prompt prefix
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        # strip out prompt prefix
        return answer.replace(prompt, "").strip()
