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


import re


def parse_llava_output(text):
    """
    Parses LLaVA output to extract object info.
    """
    objects = []

    # Normalize
    text = text.lower()

    # Example heuristic: Find mentions of "gears", "cogs", etc.
    pattern = r"(gear|cog|component|object)s?[^.]*"

    matches = re.findall(pattern, text)
    seen = set()

    for match in matches:
        if match not in seen:
            seen.add(match)
            objects.append(
                {
                    "name": match.strip(),
                    "position": "unknown",  # Could use position hints in future
                    "graspable": "pick up" in text or "grasp" in text,
                }
            )

    # If nothing found, fallback to at least say 'unknown object'
    if not objects:
        objects.append({"name": "unknown", "position": "unknown", "graspable": False})

    return objects
