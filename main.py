# main.py

from pr00t.llava_interface import LlavaInterface
from pr00t.llava_parser import parse_llava_output
from pr00t.grounding import Grounding
from IPython.display import Image, display
import warnings

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="The `vocab_size` argument is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)

# === Config ===
image_path = "./images/industry.jpg"
question = (
    "Identify all objects visible in the scene and describe their location and "
    "relative position. Highlight any objects that can be grasped or picked up by a robot arm."
)

# === Step 1: Run LLaVA ===
llava = LlavaInterface()
display(Image(filename=image_path, width=300))
raw_response = llava.ask(image_path=image_path, question=question)

# === Step 2: Parse LLaVA Output ===
print("\n=== LLaVA Raw Output ===\n")
print(raw_response)

parsed_objects = parse_llava_output(raw_response)

print("\n=== Parsed Objects ===\n")
for obj in parsed_objects:
    print(
        f"Object: {obj['name']}, Position: {obj['position']}, Graspable: {obj['graspable']}"
    )

# === Step 3: Run Grounding DINO ===
print("\n=== Running Grounding DINO ===\n")
grounding = Grounding()

# Combine object names into a prompt for detection
object_names = [obj["name"] for obj in parsed_objects if obj["graspable"]]
prompt = ". ".join(object_names) + "."

boxes, _, labels = grounding.detect(image_path=image_path, text_prompt=prompt)

# === Step 4: Output Final Grounded Objects ===
print("\n=== Grounded Objects with Bounding Boxes ===\n")
for label, box in zip(labels, boxes):
    print(f"Object: {label}, Box: {box.tolist()}")
