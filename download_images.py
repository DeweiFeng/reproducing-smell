import os
from duckduckgo_search import DDGS
import requests
from PIL import Image
from io import BytesIO

# List of perfume bases
perfume_bases = [
    "Banana", "Orange", "Pear", "Apple", "Mango", "Peach",
    "Strawberry", "Clove", "Coriander", "Garlic", "Almond", "Cumin"
]

output_dir = "perfume_images"
os.makedirs(output_dir, exist_ok=True)

def download_images(query, folder, max_images=20):
    os.makedirs(folder, exist_ok=True)
    results = DDGS().images(query, max_results=max_images)
    for i, result in enumerate(results):
        url = result["image"]
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(os.path.join(folder, f"{query}_{i}.png"))
        except Exception as e:
            print(f"Error downloading {url}: {e}")

for scent in perfume_bases:
    print(f"Downloading images for {scent}...")
    folder_path = os.path.join(output_dir, scent)
    download_images(scent, folder_path)
