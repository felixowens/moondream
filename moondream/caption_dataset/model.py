import argparse
import importlib
import os
import shutil
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import hashlib
import tqdm

from pathlib import Path
from PIL import Image
import torch

from moondream.torch.moondream import MoondreamModel, MoondreamConfig

from moondream.torch.weights import load_weights_into_model

# from .moondream import MoondreamModel, MoondreamConfig


class CaptioningModel(ABC):
    """Abstract base class for captioning models."""

    @abstractmethod
    def generate_caption(self, image_path: Path) -> str:
        """Generate a caption for the given image."""
        pass


class Moondream(CaptioningModel):
    path = "/home/felix/tools/moondream2/models/saved/moondream_base_finetuned_v1_a2_150_best.safetensors"
    prompt = "\n\nQuestion: Describe this image.\n\nAnswer:"

    def __init__(self):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        elif torch.backends.mps.is_available():
            torch.set_default_device("mps")

        config = MoondreamConfig()
        model = MoondreamModel(config)
        load_weights_into_model(self.path, model)
        self.model = model

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.path,
        #     revision="2025-01-09",
        #     trust_remote_code=True,
        #     device_map={"": "cuda"},
        # )

    def generate_caption(self, image_path: Path) -> str:
        img = Image.open(image_path)
        encoded_image = self.model.encode_image(img)

        return self.model.query(encoded_image, self.prompt)["answer"]  # type: ignore


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NamingStrategy(ABC):
    """Abstract base class for naming strategies."""

    @abstractmethod
    def generate_name(self, original_path: Path, index: int, **kwargs) -> str:
        """Generate a filename for the target image and caption."""
        pass


class SequentialNamingStrategy(NamingStrategy):
    """Names files sequentially with a prefix."""

    def __init__(self, prefix: str = "image", padding: int = 6):
        self.prefix = prefix
        self.padding = padding

    def generate_name(self, original_path: Path, index: int, **kwargs) -> str:
        return f"{self.prefix}_{str(index).zfill(self.padding)}"


class PreserveOriginalNamingStrategy(NamingStrategy):
    """Preserves original filenames."""

    def generate_name(self, original_path: Path, index: int, **kwargs) -> str:
        return original_path.stem


class CaptioningPipeline:
    """Pipeline for captioning images and organizing the output."""

    def __init__(
        self,
        model: CaptioningModel,
        source_dir: str,
        target_dir: str,
        naming_strategy: Optional[NamingStrategy] = None,
        cache_file: Optional[str] = None,
        extensions: List[str] = [],
    ):
        self.model = model
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.naming_strategy = naming_strategy or SequentialNamingStrategy()
        self.cache_file = (
            Path(cache_file) if cache_file else Path(target_dir) / "caption_cache.json"
        )
        self.extensions = extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".webp",
        ]
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save the cache to disk."""
        os.makedirs(self.cache_file.parent, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for the file to detect changes."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _get_image_files(self) -> List[Path]:
        """Get all image files from the source directory."""
        image_files = []
        for ext in self.extensions:
            image_files.extend(list(self.source_dir.glob(f"**/*{ext}")))
        return sorted(image_files)

    def process(self) -> None:
        """Process all images in the source directory."""
        image_files = self._get_image_files()
        logger.info(f"Found {len(image_files)} images to process")

        # Create target directory if it doesn't exist
        os.makedirs(self.target_dir, exist_ok=True)

        for idx, image_path in tqdm.tqdm(
            enumerate(image_files), total=len(image_files)
        ):
            try:
                self._process_single_image(image_path, idx)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

        # Save the updated cache
        self._save_cache()
        logger.info("Processing complete")

    def _process_single_image(self, image_path: Path, index: int) -> None:
        """Process a single image."""
        rel_path = str(image_path.relative_to(self.source_dir))
        file_hash = self._get_file_hash(image_path)

        # Check if the image has already been processed and hasn't changed
        if rel_path in self.cache and self.cache[rel_path]["hash"] == file_hash:
            logger.debug(f"Skipping {rel_path} (already processed, no changes)")
            return

        # Generate the target filename
        target_name = self.naming_strategy.generate_name(image_path, index)
        target_image_path = self.target_dir / f"{target_name}{image_path.suffix}"
        target_caption_path = self.target_dir / f"{target_name}.txt"

        # Generate caption
        logger.info(f"Generating caption for {rel_path}")
        caption = self.model.generate_caption(image_path)

        # Copy the image to the target directory
        shutil.copy2(image_path, target_image_path)

        # Write the caption to a text file
        with open(target_caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

        # Update the cache
        self.cache[rel_path] = {
            "hash": file_hash,
            "caption": caption,
            "target_name": target_name,
        }

        logger.info(f"Processed {rel_path} -> {target_name}")


# Add script arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Captioning pipeline")
    parser.add_argument(
        "-s", "--source_dir", type=str, required=True, help="Source directory"
    )
    parser.add_argument(
        "-t", "--target_dir", type=str, required=True, help="Target directory"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name",
        choices=["moondream"],
        default="moondream",
    )
    return parser.parse_args()


# def load_model(model_name: str) -> CaptioningModel:
#     model_path = Path(f"models/{model_name}.py")
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model {model_name} not found in models directory")
#     module = importlib.import_module(f"models.{model_name}")

#     # Get the correct class name (first letter capitalized)
#     class_name = model_name.capitalize()
#     if not hasattr(module, class_name):
#         raise AttributeError(f"Module {model_name} has no attribute {class_name}")

#     return getattr(module, class_name)()  # Instantiate the model class


if __name__ == "__main__":
    args = parse_arguments()
    model = Moondream()
    pipeline = CaptioningPipeline(model, args.source_dir, args.target_dir)
    pipeline.process()
