from contextlib import contextmanager
import datetime
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Any, Optional, List, Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import math
from safetensors.torch import save_file
from PIL import Image
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
from aim import Run
from torch.utils.data import DataLoader

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden, _lm_head, TextConfig


profiling_stats = {}


@contextmanager
def profile(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if name not in profiling_stats:
        profiling_stats[name] = []
    profiling_stats[name].append(elapsed_time)


def print_profiling_stats():
    print("\n--- Profiling Statistics ---")
    for operation, times in profiling_stats.items():
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        percent = total_time / sum([sum(t) for t in profiling_stats.values()]) * 100
        print(
            f"{operation}: {avg_time:.4f}s avg, {total_time:.2f}s total ({percent:.1f}%)"
        )


@dataclass
class TrainingConfig:
    """
    Configuration for the fine-tuning process.

    Contains all hyperparameters and file paths needed for training.
    """

    # Model paths
    model_path: str
    output_path: str

    # Dataset paths
    dataset_json: str
    image_dir: str

    # Training hyperparameters
    learning_rate: float = 3e-6
    epochs: int = 100
    grad_accum_steps: int = 35
    answer_eos: str = "<|endoftext|>"

    # Evaluation settings
    eval_split: float = 0.1  # 10% of dataset for evaluation
    eval_frequency: int = 5  # Evaluate every 5 epochs

    # Checkpointing
    save_frequency: int = 10  # Save every 10 epochs
    max_checkpoints: int = 3  # Keep only the latest 3 checkpoints


def lr_schedule(step: int, max_steps: int, config: TrainingConfig) -> float:
    """
    Implements a learning rate schedule with warmup and cosine decay.

    For beginners: This function gradually increases the learning rate during the first 10%
    of training (warmup phase), then gradually decreases it following a cosine curve.
    This helps stabilize training early on, then fine-tune more precisely later.

    Args:
        step: Current training step
        max_steps: Total number of training steps
        config: Training configuration with learning rate

    Returns:
        The calculated learning rate for the current step
    """
    x = step / max_steps
    print(f"x: {x} = {step} / {max_steps}")

    # Warmup phase (first 10% of training)
    if x < 0.1:
        return 0.1 * config.learning_rate + 0.9 * config.learning_rate * x / 0.1
    # Cosine decay phase (remaining 90% of training)
    else:
        return (
            0.1 * config.learning_rate
            + 0.9 * config.learning_rate * (1 + math.cos(math.pi * (x - 0.1))) / 2
        )


def text_loss(
    inputs_embeds: torch.Tensor, w: nn.Module, labels: torch.Tensor, config: TextConfig
) -> torch.Tensor:
    """
    Calculates the cross-entropy loss between predicted and actual tokens.

    For beginners: This function processes the embedded inputs through the model's hidden layers,
    generates predictions for the next token at each position, and compares these predictions
    with the actual next tokens using cross-entropy loss to measure prediction accuracy.

    Args:
        inputs_embeds: Embedded input sequences
        w: Text model weights
        labels: Target token IDs
        config: Text model configuration

    Returns:
        Computed loss value
    """
    batch_size, q_len, _ = inputs_embeds.shape

    # Generate hidden representations
    hidden_BTC = _produce_hidden(inputs_embeds, w, config)

    # Generate logits (raw predictions)
    lm_logits = _lm_head(hidden_BTC, w)

    loss = None
    if labels is not None:
        _, _, l_len = labels.shape

        # Align predictions with labels
        shift_index = (q_len - l_len) - 1
        shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
        shifted_labels = labels.contiguous()

        # Calculate cross-entropy loss
        loss = nn.CrossEntropyLoss()(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )

    return loss


class ImageTextDataset(Dataset):
    """
    Dataset for image and text pairs used in fine-tuning.

    For beginners: This class loads images and their captions from a JSON file,
    formatting them as question-answer pairs for training the model to describe images.
    Each sample includes an image and a corresponding QA pair.
    """

    def __init__(
        self, json_file: str, image_dir: str, answer_eos: str, split: str = "train"
    ):
        """
        Initialize the dataset.

        Args:
            json_file: Path to the JSON file containing image metadata
            image_dir: Directory containing the images
            answer_eos: End of sequence token for answers
            split: Dataset split (e.g., "train"), kept for compatibility
        """
        self.image_dir = image_dir
        self.answer_eos = answer_eos

        # Load dataset
        with open(json_file, "r") as f:
            self.data = json.load(f)

        # Convert dictionary to list for easier indexing
        self.data_list = list(self.data.items())
        print(f"Initialized dataset with {len(self.data_list)} samples.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the image and a QA pair
        """
        key, sample = self.data_list[idx]

        # Load the image
        image_path = os.path.join(self.image_dir, sample["filename"])
        image = Image.open(image_path).convert("RGB")

        # Extract the caption
        description = sample["file_attributes"]["caption"]

        # Format the QA pair
        qa = {
            "question": "\n\nQuestion: Describe this image.\n\nAnswer:",
            "answer": f"{description}{self.answer_eos}",
        }

        return {
            "image": image,
            "qa": qa,
        }


def evaluate_model(
    model: MoondreamModel, eval_dataset: Dataset, device: torch.device, aim_run: Run
) -> Dict[str, float]:
    """
    Evaluate model performance on the evaluation dataset.

    Args:
        model: The model to evaluate
        eval_dataset: Evaluation dataset
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    perplexity_sum = 0.0

    with torch.no_grad():
        for i in tqdm(range(len(eval_dataset)), desc="Evaluating"):  # type: ignore
            sample = eval_dataset[i]

            # Process image
            img_emb = model._run_vision_encoder(sample["image"])

            # Process question and answer
            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]], device=device),
                model.text,
            )

            question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=device),
                model.text,
            ).squeeze(0)

            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=device),
                model.text,
            ).squeeze(0)

            # TODO: need to correct this to track the average length of the question and answer over the entire eval dataset so we can check over time
            aim_run.track(
                len(question_tokens), name="eval/question_tokens_length", step=i
            )
            aim_run.track(len(answer_tokens), name="eval/answer_tokens_length", step=i)

            # Combine embeddings
            inputs_embeds = torch.cat(
                [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
            )

            # Calculate loss
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=torch.tensor([[answer_tokens]], device=device),
                config=model.config.text,
            )

            total_loss += loss.item()
            perplexity = torch.exp(loss)
            perplexity_sum += perplexity.item()

    avg_loss = total_loss / len(eval_dataset)  # type: ignore
    avg_perplexity = perplexity_sum / len(eval_dataset)  # type: ignore

    model.train()

    return {"eval_loss": avg_loss, "eval_perplexity": avg_perplexity}


def manage_checkpoints(
    checkpoints_dir: Path, filename: str, max_checkpoints: int
) -> List[str]:
    """
    Manage checkpoint files, keeping only the specified maximum number.

    Args:
        checkpoints_dir: Directory containing checkpoints
        filename: Name of the newly created checkpoint
        max_checkpoints: Maximum number of checkpoints to keep

    Returns:
        List of remaining checkpoint filenames
    """
    checkpoints = sorted(
        [
            f
            for f in os.listdir(checkpoints_dir)
            if f.startswith("checkpoint_") and f.endswith(".safetensors")
        ]
    )

    # If we exceed the maximum number of checkpoints, remove the oldest ones
    while len(checkpoints) >= max_checkpoints:
        oldest_checkpoint = os.path.join(checkpoints_dir, checkpoints[0])
        os.remove(oldest_checkpoint)
        print(f"Removed old checkpoint: {oldest_checkpoint}")
        checkpoints = checkpoints[1:]

    # Add the new checkpoint to the list
    checkpoints.append(filename)
    return checkpoints


def clear_or_backup_checkpoints(output_path):
    checkpoints_dir = Path(output_path).parent / "checkpoints"
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        # Ask for confirmation
        response = input(
            f"Checkpoints directory {checkpoints_dir} is not empty. Enter 'clear' to remove existing files, 'backup' to rename the directory, or anything else to continue: "
        )
        if response.lower() == "clear":
            for file in checkpoints_dir.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"Cleared checkpoints directory: {checkpoints_dir}")
        elif response.lower() == "backup":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = checkpoints_dir.with_name(
                f"{checkpoints_dir.name}_backup_{timestamp}"
            )
            shutil.move(str(checkpoints_dir), str(backup_dir))
            print(f"Backed up checkpoints directory to: {backup_dir}")
            # Create a new empty directory
            checkpoints_dir.mkdir(parents=True, exist_ok=True)


def train_model(config: TrainingConfig) -> None:
    """
    Train the model using the specified configuration.

    For beginners: This function handles the complete fine-tuning process:
    1. It loads the pre-trained model
    2. Sets up optimization and training parameters
    3. For each image, it processes the image, question, and answer
    4. It calculates how well the model is doing and updates its weights
    5. It tracks progress and saves the improved model

    Args:
        config: Training configuration parameters
    """
    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.set_default_device(device)

    # Initialize Aim run for experiment tracking
    aim_run = Run(experiment="moondream-ft")

    description, tags = get_experiment_info()
    if description:
        aim_run["description"] = description
    if tags:
        aim_run["tags"] = tags

    aim_run["hparams"] = {
        "epochs": config.epochs,
        "grad_accum_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "eval_split": config.eval_split,
        "eval_frequency": config.eval_frequency,
        "save_frequency": config.save_frequency,
        "max_checkpoints": config.max_checkpoints,
        "device": str(device),
        "model_path": config.model_path,
        "output_path": config.output_path,
        "dataset_json": config.dataset_json,
        "image_dir": config.image_dir,
    }

    # Set up dataset
    full_dataset = ImageTextDataset(
        config.dataset_json, config.image_dir, config.answer_eos
    )

    # Split dataset into train and eval
    eval_size = int(len(full_dataset) * config.eval_split)
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator(device="cuda").manual_seed(42),
    )

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Eval dataset size: {len(eval_dataset)} samples")

    # Create checkpoints directory
    output_path = Path(config.output_path)
    checkpoints_dir = output_path.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total steps
    steps_per_epoch = len(train_dataset)

    total_steps = config.epochs * steps_per_epoch // config.grad_accum_steps

    if len(train_dataset) % config.grad_accum_steps != 0:
        closest_factor = max(
            [
                i
                for i in range(1, len(train_dataset) + 1)
                if len(train_dataset) % i == 0 and i <= len(train_dataset)
            ]
        )
        print(
            f"Warning: grad_accum_steps ({config.grad_accum_steps}) is not a factor of dataset size ({len(train_dataset)})"
        )
        print(f"Adjusting to {closest_factor}")
        config.grad_accum_steps = closest_factor
        # Update total steps calculation
        total_steps = config.epochs * steps_per_epoch // config.grad_accum_steps

    # Set up progress tracking
    pbar = tqdm(total=total_steps, desc="Training Progress")
    step_count = 0
    total_tokens = 0
    epoch_loss = 0.0
    samples_in_epoch = 0

    # Initialize model
    model_config = MoondreamConfig()
    model = MoondreamModel(model_config)
    load_weights_into_model(config.model_path, model)

    # Pre-compute and cache embeddings for all samples
    print("Pre-computing embeddings for all samples...")
    cached_embeddings = {}
    with torch.no_grad():
        for idx in tqdm(range(len(full_dataset)), desc="Caching Embeddings"):
            sample = full_dataset[idx]
            sample_id = idx  # Use index as a unique identifier

            # Cache image embeddings
            img_emb = model._run_vision_encoder(sample["image"])

            # Cache text embeddings
            question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=model.device),  # type: ignore
                model.text,
            ).squeeze(0)

            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=model.device),  # type: ignore
                model.text,
            ).squeeze(0)

            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),  # type: ignore
                model.text,
            )

            cached_embeddings[sample_id] = {
                "img_emb": img_emb,
                "question_emb": question_emb,
                "question_tokens": question_tokens,
                "answer_emb": answer_emb,
                "answer_tokens": answer_tokens,
                "bos_emb": bos_emb,
            }

    # Set up optimizer
    optimizer = AdamW8bit(
        [{"params": model.text.parameters()}],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    def custom_collate_fn(batch):
        """
        Custom collate function for DataLoader to handle PIL images
        """
        images = [item["image"] for item in batch]
        qa_pairs = [item["qa"] for item in batch]

        # Return as a dictionary with lists
        return {
            "image": images,  # List of PIL images
            "qa": qa_pairs,  # List of qa pairs
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Or larger if memory allows
        shuffle=True,
        num_workers=2,  # Use multiple workers for parallel loading
        pin_memory=True,  # Faster transfer to GPU
        prefetch_factor=2,  # Pre-fetch samples
        generator=torch.Generator(device="cuda"),
        collate_fn=custom_collate_fn,
    )

    # Calculate total steps based on DataLoader
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch // config.grad_accum_steps

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Reset epoch metrics
        epoch_loss = 0.0
        samples_in_epoch = 0
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()  # type: ignore

        # Shuffle the training indices for each epoch
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for batch_idx, batch in enumerate(train_loader):
            with profile("get_sample"):
                step_count += 1
                sample = batch  # Get the original index
                sample_id = train_dataset.indices[batch_idx]

            # TODO:
            # For the cached embeddings, you'd need to adapt this approach
            # You could either:
            # 1. Access the original dataset index from your train_dataset
            # 2. Modify your caching strategy to work with batches directly

            with profile("get_cached_embeddings"):
                # Use cached embeddings instead of recomputing
                img_emb = cached_embeddings[sample_id]["img_emb"]
                question_emb = cached_embeddings[sample_id]["question_emb"]
                answer_tokens = cached_embeddings[sample_id]["answer_tokens"]
                answer_emb = cached_embeddings[sample_id]["answer_emb"]
                bos_emb = cached_embeddings[sample_id]["bos_emb"]
                # Combine embeddings
                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
                )

            with profile("calculate_text_loss"):
                # Calculate loss
                loss = text_loss(
                    inputs_embeds=inputs_embeds,
                    w=model.text,
                    labels=torch.tensor([[answer_tokens]], device=model.device),  # type: ignore
                    config=model_config.text,
                )

            with profile("track_metrics"):
                # Track tokens processed
                total_tokens += len(answer_tokens)

                # Track loss for epoch calculation
                epoch_loss += loss.item()
                samples_in_epoch += 1

            with profile("backpropagate"):
                # Backpropagate
                loss.backward()

            with profile("update_weights"):
                # Update weights after accumulating gradients
                if step_count % config.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update learning rate
                    current_step = step_count // config.grad_accum_steps
                    print(f"Current step: {current_step}")
                    print(f"Total steps: {total_steps}")
                    # TODO: check if lr schedule is bugged

                    lr = lr_schedule(current_step, total_steps, config)
                    print(f"Learning rate: {lr}")
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                    # Calculate perplexity (exponentiated loss)
                    perplexity = torch.exp(loss)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "epoch": epoch,
                            "step": current_step,
                            "loss": loss.item(),
                            "perplexity": perplexity.item(),
                        }
                    )
                    pbar.update(1)

                    # Log metrics with Aim
                    aim_run.track(loss.item(), name="train/loss", step=current_step)
                    aim_run.track(
                        perplexity.item(),
                        name="train/perplexity",
                        step=current_step,
                    )
                    aim_run.track(lr, name="train/learning_rate", step=current_step)
                    aim_run.track(
                        total_tokens, name="train/total_tokens", step=current_step
                    )

        # End of epoch
        end_time.record()  # type: ignore
        torch.cuda.synchronize()
        epoch_time_ms = start_time.elapsed_time(end_time)
        epoch_time_s = epoch_time_ms / 1000

        # Calculate and log epoch metrics
        avg_epoch_loss = epoch_loss / samples_in_epoch
        avg_epoch_perplexity = math.exp(avg_epoch_loss)
        tokens_per_second = total_tokens / epoch_time_s if epoch_time_s > 0 else 0

        aim_run.track(avg_epoch_loss, name="train/epoch_loss", epoch=epoch)
        aim_run.track(avg_epoch_perplexity, name="train/epoch_perplexity", epoch=epoch)
        aim_run.track(epoch_time_s, name="train/epoch_time_seconds", epoch=epoch)
        aim_run.track(tokens_per_second, name="train/tokens_per_second", epoch=epoch)

        print(f"\nEpoch {epoch} stats:")
        print(f"  Avg loss: {avg_epoch_loss:.4f}")
        print(f"  Avg perplexity: {avg_epoch_perplexity:.4f}")
        print(f"  Tokens/second: {tokens_per_second:.2f}")
        print(f"  Epoch time: {epoch_time_s:.2f}s")

        print_profiling_stats()

        # Run evaluation if needed
        if epoch % config.eval_frequency == 0:
            print(f"\nRunning evaluation after epoch {epoch}...")
            eval_metrics = evaluate_model(model, eval_dataset, device, aim_run)

            # Log evaluation metrics
            for metric_name, metric_value in eval_metrics.items():
                aim_run.track(metric_value, name=f"eval/{metric_name}", epoch=epoch)
                print(f"  {metric_name}: {metric_value:.4f}")

        # Save checkpoint if needed
        if epoch % config.save_frequency == 0:
            checkpoint_file = f"checkpoint_epoch_{epoch}.safetensors"
            checkpoint_path = checkpoints_dir / checkpoint_file
            save_file(model.state_dict(), str(checkpoint_path))
            print(f"\nSaved checkpoint at epoch {epoch} to {checkpoint_path}")

            # Manage checkpoints
            manage_checkpoints(checkpoints_dir, checkpoint_file, config.max_checkpoints)

    # Close progress bar
    pbar.close()

    # Save the final fine-tuned model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(output_path))
    print(f"Final model saved to {output_path}")

    # Run final evaluation
    print("\nRunning final evaluation...")
    final_metrics = evaluate_model(model, eval_dataset, device, aim_run)
    for metric_name, metric_value in final_metrics.items():
        aim_run.track(metric_value, name=f"eval/{metric_name}", epoch=config.epochs)
        print(f"  {metric_name}: {metric_value:.4f}")


# Prompt for experiment description and tags
# In the main function, before calling train_model:
def get_experiment_info():
    description = input(
        "Enter a description for this experiment (or press Enter to skip): "
    )
    tags_input = input(
        "Enter tags for this experiment (comma-separated, or press Enter to skip): "
    )
    tags = [tag.strip() for tag in tags_input.split(",")] if tags_input.strip() else []
    return description, tags


def validate_hyperparams(config, dataset_size):
    warnings = []

    # Check grad_accum_steps
    if config.grad_accum_steps > dataset_size:
        warnings.append(
            f"grad_accum_steps ({config.grad_accum_steps}) is larger than dataset size ({dataset_size})"
        )

    if dataset_size % config.grad_accum_steps != 0:
        warnings.append(
            f"grad_accum_steps ({config.grad_accum_steps}) is not a factor of dataset size ({dataset_size})"
        )

    # Check that eval_frequency and save_frequency are sensible
    if config.eval_frequency > config.epochs:
        warnings.append(
            f"eval_frequency ({config.eval_frequency}) is greater than total epochs ({config.epochs})"
        )

    if config.save_frequency > config.epochs:
        warnings.append(
            f"save_frequency ({config.save_frequency}) is greater than total epochs ({config.epochs})"
        )

    # Learning rate checks
    if config.learning_rate > 1e-3:
        warnings.append(
            f"learning_rate ({config.learning_rate}) seems high, typical values for fine-tuning are 1e-4 to 1e-6"
        )

    # Print warnings
    if warnings:
        print("\nHyperparameter validation warnings:")
        for warning in warnings:
            print(f"- {warning}")

        proceed = input("\nDo you want to proceed with training anyway? (y/n): ")
        if proceed.lower() != "y":
            raise ValueError("Training aborted due to hyperparameter validation issues")


def main() -> None:
    """
    Main entry point for the script.

    Sets up the configuration and starts the training process.
    """

    # (to avoid conflicts with existing checkpoints) or handle conflicts gracefully.
    config = TrainingConfig(
        model_path="/home/felix/tools/moondream2/models/saved/moondream_base_finetuned_v1_a2_100.safetensors",
        output_path="/home/felix/tools/moondream2/models/moondream_base_finetuned_v1_a2_150.safetensors",
        dataset_json="/home/felix/projects/dataset-tools/augment_caption_dataset/testing/dataset.json",
        image_dir="/home/felix/projects/dataset-tools/augment_caption_dataset/testing",
        epochs=50,
        grad_accum_steps=246,  # Should match the dataset size
        eval_split=0.1,  # 10% of dataset for evaluation
        eval_frequency=5,  # Evaluate every 5 epochs
        save_frequency=25,  # Save every 10 epochs
        max_checkpoints=5,  # Keep only the last 3 checkpoints
        learning_rate=5e-6,
    )

    temp_dataset = ImageTextDataset(
        config.dataset_json, config.image_dir, config.answer_eos
    )
    validate_hyperparams(config, len(temp_dataset))
    clear_or_backup_checkpoints(config.output_path)

    train_model(config)


if __name__ == "__main__":
    """
    To run: python -m moondream.finetune.finetune_text
    (Replace the paths in the config with your appropriate paths)
    """
    main()
