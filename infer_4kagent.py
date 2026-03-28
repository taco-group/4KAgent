import os
import gc
import torch
import argparse
from pathlib import Path

from pipeline.the4kagent_pipeline import The4KAgent
from utils.custom_types import *


def parse_args():
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument("--input_dir", type=str, default="./dataset/LQ", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./outputs/LQ_results", help="Path to the output directory")
    parser.add_argument("--profile_name", type=str, default="", help="Profile Name for the 4KAgent")
    parser.add_argument("--tool_run_gpu_id", type=int, default=0, help="GPU ID to run tools the toolbox")
    parser.add_argument("--llm_provider", type=str, default="gpt4",
                        choices=["gpt4", "azure", "minimax"],
                        help="LLM provider for the reasoning agent (default: gpt4)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile_name = args.profile_name
    tool_run_gpu_id = args.tool_run_gpu_id
    llm_provider = args.llm_provider

    output_dir.mkdir(parents=True, exist_ok=True)

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    images = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in exts])

    if not images:
        print(f"No images found in {input_dir}")
        return

    for image_path in images:
        print(f"\n[Checking]: {image_path.name}")

        image_output_dir = output_dir / image_path.stem
        result_png_candidates = list(image_output_dir.glob("*/result.png"))

        if result_png_candidates:
            print(f"[Skip] Already processed: {image_path.name}")
            continue

        image_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Processing] {image_path.name}")
        agent = The4KAgent(
            input_path=image_path,
            output_dir=image_output_dir,
            with_retrieval=True,
            with_reflection=True,
            silent=False,
            tool_run_gpu_id=tool_run_gpu_id,
            profile_name=profile_name,
            llm_provider=llm_provider
        )

        agent.run()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
