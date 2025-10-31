# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool


def save_video_to_file(video, save_dir: str, test_idx: int, seed: int, prompt: str, task: str, fps: int):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
    suffix = '.mp4'
    file_name = f"{task}_{test_idx:04d}_seed{seed}_{formatted_prompt}_{formatted_time}{suffix}"
    save_file = os.path.join(save_dir, file_name)
    logging.info(f"Saving generated video to {save_file}")
    save_video(
        tensor=video[None],
        save_file=save_file,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))
    return 0


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    if args.task == "i2v-A14B":
        assert args.image is not None or args.image_path_file is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    
    if args.lora_dir is not None and args.sample_solver != "euler":
        warnings.warn(f"Please use euler solver because it's used in distillation")

    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()), help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
    parser.add_argument("--lora_dir", type=str, default=None, help="The path to the lora directory.")
    parser.add_argument("--save_dir", type=str, default="test_results", help="The directory to save the generated videos.")
    parser.add_argument("--offload_model", type=str2bool, default=False, help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--ulysses_size", type=int, default=8, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
    parser.add_argument("--prompt_file", type=str, default=None, help="A txt file of the prompt list.")
    parser.add_argument("--base_seed", type=int, default=42, help="The seed to use for generating the video.")
    parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
    parser.add_argument("--image_path_file", type=str, default=None, help="A txt file of the image path list")
    parser.add_argument("--sample_solver", type=str, default='euler', choices=['euler', 'unipc', 'dpm++'], help="The solver used to sample.")
    parser.add_argument("--sample_steps", type=int, default=4, help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale.")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))  # 所在进程编号
    world_size = int(os.getenv("WORLD_SIZE", 1))  # 总进程数
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # 所在机器的 GPU 编号
    device = local_rank
    _init_logging(rank)
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # prompt
    if args.prompt_file is not None and os.path.isfile(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            prompt_list = f.read().splitlines()
    else:
        prompt_list = [args.prompt]

    # image
    if args.image_path_file is not None and os.path.isfile(args.image_path_file):
        with open(args.image_path_file, 'r') as f:
            image_path_list = f.read().splitlines()
    elif args.image is not None:
        image_path_list = [args.image]
    else:
        image_path_list = [None] * len(prompt_list)
    image_list = []
    for image_path in image_path_list:
        if image_path is not None:
            image_list.append(Image.open(image_path).convert("RGB"))
        else:
            image_list.append(None)
    assert len(prompt_list) == len(image_list), "Please make sure the length of prompt_list and the image_list be the same."
    
    # generate
    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            lora_dir=args.lora_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        for i, prompt in enumerate(prompt_list):
            logging.info(f"Generating video ... index:{i} seed:{args.base_seed} prompt:{prompt}")
            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,  # type: ignore
                offload_model=args.offload_model)
            if rank == 0:
                save_video_to_file(
                    video=video,
                    save_dir=args.save_dir,
                    test_idx=i,
                    seed=args.base_seed,  # type: ignore
                    prompt=prompt,
                    task=args.task,
                    fps=cfg.sample_fps
                )
            del video
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            lora_dir=args.lora_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        for i, (prompt, img) in enumerate(zip(prompt_list, image_list)):
            logging.info(f"Generating video ... index:{i} seed:{args.base_seed} prompt:{prompt}")
            video = wan_i2v.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,  # type: ignore
                offload_model=args.offload_model)
            if rank == 0:
                save_video_to_file(
                    video=video,
                    save_dir=args.save_dir,
                    test_idx=i,
                    seed=args.base_seed,  # type: ignore
                    prompt=prompt,
                    task=args.task,
                    fps=cfg.sample_fps
                )
            del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
