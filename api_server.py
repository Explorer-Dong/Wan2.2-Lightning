import argparse
import io
import os
import uuid
import json
import logging
import asyncio
import warnings
import requests
import time
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

import torch
import torch.distributed as dist

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video

warnings.filterwarnings("ignore")


# ====================== 基础工具 ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2v_ckpt_dir", type=str, default="./Wan2.2-T2V-A14B")
    parser.add_argument("--t2v_lora_dir", type=str, default="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928")
    parser.add_argument("--i2v_ckpt_dir", type=str, default="./Wan2.2-I2V-A14B")
    parser.add_argument("--i2v_lora_dir", type=str, default="./Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1")
    parser.add_argument("--save_dir", type=str, default="server")
    parser.add_argument("--ulysses_size", type=int, default=8)
    parser.add_argument("--t5_fsdp", action="store_true", default=True)
    parser.add_argument("--dit_fsdp", action="store_true", default=True)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--base_port", type=int, default=8000)


    args = parser.parse_args()
    return args


def _init_logging(rank):
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(level=level, format=f"[RANK {rank}] %(asctime)s %(levelname)s: %(message)s")


def _save_video_to_file(video, save_dir: str, fps: int):
    os.makedirs(save_dir, exist_ok=True)
    file_name = 'video.mp4'
    video_path = os.path.join(save_dir, file_name)
    logging.info(f"Saving generated video to {video_path}")
    save_video(tensor=video[None], save_file=video_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
    return video_path


def _download_image(url: str, save_path: str):
    r = requests.get(url)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img.save(save_path)
    return img


# ====================== 构造所有 pipeline ======================
def build_pipelines(args, rank, local_rank):
    device_id = local_rank
    torch.cuda.set_device(device_id)

    logging.info("Initializing WanT2V pipeline ...")
    t2v_pipe = wan.WanT2V(
        config=WAN_CONFIGS["t2v-A14B"],
        checkpoint_dir=args.t2v_ckpt_dir,
        lora_dir=args.t2v_lora_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logging.info("Initializing WanI2V pipeline ...")
    i2v_pipe = wan.WanI2V(
        config=WAN_CONFIGS["i2v-A14B"],
        checkpoint_dir=args.i2v_ckpt_dir,
        lora_dir=args.i2v_lora_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    return t2v_pipe, i2v_pipe


# ====================== 主程序入口 ======================
def main():
    # 加载运行参数
    args = parse_args()

    # 加载环境变量
    rank = int(os.getenv("RANK", 0))  # 当前进程编号
    world_size = int(os.getenv("WORLD_SIZE", 1))  # 总进程数量
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # 当前进程在当前机器上的编号

    # 初始化 logging 模块
    _init_logging(rank)

    # 初始化分布式进程管理
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size, device_id=local_rank)

    # 构造 pipeline
    t2v_pipe, i2v_pipe = build_pipelines(args, rank, local_rank)

    # 构造 FastAPI 服务
    app = FastAPI()
    app.state.t2v_pipe = t2v_pipe
    app.state.i2v_pipe = i2v_pipe
    app.state.rank = rank
    app.state.args = args
    app.state.world_size = world_size
    app.state.base_port = args.base_port

    # 内部生成逻辑（子进程，rank > 1）
    @app.post("/internal_generate")
    async def internal_generate(req: Request):
        data = await req.json()
        job_id = data["job_id"]
        service_type = data["service_type"]  # "t2v" or "i2v"
        prompt = data["prompt"]
        image_path = None
        
        save_root = os.path.join(args.save_dir, job_id)
        os.makedirs(save_root, exist_ok=True)

        cfg = WAN_CONFIGS[f"{service_type}-A14B"]
        torch.manual_seed(args.base_seed)

        if service_type == "t2v":
            logging.info(f"[RANK {rank}] Generating T2V for prompt: {prompt}")
            video = app.state.t2v_pipe.generate(
                prompt,
                size=SIZE_CONFIGS["1280*720"],
                frame_num=cfg.frame_num,
                shift=cfg.sample_shift,
                sample_solver="euler",
                sampling_steps=args.sample_steps,
                guide_scale=cfg.sample_guide_scale,
                seed=args.base_seed,
            )
        else:
            img_url = data["image_url"]
            image_path = os.path.join(save_root, "input_image.jpg")
            img = _download_image(img_url, image_path)
            logging.info(f"[RANK {rank}] Generating I2V for {img_url}")
            video = app.state.i2v_pipe.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS["1280*720"],
                frame_num=cfg.frame_num,
                shift=cfg.sample_shift,
                sample_solver="euler",
                sampling_steps=args.sample_steps,
                guide_scale=cfg.sample_guide_scale,
                seed=args.base_seed,
            )

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            video_path = _save_video_to_file(video=video, save_dir=save_root, fps=cfg.sample_fps)
            meta = {"prompt": prompt, "image_path": image_path, "video_path": video_path}
            with open(os.path.join(save_root, "meta.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        del video
        return {"status": "ok", "rank": rank}

    # 外部请求入口（主进程，rank = 0）
    if rank == 0:
        @app.post("/generate")
        async def generate_entry(req: Request):
            start_time = time.perf_counter()

            # 解析请求字段
            data = await req.json()
            service_type = data["service_type"]
            prompt = data["prompt"]
            image_url = data.get("image_url", None)

            # 本地存储数据
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            job_id = f"{service_type}_{timestamp}_{uuid.uuid4()}"
            save_root = os.path.join(args.save_dir, job_id)
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            if image_url:
                with open(os.path.join(save_root, "image_url.txt"), "w") as f:
                    f.write(image_url)

            # 构造异步请求并在后台启动，不阻塞主请求
            async def post_rank(r):
                def post_rank_inner():
                    return requests.post(
                    url=f"http://127.0.0.1:{args.base_port + r}/internal_generate",
                    json={
                            "job_id": job_id,
                            "service_type": service_type,
                            "prompt": prompt,
                            "image_url": image_url
                        },
                    timeout=600
                ).json()
                return await asyncio.to_thread(post_rank_inner)

            await asyncio.gather(*[post_rank(r) for r in range(world_size)])

            total_time = time.perf_counter() - start_time

            return JSONResponse(
                content={"job_id": job_id, "total_time_sec": round(total_time, 1)},
                status_code=200
            )

    # 启动 FastAPI 服务
    port = args.base_port + rank
    logging.info(f"[RANK {rank}] Starting server at 127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


if __name__ == "__main__":
    main()
