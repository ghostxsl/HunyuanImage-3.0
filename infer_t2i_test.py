import os
import argparse
import json
import datetime
import uuid
from tqdm import tqdm
from pathlib import Path
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument(
        "--input_file", default="input.json", type=str)
    parser.add_argument(
        "--output_file", default="output.json", type=str)
    parser.add_argument(
        "--save_dir", default="output", type=str)
    parser.add_argument(
        "--model-id", default="./HunyuanImage-3", type=str, help="Path to the model")

    parser.add_argument("--attn-impl", type=str, default="flash_attention_2",
                        choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager",
                        choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="1280x720",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-system-prompt", type=str, default=None,
                        choices=[None, "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using the custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Default to load from the model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt "
                                                          "is 'custom'.")
    parser.add_argument("--bot-task", type=str, default="image",
                        choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
                             "Default to load from the model generation config.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def json_save(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f)


def json_load(file):
    with open(file, 'r') as f:
        out = json.load(f)
    return out


def generate_timestamp_uuid():
    # 获取当前时间的纳秒级时间戳
    nano_timestamp = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).timestamp()
    # 生成一个 UUID
    unique_id = uuid.uuid4()
    # 返回时间戳和 UUID 的组合
    return nano_timestamp, unique_id


def _gen_name(folder_name):
    nano_ts, uid = generate_timestamp_uuid()
    return (
        folder_name + "/" + "_".join(str(nano_ts).split(".")) + "_" + str(uid) if folder_name and folder_name != "" else str(nano_ts) + "_" + str(uid)
    )


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    # Load dataset
    data_list = json_load(args.input_file)

    out = []
    for line in tqdm(data_list):
        try:
            prompt = line["prompt"]
            image = model.generate_image(
                prompt=prompt,
                seed=args.seed,
                image_size=args.image_size,
                use_system_prompt=args.use_system_prompt,
                system_prompt=args.system_prompt,
                bot_task=args.bot_task,
                diff_infer_steps=args.diff_infer_steps,
                verbose=args.verbose,
                stream=True,
            )

            name = _gen_name("") + "_gen.jpg"
            image.save(os.path.join(args.save_dir, name))
            line["generate_image"] = os.path.join(args.save_dir, name)

            out.append(line)
            if len(out) % 5 == 0:
                json_save(out, args.output_file)
        except Exception as e:
            print(e)

    # Final save
    json_save(out, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done!")
