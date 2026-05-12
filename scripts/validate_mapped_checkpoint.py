import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None


def load_state_dict(weights_path: Path):
    if weights_path.suffix == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError("safetensors is not installed. Run `pip install safetensors`.")
        return safetensors_load_file(str(weights_path))

    try:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(weights_path, map_location="cpu")

    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if isinstance(raw, dict):
        return raw
    raise ValueError("Unsupported checkpoint format.")


def main():
    parser = argparse.ArgumentParser(description="Validate mapped checkpoint against current ViGT model.")
    parser.add_argument("--weights", type=str, required=True, help="Path to mapped weights (.safetensors/.pt/.bin).")
    parser.add_argument("--config", type=str, default=None, help="Optional config.json path for ViGTHFConfig.")
    parser.add_argument("--strict", action="store_true", help="Fail if any mismatch/unexpected/missing key exists.")
    parser.add_argument("--max-print", type=int, default=20, help="Max keys to print per category.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    from vigt_hf import ViGTHFConfig, ViGTForInference  # noqa: WPS433

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = ViGTHFConfig(**json.load(f))
    else:
        cfg = ViGTHFConfig()

    model = ViGTForInference(cfg)
    model_sd = model.state_dict()
    ckpt_sd = load_state_dict(Path(args.weights))

    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())

    unexpected = sorted(ckpt_keys - model_keys)
    missing = sorted(model_keys - ckpt_keys)
    common = model_keys & ckpt_keys

    shape_mismatch = []
    matched_keys = []
    for key in sorted(common):
        if tuple(model_sd[key].shape) != tuple(ckpt_sd[key].shape):
            shape_mismatch.append((key, tuple(ckpt_sd[key].shape), tuple(model_sd[key].shape)))
        else:
            matched_keys.append(key)

    filtered_sd = {k: ckpt_sd[k] for k in matched_keys}
    model.load_state_dict(filtered_sd, strict=False)

    print(f"Total model keys: {len(model_keys)}")
    print(f"Total checkpoint keys: {len(ckpt_keys)}")
    print(f"Matched keys (shape ok): {len(matched_keys)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")

    max_print = max(0, args.max_print)

    if missing:
        print("\nMissing keys:")
        for key in missing[:max_print]:
            print(f"  - {key}")
    if unexpected:
        print("\nUnexpected keys:")
        for key in unexpected[:max_print]:
            print(f"  - {key}")
    if shape_mismatch:
        print("\nShape mismatches:")
        for key, ckpt_shape, model_shape in shape_mismatch[:max_print]:
            print(f"  - {key}: ckpt={ckpt_shape} model={model_shape}")

    has_issues = bool(missing or unexpected or shape_mismatch)
    if args.strict and has_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
