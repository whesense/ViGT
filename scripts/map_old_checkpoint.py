import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:
    safetensors_load_file = None
    safetensors_save_file = None


def extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in raw_obj and isinstance(raw_obj[key], dict):
                return raw_obj[key]
    if isinstance(raw_obj, dict):
        return raw_obj
    raise ValueError("Unsupported checkpoint format: expected a dict-like object.")


def strip_prefix(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def remap_key(old_key: str) -> Tuple[str | None, str]:
    key = old_key

    # Common wrappers from DDP/Lightning-like training loops.
    key = strip_prefix(key, "module.")
    key = strip_prefix(key, "model.")

    # Drop legacy heads/branches not used in current inference model.
    if key.startswith("camera_decoder.") or key.startswith("cam_embedder."):
        return None, "dropped legacy camera branch"
    if key.startswith("roi."):
        return None, "dropped roi module tensors"

    # Current HF wrapper stores the core model under `model.*`.
    if key.startswith("encoder.") or key.startswith("decoder."):
        new_key = f"model.{key}"
        new_key = _apply_key_aliases(new_key)
        return new_key, "prefixed to current wrapper"

    # Sometimes old checkpoints already had ViGT-like root naming.
    if key.startswith("vigt.encoder.") or key.startswith("vigt.decoder."):
        new_key = f"model.{key[len('vigt.'):]}"
        new_key = _apply_key_aliases(new_key)
        return new_key, "mapped vigt.* to model.*"

    # Keep unknown keys out to avoid polluting output.
    return None, "unrecognized root (dropped)"


def _apply_key_aliases(key: str) -> str:
    """Alias legacy layer names to current refactored names."""
    replacements = (
        (".resConfUnit1.", ".res1."),
        (".resConfUnit2.", ".res2."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def remap_state_dict(old_sd: Dict[str, torch.Tensor]):
    new_sd: Dict[str, torch.Tensor] = {}
    report = {
        "mapped": 0,
        "dropped": 0,
        "drop_reasons": {},
    }

    for old_key, value in old_sd.items():
        new_key, reason = remap_key(old_key)
        if new_key is None:
            report["dropped"] += 1
            report["drop_reasons"][reason] = report["drop_reasons"].get(reason, 0) + 1
            continue
        new_sd[new_key] = value
        report["mapped"] += 1

    return new_sd, report


def main():
    parser = argparse.ArgumentParser(description="Map legacy UnO/LaRa checkpoint keys to current ViGT weights.")
    parser.add_argument("--input", type=str, required=True, help="Path to old checkpoint (.pt/.pth/.ckpt).")
    parser.add_argument("--output", type=str, required=True, help="Path to output mapped state_dict (.bin/.pt).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError("safetensors is not installed. Run `pip install safetensors`.")
        raw = safetensors_load_file(str(input_path))
    else:
        try:
            raw = torch.load(input_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Backward compatibility with older torch versions.
            raw = torch.load(input_path, map_location="cpu")
    old_sd = extract_state_dict(raw)
    new_sd, report = remap_state_dict(old_sd)

    if output_path.suffix == ".safetensors":
        if safetensors_save_file is None:
            raise ImportError("safetensors is not installed. Run `pip install safetensors`.")
        safetensors_save_file(new_sd, str(output_path))
    else:
        torch.save(new_sd, output_path)

    print(f"Saved mapped checkpoint: {output_path}")
    print(f"Mapped keys: {report['mapped']}")
    print(f"Dropped keys: {report['dropped']}")
    if report["drop_reasons"]:
        print("Drop reasons:")
        for reason, count in sorted(report["drop_reasons"].items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
