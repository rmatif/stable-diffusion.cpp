"""Builds the PixelSeal weights that the invisible watermark loads, from Meta's training checkpoint
(https://github.com/facebookresearch/videoseal, MIT): the embedder UNet with its batchnorms folded into the convs, the
message embeddings and the extractor, as fp16 safetensors under the checkpoint's own tensor names.

    pip install torch safetensors numpy
    python script/convert_pixelseal.py pixelseal.safetensors
"""

import hashlib
import sys
import tempfile
import urllib.request

import torch
from safetensors.torch import save_file

URL = "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth"
SHA256 = "0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd"
BATCHNORM_EPS = 1e-5


def download(f):
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL) as response:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    f.flush()
    if digest.hexdigest() != SHA256:
        raise SystemExit(f"{URL}: sha256 {digest.hexdigest()} != {SHA256}")


def convert(state):
    tensors = {k: v for k, v in state.items() if k.startswith(("embedder.unet.", "detector."))}
    # A batchnorm sits right after its conv in every `double_conv`: "….double_conv.1" normalizes "….double_conv.0".
    for name in [k for k in tensors if k.endswith(".running_var")]:
        norm = name[: -len(".running_var")]
        parent, index = norm.rsplit(".", 1)
        conv = f"{parent}.{int(index) - 1}"
        scale = tensors[f"{norm}.weight"] / torch.sqrt(tensors[f"{norm}.running_var"] + BATCHNORM_EPS)
        tensors[f"{conv}.bias"] = tensors[f"{norm}.bias"] - tensors[f"{norm}.running_mean"] * scale
        tensors[f"{conv}.weight"] = tensors[f"{conv}.weight"] * scale[:, None, None, None]
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            del tensors[f"{norm}.{suffix}"]
    return {k: v.half().contiguous() for k, v in tensors.items()}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        download(f)
        state = torch.load(f.name, map_location="cpu", weights_only=True)["model"]
    save_file(convert(state), sys.argv[1])
