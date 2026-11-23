## Marking results with an invisible watermark

`sd-server` can embed an invisible 256-bit watermark in every image it returns, with Meta's
[PixelSeal](https://github.com/facebookresearch/videoseal). The watermark is added to the final pixels of `/generate`
and `/upscale` results, right before they are encoded.

- Build the weights once. The script downloads Meta's checkpoint (sha256-pinned) and writes a 150 MB safetensors file:

```bash
pip install torch safetensors numpy
python script/convert_pixelseal.py pixelseal.safetensors
```

- Start the server with the weights and the message, 64 hex characters, most significant bit first. Both can also come
  from the environment (`SDCPP_WATERMARK_MODEL`, `SDCPP_WATERMARK_MESSAGE`):

```bash
sd-server -m model.safetensors --watermark-model pixelseal.safetensors \
    --watermark-message 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Each image chunk of the response then carries `watermark` (`embedded`, or `present` when the image already carried the
message and was left untouched) and `watermark_ms`.

- An image counts as already watermarked when at least 95% of the message's bits read back from it. That holds for an
  image that is passed through unchanged; it does not after a x4 ESRGAN upscale (84 to 88% read back), so an upscaled
  image is marked again.
- A request fails rather than return an unmarked image.
- `/free` releases the watermark model with everything else on the GPU; the next image reloads it (about 0.2 s).

On an RTX 4090 with `-t 8` the watermark adds 12 ms to a 512x512 image, 15 ms at 1024x1024, 18 ms at 2048x2048 and
about 40 ms at 4096x4096 (2 to 6 ms when the image is already marked), and takes about 0.45 GB of VRAM. The two networks
run in fp32 on the GPU, the per-pixel work on CPU threads.

To read a watermark back, use Meta's detector: `videoseal.load("pixelseal").detect(image)["preds"][0][1:]` gives one
logit per bit, positive for a 1.

Other programs can use `new_watermark_ctx`, `watermark_image` and `free_watermark_ctx` from `stable-diffusion.h`.
