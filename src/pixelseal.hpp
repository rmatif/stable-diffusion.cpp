#ifndef __PIXELSEAL_HPP__
#define __PIXELSEAL_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*
 * PixelSeal, Meta's invisible image watermark (https://github.com/facebookresearch/videoseal, MIT).
 * A UNet turns a 256x256 luma image and a 256-bit message into a watermark signal, and a ConvNeXt-V2
 * extractor reads the bits back from an image. Blocks and tensor names follow Meta's checkpoint;
 * script/convert_pixelseal.py folds the UNet's batchnorms into its convs.
 */
namespace PixelSeal {
    constexpr int NBITS = 256;
    constexpr int SIZE  = 256;

    // A convolution with fp32 weights: Conv2d's fp16 tensor-core kernel accumulates in half precision, which costs
    // these small networks several percent of accuracy per layer.
    class Conv : public Conv2d {
    protected:
        // im2col + GEMM is the faster fp32 path as long as its column matrix stays small.
        static constexpr int64_t MAX_COLUMN_BYTES = 32 << 20;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kernel_size.second, kernel_size.first, in_channels, out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        using Conv2d::Conv2d;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            const int64_t out_w        = (x->ne[0] + 2 * padding.second - kernel_size.second) / stride.second + 1;
            const int64_t out_h        = (x->ne[1] + 2 * padding.first - kernel_size.first) / stride.first + 1;
            const int64_t column_bytes = out_w * out_h * in_channels * kernel_size.first * kernel_size.second * (int64_t)sizeof(float);
            return ggml_ext_conv_2d(ctx->ggml_ctx, x, params["weight"], bias ? params["bias"] : nullptr,
                                    stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first,
                                    column_bytes > MAX_COLUMN_BYTES);
        }
    };

    // Normalizes each pixel over its channels (videoseal's "channels_first" LayerNorm). x: [W, H, C, N]
    class ChannelNorm : public UnaryBlock {
    protected:
        int64_t channels;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        }

    public:
        ChannelNorm(int64_t channels)
            : channels(channels) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C, W, H, N]
            x = ggml_ext_layer_norm(ctx->ggml_ctx, x, params["weight"], params["bias"], 1e-6f);
            return ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
        }
    };

    // Bilinear upsampling, then a 3x3 conv over a reflection-padded input, a norm and an activation.
    class Upsample : public UnaryBlock {
    protected:
        int factor;
        bool gelu;

        // ReflectionPad2d(1); ggml only pads the first dimension by reflection.
        static ggml_tensor* reflection_pad(ggml_context* ctx, ggml_tensor* x) {
            x = ggml_pad_reflect_1d(ctx, x, 1, 1);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            x = ggml_pad_reflect_1d(ctx, x, 1, 1);
            return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        }

    public:
        Upsample(int64_t in_channels, int64_t out_channels, int factor, bool gelu)
            : factor(factor), gelu(gelu) {
            blocks["upsample_block.2"] = std::shared_ptr<GGMLBlock>(new Conv(in_channels, out_channels, {3, 3}, {1, 1}, {0, 0}, {1, 1}, false));
            blocks["upsample_block.3"] = std::shared_ptr<GGMLBlock>(new ChannelNorm(out_channels));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv = std::dynamic_pointer_cast<Conv>(blocks["upsample_block.2"]);
            auto norm = std::dynamic_pointer_cast<ChannelNorm>(blocks["upsample_block.3"]);

            if (factor > 1) {
                x = ggml_interpolate(ctx->ggml_ctx, x, x->ne[0] * factor, x->ne[1] * factor, x->ne[2], x->ne[3], GGML_SCALE_MODE_BILINEAR);
            }
            x = conv->forward(ctx, reflection_pad(ctx->ggml_ctx, x));
            x = norm->forward(ctx, x);
            return gelu ? ggml_gelu_erf(ctx->ggml_ctx, x) : ggml_relu_inplace(ctx->ggml_ctx, x);
        }
    };

    class ResnetBlock : public UnaryBlock {
    public:
        ResnetBlock(int64_t in_channels, int64_t out_channels) {
            blocks["double_conv.0"] = std::shared_ptr<GGMLBlock>(new Conv(in_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
            blocks["double_conv.3"] = std::shared_ptr<GGMLBlock>(new Conv(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
            blocks["res_conv"]      = std::shared_ptr<GGMLBlock>(new Conv(in_channels, out_channels, {1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv1    = std::dynamic_pointer_cast<Conv>(blocks["double_conv.0"]);
            auto conv2    = std::dynamic_pointer_cast<Conv>(blocks["double_conv.3"]);
            auto res_conv = std::dynamic_pointer_cast<Conv>(blocks["res_conv"]);

            auto h = ggml_relu_inplace(ctx->ggml_ctx, conv1->forward(ctx, x));
            h      = ggml_relu_inplace(ctx->ggml_ctx, conv2->forward(ctx, h));
            return ggml_add(ctx->ggml_ctx, h, res_conv->forward(ctx, x));
        }
    };

    // videoseal's `unet_base_yuv_quant` embedder.
    class UNet : public GGMLBlock {
    protected:
        const std::vector<int64_t> channels = {32, 64, 128, 256};
        const int num_bottleneck_blocks     = 8;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            params["msg_processor.msg_embeddings.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NBITS, 2 * NBITS);
        }

    public:
        UNet() {
            const int64_t bottleneck_channels = channels.back() + NBITS;
            blocks["inc"]                     = std::shared_ptr<GGMLBlock>(new ResnetBlock(1, channels[0]));
            for (size_t i = 0; i + 1 < channels.size(); i++) {
                std::string name       = "downs." + std::to_string(i);
                blocks[name + ".down"] = std::shared_ptr<GGMLBlock>(new Conv(channels[i], channels[i + 1], {3, 3}, {2, 2}, {1, 1}));
                blocks[name + ".conv"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(channels[i + 1], channels[i + 1]));
            }
            for (int i = 0; i < num_bottleneck_blocks; i++) {
                blocks["bottleneck.model." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResnetBlock(bottleneck_channels, bottleneck_channels));
            }
            int64_t in_channels = bottleneck_channels;
            for (size_t i = 0; i + 1 < channels.size(); i++) {
                std::string name         = "ups." + std::to_string(i);
                int64_t out_channels     = channels[channels.size() - 2 - i];
                blocks[name + ".up"]     = std::shared_ptr<GGMLBlock>(new Upsample(2 * in_channels, out_channels, 2, false));
                blocks[name + ".conv"]   = std::shared_ptr<GGMLBlock>(new ResnetBlock(out_channels, out_channels));
                in_channels              = out_channels;
            }
            blocks["outc"] = std::shared_ptr<GGMLBlock>(new Conv(channels[0], 1, {1, 1}));
        }

        ggml_tensor* message_embeddings() {
            return params["msg_processor.msg_embeddings.weight"];
        }

        // x: [SIZE, SIZE, 1, N] luma in [-1, 1]; message: [1, 1, NBITS, 1], the summed embeddings of the message's bits
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* message) {
            auto forward = [&](const std::string& name, ggml_tensor* x) {
                return std::dynamic_pointer_cast<UnaryBlock>(blocks[name])->forward(ctx, x);
            };

            std::vector<ggml_tensor*> skips;
            x = forward("inc", x);
            for (size_t i = 0; i + 1 < channels.size(); i++) {
                std::string name = "downs." + std::to_string(i);
                x                = forward(name + ".conv", forward(name + ".down", x));
                skips.push_back(x);
            }

            auto like = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], NBITS, x->ne[3]);
            x         = ggml_concat(ctx->ggml_ctx, x, ggml_repeat(ctx->ggml_ctx, message, like), 2);
            skips.back() = x;

            for (int i = 0; i < num_bottleneck_blocks; i++) {
                x = forward("bottleneck.model." + std::to_string(i), x);
            }
            for (size_t i = 0; i + 1 < channels.size(); i++) {
                std::string name = "ups." + std::to_string(i);
                auto skip        = ggml_scale(ctx->ggml_ctx, skips[skips.size() - 1 - i], 0.70710678f);
                x                = ggml_concat(ctx->ggml_ctx, x, skip, 2);
                x                = forward(name + ".conv", forward(name + ".up", x));
            }
            return ggml_tanh(ctx->ggml_ctx, forward("outc", x));
        }
    };

    class ConvNeXtBlock : public UnaryBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            params["dwconv.weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 7, 7, 1, dim);
            params["dwconv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["grn.gamma"]     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4 * dim, 1, 1, 1);
            params["grn.beta"]      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4 * dim, 1, 1, 1);
        }

    public:
        ConvNeXtBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, 1e-6f));
            blocks["pwconv1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 4 * dim));
            blocks["pwconv2"] = std::shared_ptr<GGMLBlock>(new Linear(4 * dim, dim));
        }

        // x: [W, H, dim, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto norm    = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto pwconv1 = std::dynamic_pointer_cast<Linear>(blocks["pwconv1"]);
            auto pwconv2 = std::dynamic_pointer_cast<Linear>(blocks["pwconv2"]);
            auto c       = ctx->ggml_ctx;
            int64_t w = x->ne[0], h = x->ne[1];

            auto y = ggml_conv_2d_dw_direct(c, params["dwconv.weight"], x, 1, 1, 3, 3, 1, 1);
            y      = ggml_add(c, y, ggml_reshape_4d(c, params["dwconv.bias"], 1, 1, dim, 1));
            y      = ggml_cont(c, ggml_ext_torch_permute(c, y, 2, 0, 1, 3));
            y      = ggml_reshape_2d(c, y, dim, w * h);  // [dim, pixels]: the rest of the block works per pixel
            y      = norm->forward(ctx, y);
            y      = ggml_gelu_erf(c, pwconv1->forward(ctx, y));

            // Global response normalization: each channel is rescaled by its L2 norm over the image, relative to the
            // mean norm of all channels. The reference adds 1e-6 to that mean, far below what it can be.
            auto norms = ggml_sqrt(c, ggml_sum_rows(c, ggml_cont(c, ggml_transpose(c, ggml_sqr(c, y)))));  // [1, 4 * dim]
            norms      = ggml_div(c, norms, ggml_mean(c, ggml_reshape_1d(c, norms, 4 * dim)));
            auto gain  = ggml_mul(c, ggml_reshape_2d(c, norms, 4 * dim, 1), ggml_reshape_2d(c, params["grn.gamma"], 4 * dim, 1));
            y          = ggml_mul(c, y, ggml_scale_bias(c, gain, 1.f, 1.f));
            y          = ggml_add(c, y, ggml_reshape_2d(c, params["grn.beta"], 4 * dim, 1));

            y = pwconv2->forward(ctx, y);
            y = ggml_reshape_4d(c, y, dim, w, h, 1);
            y = ggml_cont(c, ggml_ext_torch_permute(c, y, 1, 2, 0, 3));
            return ggml_add(c, x, y);
        }
    };

    // videoseal's `convnext_tiny` extractor.
    class Extractor : public UnaryBlock {
    protected:
        const std::vector<int> depths   = {3, 3, 9, 3};
        const std::vector<int64_t> dims = {96, 192, 384, 768};

    public:
        Extractor() {
            blocks["convnext.downsample_layers.0.0"] = std::shared_ptr<GGMLBlock>(new Conv(3, dims[0], {4, 4}, {4, 4}));
            blocks["convnext.downsample_layers.0.1"] = std::shared_ptr<GGMLBlock>(new ChannelNorm(dims[0]));
            for (size_t i = 1; i < dims.size(); i++) {
                std::string name     = "convnext.downsample_layers." + std::to_string(i);
                blocks[name + ".0"]  = std::shared_ptr<GGMLBlock>(new ChannelNorm(dims[i - 1]));
                blocks[name + ".1"]  = std::shared_ptr<GGMLBlock>(new Conv(dims[i - 1], dims[i], {2, 2}, {2, 2}));
            }
            for (size_t i = 0; i < dims.size(); i++) {
                for (int j = 0; j < depths[i]; j++) {
                    blocks["convnext.stages." + std::to_string(i) + "." + std::to_string(j)] = std::shared_ptr<GGMLBlock>(new ConvNeXtBlock(dims[i]));
                }
            }
            blocks["pixel_decoder.output_upscaling.0"] = std::shared_ptr<GGMLBlock>(new Upsample(dims.back(), dims.back(), 1, true));
            blocks["pixel_decoder.linear"]             = std::shared_ptr<GGMLBlock>(new Linear(dims.back(), NBITS + 1));
        }

        // x: [SIZE, SIZE, 3, 1] in [-1, 1]. Returns [NBITS + 1, 1] logits; the first one is not a message bit.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto forward = [&](const std::string& name, ggml_tensor* x) {
                return std::dynamic_pointer_cast<UnaryBlock>(blocks[name])->forward(ctx, x);
            };

            for (size_t i = 0; i < dims.size(); i++) {
                std::string name = "convnext.downsample_layers." + std::to_string(i);
                x                = forward(name + ".1", forward(name + ".0", x));
                for (int j = 0; j < depths[i]; j++) {
                    x = forward("convnext.stages." + std::to_string(i) + "." + std::to_string(j), x);
                }
            }
            x = forward("pixel_decoder.output_upscaling.0", x);
            x = ggml_mean(ctx->ggml_ctx, ggml_reshape_2d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1], x->ne[2]));  // [1, channels]
            x = ggml_reshape_2d(ctx->ggml_ctx, x, dims.back(), 1);
            return forward("pixel_decoder.linear", x);
        }
    };

    // Each network gets its own runner: a runner keeps one compute buffer, sized for the graph it runs.
    struct Embedder : public GGMLRunner {
        UNet unet;

        Embedder(ggml_backend_t backend)
            : GGMLRunner(backend, false) {
            unet.init(params_ctx, {}, "embedder.unet");
        }

        std::string get_desc() override {
            return "pixelseal embedder";
        }

        bool compute(int n_threads, ggml_tensor* keys, ggml_tensor* message, ggml_tensor** delta) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(4096);
                auto runner_ctx = get_context();
                ggml_build_forward_expand(gf, unet.forward(&runner_ctx, to_backend(keys), to_backend(message)));
                return gf;
            };
            return GGMLRunner::compute(get_graph, n_threads, false, delta);
        }
    };

    struct Reader : public GGMLRunner {
        Extractor extractor;

        Reader(ggml_backend_t backend)
            : GGMLRunner(backend, false) {
            extractor.init(params_ctx, {}, "detector");
        }

        std::string get_desc() override {
            return "pixelseal extractor";
        }

        bool compute(int n_threads, ggml_tensor* keys, ggml_tensor** logits) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(4096);
                auto runner_ctx = get_context();
                // The watermark lives in luma, so a gray image carries everything the extractor reads.
                auto x    = to_backend(keys);
                auto like = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 3, x->ne[3]);
                ggml_build_forward_expand(gf, extractor.forward(&runner_ctx, ggml_repeat(compute_ctx, x, like)));
                return gf;
            };
            return GGMLRunner::compute(get_graph, n_threads, false, logits);
        }
    };
}  // namespace PixelSeal

#endif  // __PIXELSEAL_HPP__
