#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"
#include "pixelseal.hpp"
#include "stable-diffusion.h"

namespace {
    // Strength of the watermark signal, as in videoseal's PixelSeal card.
    const float SCALING_W = 0.2f;
    // Share of the message's bits a read must match for an image to count as already watermarked; an unmarked image
    // matches about half of them.
    const float MARKED_BIT_ACCURACY = 0.95f;

    // Runs fn(begin, end) over [0, rows) on up to n_threads threads; a thread is only worth starting for about
    // 100k pixels of work.
    template <typename F>
    void parallel_rows(int n_threads, int rows, int row_width, F&& fn) {
        const int n = (int)std::max<int64_t>(1, std::min<int64_t>({n_threads, rows, (int64_t)rows * row_width / 100000}));
        if (n == 1) {
            fn(0, rows);
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(n - 1);
        for (int i = 1; i < n; i++) {
            workers.emplace_back([&fn, i, n, rows]() { fn(rows * i / n, rows * (i + 1) / n); });
        }
        fn(0, rows / n);
        for (auto& worker : workers) {
            worker.join();
        }
    }

    // One axis of torch's antialiased bilinear resize (F.interpolate(mode="bilinear", antialias=True)): each output
    // sample is a triangle-weighted mean of `taps` consecutive inputs, the triangle widening when the axis shrinks.
    struct ResizeAxis {
        int taps = 0;
        std::vector<int> first;
        std::vector<float> weights;

        ResizeAxis(int in_size, int out_size) {
            const double scale   = (double)in_size / out_size;
            const double support = std::max(scale, 1.0);
            taps                 = 2 * (int)std::ceil(support) + 1;
            first.resize(out_size);
            weights.assign((size_t)out_size * taps, 0.f);
            for (int i = 0; i < out_size; i++) {
                const double center = scale * (i + 0.5);
                const int begin     = std::max(0, (int)(center - support + 0.5));
                const int end       = std::min(in_size, (int)(center + support + 0.5));
                double total        = 0;
                for (int j = begin; j < end; j++) {
                    total += std::max(0.0, 1.0 - std::abs((j - center + 0.5) / support));
                }
                first[i] = begin;
                for (int j = begin; j < end; j++) {
                    weights[(size_t)i * taps + (j - begin)] = (float)(std::max(0.0, 1.0 - std::abs((j - center + 0.5) / support)) / total);
                }
            }
        }
    };

    // out[out_size] = one row resized along x.
    void resize_row(const ResizeAxis& axis, const float* in, int in_size, float* out, int out_size) {
        for (int x = 0; x < out_size; x++) {
            const float* w = &axis.weights[(size_t)x * axis.taps];
            const float* p = in + axis.first[x];
            const int taps = std::min(axis.taps, in_size - axis.first[x]);
            float sum      = 0;
            for (int k = 0; k < taps; k++) {
                sum += w[k] * p[k];
            }
            out[x] = sum;
        }
    }

    // out[width] = row y of `rows` (in_size rows of `width`) resized along y.
    void resize_column(const ResizeAxis& axis, int y, const float* rows, int in_size, float* out, int width) {
        const float* w = &axis.weights[(size_t)y * axis.taps];
        const int taps = std::min(axis.taps, in_size - axis.first[y]);
        std::fill(out, out + width, 0.f);
        for (int k = 0; k < taps; k++) {
            const float* in = rows + (size_t)(axis.first[y] + k) * width;
            const float wk  = w[k];
            for (int x = 0; x < width; x++) {
                out[x] += wk * in[x];
            }
        }
    }

    // x^1.2 for x >= 0, within 2e-4: x^(3/16) from four square roots, refined to x^(1/5) by Newton steps. It keeps
    // the per-pixel loop free of libm calls, which the compiler cannot vectorize.
    inline float pow_1_2(float x) {
        const float v = std::max(x, 1e-12f);
        const float t = std::sqrt(std::sqrt(std::sqrt(std::sqrt(v))));
        float y       = t * t * t;
        for (int i = 0; i < 3; i++) {
            const float y2 = y * y;
            y              = 0.2f * (4.f * y + v / (y2 * y2));
        }
        return x * y;
    }
}  // namespace

struct sd_watermark_ctx_t {
    ggml_backend_t backend = nullptr;
    std::unique_ptr<PixelSeal::Embedder> embedder;
    std::unique_ptr<PixelSeal::Reader> reader;
    int n_threads = 1;

    bool bits[PixelSeal::NBITS];
    ggml_context* io_ctx = nullptr;
    ggml_tensor* keys    = nullptr;  // [SIZE, SIZE, 1, 1]
    ggml_tensor* message = nullptr;  // [1, 1, NBITS, 1]
    ggml_tensor* delta   = nullptr;  // [SIZE, SIZE, 1, 1]
    ggml_tensor* logits  = nullptr;  // [NBITS + 1, 1]

    std::vector<float> luma, rows;

    ~sd_watermark_ctx_t() {
        reader.reset();
        embedder.reset();
        if (io_ctx != nullptr) {
            ggml_free(io_ctx);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    bool init(const std::string& model_path, const std::string& message_hex) {
        if (message_hex.size() != PixelSeal::NBITS / 4 || message_hex.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
            LOG_ERROR("watermark message must be %d hex characters (%d bits)", PixelSeal::NBITS / 4, PixelSeal::NBITS);
            return false;
        }
        for (int i = 0; i < PixelSeal::NBITS; i++) {
            const int digit = std::stoi(message_hex.substr(i / 4, 1), nullptr, 16);
            bits[i]         = (digit >> (3 - i % 4)) & 1;  // most significant bit first
        }

#ifdef SD_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        backend = ggml_backend_vk_init(0);
#endif
#ifdef SD_USE_OPENCL
        backend = ggml_backend_opencl_init();
#endif
#ifdef SD_USE_SYCL
        backend = ggml_backend_sycl_init(0);
#endif
        if (!backend) {
            backend = ggml_backend_cpu_init();
        }

        ModelLoader model_loader;
        if (!model_loader.init_from_file(model_path)) {
            LOG_ERROR("init watermark model loader from file failed: '%s'", model_path.c_str());
            return false;
        }
        embedder = std::make_unique<PixelSeal::Embedder>(backend);
        reader   = std::make_unique<PixelSeal::Reader>(backend);
        if (!embedder->alloc_params_buffer() || !reader->alloc_params_buffer()) {
            return false;
        }
        std::map<std::string, ggml_tensor*> tensors;
        embedder->unet.get_param_tensors(tensors, "embedder.unet");
        reader->extractor.get_param_tensors(tensors, "detector");
        if (!model_loader.load_tensors(tensors, {}, n_threads)) {
            LOG_ERROR("load watermark tensors from '%s' failed", model_path.c_str());
            return false;
        }

        ggml_init_params params;
        params.mem_size   = 4 * ggml_tensor_overhead() + sizeof(float) * (2 * PixelSeal::SIZE * PixelSeal::SIZE + 2 * PixelSeal::NBITS + 1) + 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        io_ctx            = ggml_init(params);
        keys              = ggml_new_tensor_4d(io_ctx, GGML_TYPE_F32, PixelSeal::SIZE, PixelSeal::SIZE, 1, 1);
        delta             = ggml_new_tensor_4d(io_ctx, GGML_TYPE_F32, PixelSeal::SIZE, PixelSeal::SIZE, 1, 1);
        message           = ggml_new_tensor_4d(io_ctx, GGML_TYPE_F32, 1, 1, PixelSeal::NBITS, 1);
        logits            = ggml_new_tensor_2d(io_ctx, GGML_TYPE_F32, PixelSeal::NBITS + 1, 1);

        // The UNet is conditioned on the sum of one embedding per bit: row 2k for a 0 at bit k, row 2k + 1 for a 1.
        ggml_tensor* embeddings = embedder->unet.message_embeddings();
        std::vector<float> table(ggml_nelements(embeddings));
        ggml_backend_tensor_get(embeddings, table.data(), 0, ggml_nbytes(embeddings));
        float* sum = (float*)message->data;
        std::fill(sum, sum + PixelSeal::NBITS, 0.f);
        for (int k = 0; k < PixelSeal::NBITS; k++) {
            const float* row = &table[(size_t)(2 * k + bits[k]) * PixelSeal::NBITS];
            for (int d = 0; d < PixelSeal::NBITS; d++) {
                sum[d] += row[d];
            }
        }
        return true;
    }

    bool apply(sd_image_t image, bool* already_marked) {
        const int width = (int)image.width, height = (int)image.height, channels = (int)image.channel;
        if (image.data == nullptr || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
            LOG_ERROR("watermark: unsupported image (%dx%d, %d channels)", width, height, channels);
            return false;
        }
        const int colors = channels == 1 ? 1 : 3;
        const size_t pixels = (size_t)width * height;

        // The watermark is computed from, and added to, luma (0..255 here); alpha is left alone. The networks see
        // luma resized to SIZE x SIZE: along x as each row is produced, then along y.
        const int size = PixelSeal::SIZE;
        luma.resize(pixels);
        rows.resize((size_t)height * size);
        {
            const ResizeAxis x_axis(width, size), y_axis(height, size);
            parallel_rows(n_threads, height, width, [&](int begin, int end) {
                for (int y = begin; y < end; y++) {
                    const uint8_t* in = image.data + (size_t)y * width * channels;
                    float* out        = luma.data() + (size_t)y * width;
                    if (colors == 1) {
                        for (int x = 0; x < width; x++) {
                            out[x] = in[x];
                        }
                    } else {
                        for (int x = 0; x < width; x++) {
                            out[x] = 0.299f * in[x * channels] + 0.587f * in[x * channels + 1] + 0.114f * in[x * channels + 2];
                        }
                    }
                    resize_row(x_axis, out, width, rows.data() + (size_t)y * size, size);
                }
            });
            float* key = (float*)keys->data;
            for (int y = 0; y < size; y++) {
                resize_column(y_axis, y, rows.data(), height, key + (size_t)y * size, size);
            }
            for (int i = 0; i < size * size; i++) {
                key[i] = key[i] * (2.f / 255.f) - 1.f;
            }
        }

        if (!reader->compute(n_threads, keys, &logits)) {
            return false;
        }
        const float* read = (const float*)logits->data + 1;
        int matches       = 0;
        for (int k = 0; k < PixelSeal::NBITS; k++) {
            matches += (read[k] > 0) == bits[k];
        }
        const bool marked = matches >= MARKED_BIT_ACCURACY * PixelSeal::NBITS;
        if (already_marked != nullptr) {
            *already_marked = marked;
        }
        if (marked) {
            return true;
        }

        if (!embedder->compute(n_threads, keys, message, &delta)) {
            return false;
        }
        // The signal is resized to the image the same way: along x here, along y as each image row is written.
        const ResizeAxis x_axis(size, width), y_axis(size, height);
        rows.resize((size_t)size * width);
        parallel_rows(n_threads, size, width, [&](int begin, int end) {
            for (int y = begin; y < end; y++) {
                resize_row(x_axis, (const float*)delta->data + (size_t)y * size, size, rows.data() + (size_t)y * width, width);
            }
        });

        // videoseal's JND attenuation (jnd_1_1): the signal is scaled per pixel by how much change the eye tolerates
        // there, from the local background luminance and the local contrast, both taken from luma with zero padding.
        parallel_rows(n_threads, height, width, [&](int begin, int end) {
            const int padded = width + 4;
            std::vector<float> buffer((size_t)padded * 6, 0.f);
            float* sum5   = buffer.data() + 2;
            float* sum3   = sum5 + padded;
            float* blur   = sum3 + padded;
            float* diff   = blur + padded;
            float* signal = diff + padded;
            float* change = signal + padded;
            for (int y = begin; y < end; y++) {
                std::fill(buffer.begin(), buffer.begin() + (size_t)padded * 4, 0.f);
                for (int dy = -2; dy <= 2; dy++) {
                    if (y + dy < 0 || y + dy >= height) {
                        continue;
                    }
                    const float* row = luma.data() + (size_t)(y + dy) * width;
                    for (int x = 0; x < width; x++) {
                        sum5[x] += row[x];
                    }
                    if (dy >= -1 && dy <= 1) {
                        const float gain = dy == 0 ? 2.f : 1.f;
                        const float sign = (float)-dy;
                        for (int x = 0; x < width; x++) {
                            sum3[x] += row[x];
                            blur[x] += gain * row[x];
                            diff[x] += sign * row[x];
                        }
                    }
                }
                resize_column(y_axis, y, rows.data(), size, signal, width);

                const float* center = luma.data() + (size_t)y * width;
                for (int x = 0; x < width; x++) {
                    const float background = (sum5[x - 2] + sum5[x - 1] + sum5[x] + sum5[x + 1] + sum5[x + 2] + sum3[x - 1] + sum3[x] + sum3[x + 1] - 2.f * center[x]) / 32.f;
                    const float dark       = 17.f * (1.f - std::sqrt(background / 127.f + 1e-5f));
                    const float bright     = 3.f / 128.f * (background - 127.f) + 3.f;
                    const float luminance  = background <= 127.f ? dark : bright;
                    const float gx         = blur[x + 1] - blur[x - 1];
                    const float gy         = diff[x - 1] + 2.f * diff[x] + diff[x + 1];
                    const float gradient   = gx * gx + gy * gy;
                    const float contrast   = 0.117f * 16.f * pow_1_2(gradient) / (gradient + 26.f * 26.f);
                    const float visibility = std::max(luminance + contrast - 0.3f * std::min(luminance, contrast), 0.f);
                    change[x]              = SCALING_W * visibility * signal[x];
                }

                uint8_t* out = image.data + (size_t)y * width * channels;
                for (int x = 0; x < width; x++) {
                    const int rounded = (int)(change[x] + 256.5f) - 256;  // floor(change + 0.5); change is far above -256
                    for (int c = 0; c < colors; c++) {
                        const int value       = out[x * channels + c] + rounded;
                        out[x * channels + c] = (uint8_t)std::min(std::max(value, 0), 255);
                    }
                }
            }
        });
        return true;
    }
};

sd_watermark_ctx_t* new_watermark_ctx(const char* model_path, const char* message_hex, int n_threads) {
    sd_watermark_ctx_t* ctx = new sd_watermark_ctx_t();
    ctx->n_threads          = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    if (!ctx->init(model_path, message_hex)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

bool watermark_image(sd_watermark_ctx_t* ctx, sd_image_t image, bool* already_marked) {
    return ctx->apply(image, already_marked);
}

void free_watermark_ctx(sd_watermark_ctx_t* ctx) {
    delete ctx;
}
