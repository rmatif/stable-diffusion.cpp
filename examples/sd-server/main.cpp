#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <iostream>
#include <memory>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "stable-diffusion.h"

#include "httplib.h"
#include "json.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string to_lower_copy(const std::string& input) {
    std::string lowered = input;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lowered;
}

std::string trim_copy(const std::string& input) {
    size_t first = input.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    size_t last = input.find_last_not_of(" \t\n\r");
    return input.substr(first, last - first + 1);
}

std::vector<std::string> split_and_trim(const std::string& input, char delimiter) {
    std::vector<std::string> parts;
    if (input.empty()) {
        return parts;
    }
    size_t start_pos = 0;
    while (start_pos <= input.size()) {
        size_t end = input.find(delimiter, start_pos);
        std::string part;
        if (end == std::string::npos) {
            part = input.substr(start_pos);
            parts.push_back(trim_copy(part));
            break;
        }
        part = input.substr(start_pos, end - start_pos);
        parts.push_back(trim_copy(part));
        start_pos = end + 1;
        if (start_pos == input.size()) {
            parts.emplace_back();
            break;
        }
    }
    return parts;
}

int64_t generate_random_seed() {
    std::random_device rd;
    uint64_t high = static_cast<uint64_t>(rd()) << 32u;
    uint64_t low = static_cast<uint64_t>(rd());
    uint64_t combined = (high | low) & std::numeric_limits<int64_t>::max();
    if (combined == 0) {
        combined = 1;
    }
    return static_cast<int64_t>(combined);
}

const char* log_level_tag(sd_log_level_t level) {
    switch (level) {
        case SD_LOG_DEBUG:
            return "DEBUG";
        case SD_LOG_INFO:
            return "INFO";
        case SD_LOG_WARN:
            return "WARN";
        case SD_LOG_ERROR:
            return "ERROR";
        default:
            return "INFO";
    }
}

std::string log_level_to_string(sd_log_level_t level) {
    switch (level) {
        case SD_LOG_DEBUG:
            return "debug";
        case SD_LOG_INFO:
            return "info";
        case SD_LOG_WARN:
            return "warn";
        case SD_LOG_ERROR:
            return "error";
        default:
            return "info";
    }
}

std::string base64_encode(const unsigned char* data, size_t length) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if (length == 0) {
        return {};
    }

    std::string encoded;
    encoded.reserve(((length + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < length) {
        unsigned char b0 = data[i];
        unsigned char b1 = data[i + 1];
        unsigned char b2 = data[i + 2];

        encoded.push_back(table[(b0 >> 2) & 0x3F]);
        encoded.push_back(table[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
        encoded.push_back(table[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)]);
        encoded.push_back(table[b2 & 0x3F]);

        i += 3;
    }

    if (i < length) {
        unsigned char b0 = data[i];
        encoded.push_back(table[(b0 >> 2) & 0x3F]);
        if (i + 1 < length) {
            unsigned char b1 = data[i + 1];
            encoded.push_back(table[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
            encoded.push_back(table[(b1 & 0x0F) << 2]);
            encoded.push_back('=');
        } else {
            encoded.push_back(table[(b0 & 0x03) << 4]);
            encoded.push_back('=');
            encoded.push_back('=');
        }
    }

    return encoded;
}

struct Utf8SplitResult {
    std::string valid;
    std::string remainder;
};

Utf8SplitResult extract_complete_utf8(const std::string& input) {
    Utf8SplitResult result;
    result.valid.reserve(input.size());

    const std::size_t size = input.size();
    std::size_t i = 0;
    while (i < size) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        if (c < 0x80) {
            result.valid.push_back(static_cast<char>(c));
            ++i;
            continue;
        }

        std::size_t expected = 0;
        if (c >= 0xC2 && c <= 0xDF) {
            expected = 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            expected = 3;
        } else if (c >= 0xF0 && c <= 0xF4) {
            expected = 4;
        } else {
            result.valid.push_back('?');
            ++i;
            continue;
        }

        if (i + expected > size) {
            result.remainder = input.substr(i);
            return result;
        }

        bool valid_sequence = true;
        for (std::size_t j = 1; j < expected; ++j) {
            unsigned char continuation = static_cast<unsigned char>(input[i + j]);
            if ((continuation & 0xC0) != 0x80) {
                valid_sequence = false;
                break;
            }
        }
        if (!valid_sequence) {
            result.valid.push_back('?');
            ++i;
            continue;
        }

        if (expected == 3) {
            unsigned char b1 = static_cast<unsigned char>(input[i + 1]);
            if (c == 0xE0 && b1 < 0xA0) {
                result.valid.push_back('?');
                ++i;
                continue;
            }
            if (c == 0xED && b1 >= 0xA0) {
                result.valid.push_back('?');
                ++i;
                continue;
            }
        } else if (expected == 4) {
            unsigned char b1 = static_cast<unsigned char>(input[i + 1]);
            if (c == 0xF0 && b1 < 0x90) {
                result.valid.push_back('?');
                ++i;
                continue;
            }
            if (c == 0xF4 && b1 >= 0x90) {
                result.valid.push_back('?');
                ++i;
                continue;
            }
        }

        result.valid.append(input, i, expected);
        i += expected;
    }

    return result;
}

struct OwnedImage {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channel = 0;
    std::vector<uint8_t> data;

    bool valid() const {
        return width > 0 && height > 0 && !data.empty() && channel > 0;
    }

    sd_image_t as_sd_image() const {
        sd_image_t image;
        image.width = width;
        image.height = height;
        image.channel = channel;
        image.data = data.empty() ? nullptr : const_cast<uint8_t*>(data.data());
        return image;
    }
};

bool load_image_file(const std::string& path,
                     int expected_width,
                     int expected_height,
                     int expected_channel,
                     OwnedImage& out,
                     std::string& error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    if (expected_channel <= 0) {
        expected_channel = 3;
    }

    stbi_uc* raw_pixels = stbi_load(path.c_str(), &width, &height, &channels, expected_channel);
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels_guard(raw_pixels, stbi_image_free);

    if (pixels_guard == nullptr) {
        error = "failed to load image from '" + path + "'";
        return false;
    }

    const int actual_channel = expected_channel;
    if (width <= 0 || height <= 0) {
        error = "image '" + path + "' has invalid dimensions";
        return false;
    }

    std::vector<uint8_t> buffer;
    buffer.assign(pixels_guard.get(), pixels_guard.get() + static_cast<size_t>(width) * height * actual_channel);

    if (expected_width > 0 && expected_height > 0 && (width != expected_width || height != expected_height)) {
        float dst_aspect = static_cast<float>(expected_width) / static_cast<float>(expected_height);
        float src_aspect = static_cast<float>(width) / static_cast<float>(height);

        int crop_x = 0;
        int crop_y = 0;
        int crop_w = width;
        int crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = static_cast<int>(std::lround(height * dst_aspect));
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = static_cast<int>(std::lround(width / dst_aspect));
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0 || crop_w != width || crop_h != height) {
            std::vector<uint8_t> cropped(static_cast<size_t>(crop_w) * crop_h * actual_channel);
            for (int row = 0; row < crop_h; ++row) {
                const uint8_t* src = buffer.data() + (static_cast<size_t>(crop_y + row) * width + crop_x) * actual_channel;
                uint8_t* dst = cropped.data() + static_cast<size_t>(row) * crop_w * actual_channel;
                std::memcpy(dst, src, static_cast<size_t>(crop_w) * actual_channel);
            }
            buffer.swap(cropped);
            width = crop_w;
            height = crop_h;
        }

        if (width != expected_width || height != expected_height) {
            std::vector<uint8_t> resized(static_cast<size_t>(expected_width) * expected_height * actual_channel);
            if (!stbir_resize_uint8(buffer.data(),
                                    width,
                                    height,
                                    0,
                                    resized.data(),
                                    expected_width,
                                    expected_height,
                                    0,
                                    actual_channel)) {
                error = "failed to resize image '" + path + "'";
                return false;
            }
            buffer.swap(resized);
            width = expected_width;
            height = expected_height;
        }
    }

    out.width = static_cast<uint32_t>(width);
    out.height = static_cast<uint32_t>(height);
    out.channel = static_cast<uint32_t>(actual_channel);
    out.data = std::move(buffer);
    return true;
}

bool load_images_from_directory(const std::string& directory,
                                int expected_width,
                                int expected_height,
                                int expected_channel,
                                int max_images,
                                std::vector<OwnedImage>& images,
                                std::string& error) {
    std::error_code ec;
    if (!fs::exists(directory, ec) || !fs::is_directory(directory, ec)) {
        error = "directory '" + directory + "' does not exist or is not accessible";
        return false;
    }

    std::vector<fs::directory_entry> candidates;
    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (ec) {
            error = "failed to iterate directory '" + directory + "'";
            return false;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        candidates.push_back(entry);
    }

    std::sort(candidates.begin(), candidates.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
        return to_lower_copy(a.path().filename().string()) < to_lower_copy(b.path().filename().string());
    });

    images.clear();
    images.reserve(candidates.size());

    auto has_image_extension = [](const fs::path& path) {
        std::string ext = to_lower_copy(path.extension().string());
        return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
    };

    for (const auto& entry : candidates) {
        if (!has_image_extension(entry.path())) {
            continue;
        }
        OwnedImage loaded;
        if (!load_image_file(entry.path().string(), expected_width, expected_height, expected_channel, loaded, error)) {
            return false;
        }
        images.push_back(std::move(loaded));
        if (max_images > 0 && static_cast<int>(images.size()) >= max_images) {
            break;
        }
    }

    if (images.empty()) {
        error = "no images found in directory '" + directory + "'";
        return false;
    }

    return true;
}

struct CLIOptions {
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string qwen2vl_path;
    std::string qwen2vl_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string control_net_path;
    std::string lora_model_dir;
    std::string embedding_dir;
    std::string photo_maker_path;
    int port = 8000;
    int n_threads = -1;
    bool verbose = false;
    bool diffusion_flash_attn = false;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct = false;
    bool offload_params_to_cpu = false;
    bool control_net_cpu = false;
    bool clip_on_cpu = false;
    bool vae_on_cpu = false;
    bool force_sdxl_vae_conv_scale = false;
    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask = false;
    int chroma_t5_mask_pad = 1;
    float flow_shift = std::numeric_limits<float>::infinity();
    sd_type_t wtype = SD_TYPE_COUNT;
    rng_type_t rng_type = CUDA_RNG;
    prediction_t prediction = DEFAULT_PRED;
    bool easycache_provided = false;
    sd_easycache_params_t easycache_params = {false, 0.2f, 0.15f, 0.95f};
};

void print_usage() {
    std::cout
        << "Usage: sd-server -m <model_path> [options]\n"
        << "\n"
        << "Model & encoder paths:\n"
        << "  -m, --model <path>                      Primary model path (.gguf)\n"
        << "      --diffusion-model <path>            Standalone diffusion model path\n"
        << "      --high-noise-diffusion-model <path> Standalone high-noise diffusion model path\n"
        << "      --vae <path>                        Standalone VAE path\n"
        << "      --taesd <path>                      Tiny autoencoder path\n"
        << "      --clip_l <path>                     CLIP-L text encoder\n"
        << "      --clip_g <path>                     CLIP-G text encoder\n"
        << "      --clip_vision <path>                CLIP-Vision encoder\n"
        << "      --t5xxl <path>                      T5 XXL text encoder\n"
        << "      --qwen2vl <path>                    Qwen2VL text encoder\n"
        << "      --qwen2vl_vision <path>             Qwen2VL vision encoder\n"
        << "      --control-net <path>                ControlNet model path\n"
        << "      --lora-model-dir <path>             Directory containing LoRA weights\n"
        << "      --embd-dir <path>                   Directory containing textual inversion embeddings\n"
        << "      --photo-maker <path>                PhotoMaker model path\n"
        << "\n"
        << "Runtime options:\n"
        << "  -p, --port <port>                       HTTP port (default 8000)\n"
        << "  -t, --threads <n>                       Number of CPU threads (-1 auto)\n"
        << "      --type <format>                     Weight type override (e.g. f16, q8_0)\n"
        << "      --rng <type>                        RNG, one of [std_default, cuda]\n"
        << "      --prediction <type>                 Prediction override [eps, v, edm_v, sd3_flow, flux_flow]\n"
        << "      --flow-shift <value>                Flow model shift override\n"
        << "      --easycache <thr,start,end>         Enable EasyCache with threshold/start/end percents\n"
        << "      --chroma-t5-mask-pad <int>          Padding for Chroma T5 mask\n"
        << "\n"
        << "Device placement:\n"
        << "      --offload-to-cpu                    Offload model weights to CPU RAM\n"
        << "      --control-net-cpu                   Keep ControlNet on CPU\n"
        << "      --clip-on-cpu                       Keep CLIP on CPU\n"
        << "      --vae-on-cpu                        Keep VAE on CPU\n"
        << "\n"
        << "Kernel options:\n"
        << "      --diffusion-fa                      Enable flash attention in diffusion UNet\n"
        << "      --diffusion-conv-direct             Use ggml_conv2d_direct for diffusion\n"
        << "      --vae-conv-direct                   Use ggml_conv2d_direct for VAE\n"
        << "      --force-sdxl-vae-conv-scale         Force conv scale for SDXL VAE\n"
        << "\n"
        << "Chroma:\n"
        << "      --chroma-disable-dit-mask           Disable DiT mask usage\n"
        << "      --chroma-enable-t5-mask             Enable T5 mask usage\n"
        << "\n"
        << "General:\n"
        << "  -v, --verbose                           Verbose logging\n"
        << "  -h, --help                              Show this help message\n"
        << std::endl;
}

bool parse_arguments(int argc, char** argv, CLIOptions& options, bool& show_help, std::string& error) {
    show_help = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                error = "missing value for -m/--model";
                return false;
            }
            options.model_path = argv[++i];
        } else if (arg == "--clip_l") {
            if (i + 1 >= argc) {
                error = "missing value for --clip_l";
                return false;
            }
            options.clip_l_path = argv[++i];
        } else if (arg == "--clip_g") {
            if (i + 1 >= argc) {
                error = "missing value for --clip_g";
                return false;
            }
            options.clip_g_path = argv[++i];
        } else if (arg == "--clip_vision") {
            if (i + 1 >= argc) {
                error = "missing value for --clip_vision";
                return false;
            }
            options.clip_vision_path = argv[++i];
        } else if (arg == "--t5xxl") {
            if (i + 1 >= argc) {
                error = "missing value for --t5xxl";
                return false;
            }
            options.t5xxl_path = argv[++i];
        } else if (arg == "--qwen2vl") {
            if (i + 1 >= argc) {
                error = "missing value for --qwen2vl";
                return false;
            }
            options.qwen2vl_path = argv[++i];
        } else if (arg == "--qwen2vl_vision") {
            if (i + 1 >= argc) {
                error = "missing value for --qwen2vl_vision";
                return false;
            }
            options.qwen2vl_vision_path = argv[++i];
        } else if (arg == "--diffusion-model") {
            if (i + 1 >= argc) {
                error = "missing value for --diffusion-model";
                return false;
            }
            options.diffusion_model_path = argv[++i];
        } else if (arg == "--high-noise-diffusion-model") {
            if (i + 1 >= argc) {
                error = "missing value for --high-noise-diffusion-model";
                return false;
            }
            options.high_noise_diffusion_model_path = argv[++i];
        } else if (arg == "--vae") {
            if (i + 1 >= argc) {
                error = "missing value for --vae";
                return false;
            }
            options.vae_path = argv[++i];
        } else if (arg == "--taesd") {
            if (i + 1 >= argc) {
                error = "missing value for --taesd";
                return false;
            }
            options.taesd_path = argv[++i];
        } else if (arg == "--control-net") {
            if (i + 1 >= argc) {
                error = "missing value for --control-net";
                return false;
            }
            options.control_net_path = argv[++i];
        } else if (arg == "--lora-model-dir") {
            if (i + 1 >= argc) {
                error = "missing value for --lora-model-dir";
                return false;
            }
            options.lora_model_dir = argv[++i];
        } else if (arg == "--embd-dir") {
            if (i + 1 >= argc) {
                error = "missing value for --embd-dir";
                return false;
            }
            options.embedding_dir = argv[++i];
        } else if (arg == "--photo-maker") {
            if (i + 1 >= argc) {
                error = "missing value for --photo-maker";
                return false;
            }
            options.photo_maker_path = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 >= argc) {
                error = "missing value for --port";
                return false;
            }
            try {
                options.port = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                error = "invalid port value";
                return false;
            }
            if (options.port <= 0 || options.port > 65535) {
                error = "port must be between 1 and 65535";
                return false;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 >= argc) {
                error = "missing value for -t/--threads";
                return false;
            }
            try {
                options.n_threads = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                error = "invalid threads value";
                return false;
            }
            if (options.n_threads == 0) {
                error = "thread count must not be zero";
                return false;
            }
        } else if (arg == "--type") {
            if (i + 1 >= argc) {
                error = "missing value for --type";
                return false;
            }
            std::string value = to_lower_copy(argv[++i]);
            sd_type_t type = str_to_sd_type(value.c_str());
            if (type == SD_TYPE_COUNT) {
                error = "invalid weight type '" + value + "'";
                return false;
            }
            options.wtype = type;
        } else if (arg == "--rng") {
            if (i + 1 >= argc) {
                error = "missing value for --rng";
                return false;
            }
            std::string value = to_lower_copy(argv[++i]);
            rng_type_t rng = str_to_rng_type(value.c_str());
            if (rng == RNG_TYPE_COUNT) {
                error = "invalid rng '" + value + "'";
                return false;
            }
            options.rng_type = rng;
        } else if (arg == "--prediction") {
            if (i + 1 >= argc) {
                error = "missing value for --prediction";
                return false;
            }
            std::string value = to_lower_copy(argv[++i]);
            prediction_t prediction = str_to_prediction(value.c_str());
            if (prediction == PREDICTION_COUNT) {
                error = "invalid prediction '" + value + "'";
                return false;
            }
            options.prediction = prediction;
        } else if (arg == "--flow-shift") {
            if (i + 1 >= argc) {
                error = "missing value for --flow-shift";
                return false;
            }
            std::string value = argv[++i];
            if (value == "auto") {
                options.flow_shift = std::numeric_limits<float>::infinity();
            } else {
                try {
                    options.flow_shift = std::stof(value);
                } catch (const std::exception&) {
                    error = "invalid float for --flow-shift";
                    return false;
                }
            }
        } else if (arg == "--easycache") {
            if (i + 1 >= argc) {
                error = "missing value for --easycache";
                return false;
            }
            std::string value = argv[++i];
            float parsed[3] = {0.0f, 0.0f, 0.0f};
            std::stringstream ss(value);
            std::string token;
            int idx = 0;
            auto trim = [](std::string& s) {
                const char* whitespace = " \t\r\n";
                auto start = s.find_first_not_of(whitespace);
                if (start == std::string::npos) {
                    s.clear();
                    return;
                }
                auto end = s.find_last_not_of(whitespace);
                s = s.substr(start, end - start + 1);
            };
            while (std::getline(ss, token, ',')) {
                trim(token);
                if (token.empty()) {
                    error = "invalid easycache value";
                    return false;
                }
                if (idx >= 3) {
                    error = "easycache expects exactly 3 comma-separated values";
                    return false;
                }
                try {
                    parsed[idx] = std::stof(token);
                } catch (const std::exception&) {
                    error = "invalid easycache value";
                    return false;
                }
                idx++;
            }
            if (idx != 3) {
                error = "easycache expects exactly 3 comma-separated values";
                return false;
            }
            if (parsed[0] < 0.0f) {
                error = "easycache threshold must be non-negative";
                return false;
            }
            if (parsed[1] < 0.0f || parsed[1] >= 1.0f || parsed[2] <= 0.0f || parsed[2] > 1.0f || parsed[1] >= parsed[2]) {
                error = "easycache start/end percents must satisfy 0.0 <= start < end <= 1.0";
                return false;
            }
            options.easycache_params.enabled = true;
            options.easycache_params.reuse_threshold = parsed[0];
            options.easycache_params.start_percent = parsed[1];
            options.easycache_params.end_percent = parsed[2];
            options.easycache_provided = true;
        } else if (arg == "-v" || arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--diffusion-fa") {
            options.diffusion_flash_attn = true;
        } else if (arg == "--diffusion-conv-direct") {
            options.diffusion_conv_direct = true;
        } else if (arg == "--vae-conv-direct") {
            options.vae_conv_direct = true;
        } else if (arg == "--offload-to-cpu") {
            options.offload_params_to_cpu = true;
        } else if (arg == "--control-net-cpu") {
            options.control_net_cpu = true;
        } else if (arg == "--clip-on-cpu") {
            options.clip_on_cpu = true;
        } else if (arg == "--vae-on-cpu") {
            options.vae_on_cpu = true;
        } else if (arg == "--force-sdxl-vae-conv-scale") {
            options.force_sdxl_vae_conv_scale = true;
        } else if (arg == "--chroma-disable-dit-mask") {
            options.chroma_use_dit_mask = false;
        } else if (arg == "--chroma-enable-t5-mask") {
            options.chroma_use_t5_mask = true;
        } else if (arg == "--chroma-t5-mask-pad") {
            if (i + 1 >= argc) {
                error = "missing value for --chroma-t5-mask-pad";
                return false;
            }
            try {
                options.chroma_t5_mask_pad = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                error = "invalid integer for --chroma-t5-mask-pad";
                return false;
            }
            if (options.chroma_t5_mask_pad < 0) {
                error = "--chroma-t5-mask-pad must be non-negative";
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
            return false;
        } else {
            error = "unknown argument: " + arg;
            return false;
        }
    }

    if (options.n_threads < 0) {
        options.n_threads = -1;
    }

    if (options.model_path.empty() && options.diffusion_model_path.empty()) {
        error = "model path is required (-m/--model or --diffusion-model)";
        return false;
    }

    return true;
}

struct CtxConfig {
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string qwen2vl_path;
    std::string qwen2vl_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string control_net_path;
    std::string lora_model_dir;
    std::string embedding_dir;
    std::string photo_maker_path;
    bool vae_decode_only = true;
    bool free_params_immediately = true;
    int n_threads = -1;
    sd_type_t wtype = SD_TYPE_COUNT;
    rng_type_t rng_type = CUDA_RNG;
    bool offload_params_to_cpu = false;
    bool keep_clip_on_cpu = false;
    bool keep_control_net_on_cpu = false;
    bool keep_vae_on_cpu = false;
    bool diffusion_flash_attn = false;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct = false;
    bool force_sdxl_vae_conv_scale = false;
    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask = false;
    int chroma_t5_mask_pad = 1;
    float flow_shift = std::numeric_limits<float>::infinity();
    prediction_t prediction = DEFAULT_PRED;

    bool operator==(const CtxConfig& other) const {
        return model_path == other.model_path &&
               clip_l_path == other.clip_l_path &&
               clip_g_path == other.clip_g_path &&
               clip_vision_path == other.clip_vision_path &&
               t5xxl_path == other.t5xxl_path &&
               qwen2vl_path == other.qwen2vl_path &&
               qwen2vl_vision_path == other.qwen2vl_vision_path &&
               diffusion_model_path == other.diffusion_model_path &&
               high_noise_diffusion_model_path == other.high_noise_diffusion_model_path &&
               vae_path == other.vae_path &&
               taesd_path == other.taesd_path &&
               control_net_path == other.control_net_path &&
               lora_model_dir == other.lora_model_dir &&
               embedding_dir == other.embedding_dir &&
               photo_maker_path == other.photo_maker_path &&
               vae_decode_only == other.vae_decode_only &&
               n_threads == other.n_threads &&
               wtype == other.wtype &&
               rng_type == other.rng_type &&
               offload_params_to_cpu == other.offload_params_to_cpu &&
               keep_clip_on_cpu == other.keep_clip_on_cpu &&
               keep_control_net_on_cpu == other.keep_control_net_on_cpu &&
               keep_vae_on_cpu == other.keep_vae_on_cpu &&
               diffusion_flash_attn == other.diffusion_flash_attn &&
               diffusion_conv_direct == other.diffusion_conv_direct &&
               vae_conv_direct == other.vae_conv_direct &&
               force_sdxl_vae_conv_scale == other.force_sdxl_vae_conv_scale &&
               chroma_use_dit_mask == other.chroma_use_dit_mask &&
               chroma_use_t5_mask == other.chroma_use_t5_mask &&
               chroma_t5_mask_pad == other.chroma_t5_mask_pad &&
               flow_shift == other.flow_shift &&
               prediction == other.prediction;
    }

    bool operator!=(const CtxConfig& other) const { return !(*this == other); }

    sd_ctx_params_t to_sd_params() const {
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);

        params.model_path                      = model_path.c_str();
        params.clip_l_path                     = clip_l_path.c_str();
        params.clip_g_path                     = clip_g_path.c_str();
        params.clip_vision_path                = clip_vision_path.c_str();
        params.t5xxl_path                      = t5xxl_path.c_str();
        params.qwen2vl_path                    = qwen2vl_path.c_str();
        params.qwen2vl_vision_path             = qwen2vl_vision_path.c_str();
        params.diffusion_model_path            = diffusion_model_path.c_str();
        params.high_noise_diffusion_model_path = high_noise_diffusion_model_path.c_str();
        params.vae_path                        = vae_path.c_str();
        params.taesd_path                      = taesd_path.c_str();
        params.control_net_path                = control_net_path.c_str();
        params.lora_model_dir                  = lora_model_dir.c_str();
        params.embedding_dir                   = embedding_dir.c_str();
        params.photo_maker_path                = photo_maker_path.c_str();
        params.vae_decode_only                 = vae_decode_only;
        params.free_params_immediately         = free_params_immediately;
        if (n_threads > 0) {
            params.n_threads = n_threads;
        }
        params.wtype                   = wtype;
        params.rng_type                = rng_type;
        params.offload_params_to_cpu   = offload_params_to_cpu;
        params.keep_clip_on_cpu        = keep_clip_on_cpu;
        params.keep_control_net_on_cpu = keep_control_net_on_cpu;
        params.keep_vae_on_cpu         = keep_vae_on_cpu;
        params.diffusion_flash_attn    = diffusion_flash_attn;
        params.diffusion_conv_direct   = diffusion_conv_direct;
        params.vae_conv_direct         = vae_conv_direct;
        params.force_sdxl_vae_conv_scale = force_sdxl_vae_conv_scale;
        params.chroma_use_dit_mask     = chroma_use_dit_mask;
        params.chroma_use_t5_mask      = chroma_use_t5_mask;
        params.chroma_t5_mask_pad      = chroma_t5_mask_pad;
        params.flow_shift              = flow_shift;
        params.prediction              = prediction;

        return params;
    }
};

struct LogEntry {
    sd_log_level_t level = SD_LOG_INFO;
    std::string message;
};

struct LogCollector {
    void add(sd_log_level_t level, const std::string& text) {
        entries.push_back({level, text});
    }

    std::vector<LogEntry> entries;
};
struct ServerState {
    std::mutex mutex;
    std::mutex log_mutex;
    sd_ctx_t* ctx = nullptr;
    CtxConfig ctx_config;
    CtxConfig default_config;
    LogCollector* active_collector = nullptr;
    std::string pending_log_fragment;
    bool verbose = false;
    sd_easycache_params_t default_easycache = {false, 0.2f, 0.15f, 0.95f};
};

class LogCaptureScope {
   public:
    LogCaptureScope(ServerState& state, LogCollector& collector) : state_(state), collector_(collector) {
        std::lock_guard<std::mutex> guard(state_.log_mutex);
        state_.active_collector = &collector_;
    }

    ~LogCaptureScope() {
        std::lock_guard<std::mutex> guard(state_.log_mutex);
        if (state_.active_collector == &collector_) {
            state_.active_collector = nullptr;
        }
    }

   private:
    ServerState& state_;
    LogCollector& collector_;
};

struct GenerationRequest {
    std::vector<std::string> lora_paths;
    std::vector<float> lora_weights;
    std::string prompt;
    std::string negative_prompt;
    int clip_skip = -1;
    int width = 512;
    int height = 512;
    int sample_steps = 20;
    float cfg_scale = 7.0f;
    bool has_img_cfg_scale = false;
    float img_cfg_scale = 7.0f;
    bool override_sample_method = false;
    sample_method_t sample_method = SAMPLE_METHOD_DEFAULT;
    bool override_scheduler = false;
    scheduler_t scheduler = DEFAULT;
    int batch_count = 1;
    int64_t seed = -1;
    float eta = 0.0f;
    bool has_eta = false;
    int shifted_timestep = 0;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    bool has_vae_tiling_override = false;
    float distilled_guidance = 3.5f;
    float slg_scale = 0.0f;
    float slg_layer_start = 0.01f;
    float slg_layer_end = 0.2f;
    std::vector<int> slg_layers = {7, 8, 9};
    bool auto_resize_ref_image = true;
    bool increase_ref_index = false;
    float strength = 0.75f;
    float control_strength = 0.9f;
    bool canny_preprocess = false;
    std::string init_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    float pm_style_strength = 20.0f;
    OwnedImage init_image;
    bool has_init_image = false;
    OwnedImage mask_image;
    bool has_mask_image = false;
    OwnedImage control_image;
    bool has_control_image = false;
    std::vector<OwnedImage> ref_images;
    std::vector<OwnedImage> pm_id_images;
    sd_easycache_params_t easycache = {false, 0.2f, 0.15f, 0.95f};
    bool easycache_provided = false;
};

struct ImageResultGuard {
    sd_image_t* images = nullptr;
    int count = 0;

    ~ImageResultGuard() {
        if (images != nullptr) {
            for (int i = 0; i < count; ++i) {
                free(images[i].data);
                images[i].data = nullptr;
            }
            free(images);
        }
    }
};

json logs_to_json(const LogCollector& collector);
json make_telemetry(const LogCollector& collector,
                    const GenerationRequest& request,
                    const CtxConfig& config,
                    int64_t elapsed_ms,
                    int64_t effective_seed);

class StreamingImageResponder {
   public:
    StreamingImageResponder(ServerState& state,
                            std::unique_lock<std::mutex>&& ctx_lock,
                            std::unique_ptr<LogCaptureScope>&& capture_scope,
                            std::shared_ptr<LogCollector> collector,
                            GenerationRequest request,
                            CtxConfig ctx_config,
                            bool random_seed_requested,
                            int64_t effective_seed)
        : state_(state),
          ctx_lock_(std::move(ctx_lock)),
          capture_scope_(std::move(capture_scope)),
          collector_(std::move(collector)),
          request_(std::move(request)),
          ctx_config_(std::move(ctx_config)),
          random_seed_requested_(random_seed_requested),
          effective_seed_(effective_seed),
          default_sample_method_(sd_get_default_sample_method(state.ctx)),
          start_time_(std::chrono::steady_clock::now()) {}

    ~StreamingImageResponder() {
        finalize_resources();
    }

    bool next(httplib::DataSink& sink) {
        if (done_) {
            return false;
        }

        if (next_index_ < request_.batch_count) {
            if (!emit_image_chunk(sink, next_index_)) {
                done_ = true;
                return false;
            }
            ++next_index_;
            return true;
        }

        emit_final_summary(sink);
        done_ = true;
        return false;
    }

    void cancel() {
        done_ = true;
        finalize_resources();
    }

   private:
    bool emit_image_chunk(httplib::DataSink& sink, int index) {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);

        params.prompt = request_.prompt.c_str();
        params.negative_prompt = request_.negative_prompt.c_str();
        params.clip_skip = request_.clip_skip;
        params.width = request_.width;
        params.height = request_.height;
        params.batch_count = 1;
        params.seed = effective_seed_ + index;
        params.auto_resize_ref_image = request_.auto_resize_ref_image;
        params.increase_ref_index = request_.increase_ref_index;
        params.strength = request_.strength;
        params.control_strength = request_.control_strength;
        params.easycache = request_.easycache;
        if (request_.has_init_image) {
            params.init_image = request_.init_image.as_sd_image();
        }
        if (request_.has_mask_image) {
            params.mask_image = request_.mask_image.as_sd_image();
        }
        if (request_.has_control_image) {
            params.control_image = request_.control_image.as_sd_image();
        }
        if (request_.has_vae_tiling_override) {
            params.vae_tiling_params = request_.vae_tiling_params;
        }

        std::vector<sd_image_t> ref_views;
        if (!request_.ref_images.empty()) {
            ref_views.reserve(request_.ref_images.size());
            for (const auto& image : request_.ref_images) {
                ref_views.push_back(image.as_sd_image());
            }
            params.ref_images = ref_views.data();
            params.ref_images_count = static_cast<int>(ref_views.size());
        }

        std::vector<sd_image_t> pm_views;
        if (!request_.pm_id_images.empty()) {
            pm_views.reserve(request_.pm_id_images.size());
            for (const auto& image : request_.pm_id_images) {
                pm_views.push_back(image.as_sd_image());
            }
        }
        params.pm_params.id_images = pm_views.empty() ? nullptr : pm_views.data();
        params.pm_params.id_images_count = static_cast<int>(pm_views.size());
        params.pm_params.id_embed_path = request_.pm_id_embed_path.empty() ? nullptr : request_.pm_id_embed_path.c_str();
        params.pm_params.style_strength = request_.pm_style_strength;

        sd_sample_params_t& sample_params = params.sample_params;
        sample_params.sample_steps = request_.sample_steps;
        sample_params.guidance.txt_cfg = request_.cfg_scale;
        if (request_.has_img_cfg_scale) {
            sample_params.guidance.img_cfg = request_.img_cfg_scale;
        }
        if (!std::isfinite(sample_params.guidance.img_cfg)) {
            sample_params.guidance.img_cfg = sample_params.guidance.txt_cfg;
        }
        sample_params.guidance.distilled_guidance = request_.distilled_guidance;
        sample_params.guidance.slg.layer_start = request_.slg_layer_start;
        sample_params.guidance.slg.layer_end = request_.slg_layer_end;
        sample_params.guidance.slg.scale = request_.slg_scale;
        if (!request_.slg_layers.empty()) {
            sample_params.guidance.slg.layers = request_.slg_layers.data();
            sample_params.guidance.slg.layer_count = request_.slg_layers.size();
        } else {
            sample_params.guidance.slg.layers = nullptr;
            sample_params.guidance.slg.layer_count = 0;
        }
        if (request_.override_sample_method) {
            sample_params.sample_method = request_.sample_method;
        }
        if (sample_params.sample_method == SAMPLE_METHOD_DEFAULT) {
            sample_params.sample_method = default_sample_method_;
        }
        if (request_.override_scheduler) {
            sample_params.scheduler = request_.scheduler;
        }
        if (request_.has_eta) {
            sample_params.eta = request_.eta;
        }
        sample_params.shifted_timestep = request_.shifted_timestep;

        sd_image_t* results = generate_image(state_.ctx, &params);
        if (results == nullptr) {
            emit_error(sink, "image generation failed", index);
            return false;
        }

        ImageResultGuard guard{results, params.batch_count};

        sd_image_t& image = results[0];
        if (image.data == nullptr) {
            emit_error(sink, "image data is empty", index);
            return false;
        }

        auto encode_start = std::chrono::steady_clock::now();

        int png_size = 0;
        unsigned char* png_data = stbi_write_png_to_mem(image.data, 0, image.width, image.height, image.channel, &png_size, nullptr);
        if (png_data == nullptr) {
            emit_error(sink, "failed to encode PNG", index);
            return false;
        }
        std::string encoded = base64_encode(png_data, static_cast<size_t>(png_size));
        STBIW_FREE(png_data);

        auto encode_end = std::chrono::steady_clock::now();
        const double encode_ms = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start).count() / 1000.0;
        const std::size_t encoded_size = encoded.size();

        // Preserve the legacy -1 seed while still reporting the concrete seed that was used.
        int64_t actual_seed = random_seed_requested_ ? (effective_seed_ + index) : (request_.seed + index);
        int64_t reported_seed = random_seed_requested_ ? -1 : actual_seed;

        json image_chunk = json::object();
        image_chunk["type"] = "image";
        image_chunk["index"] = index;
        image_chunk["seed"] = reported_seed;
        image_chunk["actual_seed"] = actual_seed;
        image_chunk["width"] = image.width;
        image_chunk["height"] = image.height;
        image_chunk["format"] = "png";
        image_chunk["mime_type"] = "image/png";
        image_chunk["payload_bytes"] = png_size;
        image_chunk["encoded_bytes"] = static_cast<int64_t>(encoded_size);
        image_chunk["encode_ms"] = encode_ms;
        image_chunk["data"] = std::move(encoded);

        auto prepare_end = std::chrono::steady_clock::now();
        const double prepare_ms = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - encode_start).count() / 1000.0;
        image_chunk["dispatch_prepare_ms"] = prepare_ms;

        std::size_t serialized_bytes = 0;
        if (!write_json_array_item(sink, image_chunk, false, &serialized_bytes)) {
            done_ = true;
            finalize_resources();
            return false;
        }

        auto dispatch_end = std::chrono::steady_clock::now();
        const double dispatch_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(dispatch_end - encode_start).count() / 1000.0;
        const double write_ms = std::chrono::duration_cast<std::chrono::microseconds>(dispatch_end - prepare_end).count() / 1000.0;

        json summary_entry = json::object();
        summary_entry["index"] = index;
        summary_entry["seed"] = reported_seed;
        summary_entry["actual_seed"] = actual_seed;
        summary_entry["width"] = image.width;
        summary_entry["height"] = image.height;
        summary_entry["format"] = "png";
        summary_entry["mime_type"] = "image/png";
        summary_entry["streamed"] = true;
        summary_entry["encode_ms"] = encode_ms;
        summary_entry["dispatch_prepare_ms"] = prepare_ms;
        summary_entry["dispatch_total_ms"] = dispatch_total_ms;
        summary_entry["write_ms"] = write_ms;
        summary_entry["payload_bytes"] = png_size;
        summary_entry["encoded_bytes"] = static_cast<int64_t>(encoded_size);
        summary_entry["serialized_bytes"] = static_cast<int64_t>(serialized_bytes);
        image_summaries_.push_back(std::move(summary_entry));

        return true;
    }

    void emit_error(httplib::DataSink& sink, const std::string& message, int index) {
        encountered_error_ = true;
        done_ = true;
        const int64_t elapsed = elapsed_ms();
        json error_chunk = json::object();
        error_chunk["type"] = "error";
        error_chunk["success"] = false;
        error_chunk["error"] = message;
        error_chunk["index"] = index;
        error_chunk["requested_seed"] = request_.seed;
        error_chunk["applied_seed"] = effective_seed_;
        error_chunk["random_seed_requested"] = random_seed_requested_;
        error_chunk["elapsed_ms"] = elapsed;
        if (!ctx_config_.model_path.empty()) {
            error_chunk["model_path"] = ctx_config_.model_path;
        }
        error_chunk["logs"] = logs_to_json(*collector_);
        error_chunk["telemetry"] = make_telemetry(*collector_, request_, ctx_config_, elapsed, effective_seed_);
        if (write_json_array_item(sink, error_chunk, true)) {
            finalize_stream(sink);
        } else {
            finalize_resources();
        }
    }

    void emit_final_summary(httplib::DataSink& sink) {
        const int64_t elapsed = elapsed_ms();
        json summary = json::object();
        summary["type"] = "complete";
        summary["success"] = !encountered_error_;
        summary["batch_count"] = request_.batch_count;
        summary["requested_seed"] = request_.seed;
        summary["applied_seed"] = effective_seed_;
        summary["random_seed_requested"] = random_seed_requested_;
        summary["elapsed_ms"] = elapsed;
        if (!ctx_config_.model_path.empty()) {
            summary["model_path"] = ctx_config_.model_path;
        }
        summary["images"] = image_summaries_;
        summary["logs"] = logs_to_json(*collector_);
        summary["telemetry"] = make_telemetry(*collector_, request_, ctx_config_, elapsed, effective_seed_);
        done_ = true;
        if (write_json_array_item(sink, summary, true)) {
            finalize_stream(sink);
        } else {
            finalize_resources();
        }
    }

    int64_t elapsed_ms() const {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    }

    bool write_json_array_item(httplib::DataSink& sink,
                               const json& payload,
                               bool final_item,
                               std::size_t* serialized_size = nullptr) {
        std::string serialized = payload.dump();
        if (serialized_size != nullptr) {
            *serialized_size = serialized.size();
        }

        if (!array_opened_) {
            const char prefix[] = "[\n";
            if (!sink.write(prefix, sizeof(prefix) - 1)) {
                return false;
            }
            sink.os.flush();
            array_opened_ = true;
        }

        if (!first_object_) {
            const char separator[] = ",\n";
            if (!sink.write(separator, sizeof(separator) - 1)) {
                return false;
            }
            sink.os.flush();
        }

        std::string chunk = std::move(serialized);
        if (final_item) {
            chunk.append("\n]");
        }
        chunk.push_back('\n');
        bool ok = sink.write(chunk.data(), chunk.size());
        if (ok) {
            sink.os.flush();
        }
        first_object_ = false;
        return ok;
    }

    void finalize_stream(httplib::DataSink& sink) {
        if (sink.done) {
            sink.done();
        }
        finalize_resources();
    }

    void finalize_resources() {
        if (finalized_) {
            return;
        }
        capture_scope_.reset();
        collector_.reset();
        if (ctx_lock_.owns_lock()) {
            ctx_lock_.unlock();
        }
        finalized_ = true;
    }

    ServerState& state_;
    std::unique_lock<std::mutex> ctx_lock_;
    std::unique_ptr<LogCaptureScope> capture_scope_;
    std::shared_ptr<LogCollector> collector_;
    GenerationRequest request_;
    CtxConfig ctx_config_;
    bool random_seed_requested_ = false;
    int64_t effective_seed_ = 0;
    sample_method_t default_sample_method_ = SAMPLE_METHOD_DEFAULT;
    std::chrono::steady_clock::time_point start_time_;
    int next_index_ = 0;
    bool done_ = false;
    bool encountered_error_ = false;
    bool finalized_ = false;
    bool array_opened_ = false;
    bool first_object_ = true;
    std::vector<json> image_summaries_;
};

bool apply_context_overrides(const json& body, CtxConfig& config, std::string& error) {
    auto assign_string = [&](const char* key, std::string& target) -> bool {
        auto it = body.find(key);
        if (it == body.end()) {
            return true;
        }
        if (!it->is_string()) {
            error = std::string("field '") + key + "' must be a string";
            return false;
        }
        target = it->get<std::string>();
        return true;
    };

    auto assign_bool = [&](const char* key, bool& target) -> bool {
        auto it = body.find(key);
        if (it == body.end()) {
            return true;
        }
        if (!it->is_boolean()) {
            error = std::string("field '") + key + "' must be a boolean";
            return false;
        }
        target = it->get<bool>();
        return true;
    };

    auto assign_int = [&](const char* key, int& target) -> bool {
        auto it = body.find(key);
        if (it == body.end()) {
            return true;
        }
        if (!it->is_number_integer()) {
            error = std::string("field '") + key + "' must be an integer";
            return false;
        }
        target = static_cast<int>(it->get<int64_t>());
        return true;
    };

    auto assign_float = [&](const char* key, float& target) -> bool {
        auto it = body.find(key);
        if (it == body.end()) {
            return true;
        }
        if (!it->is_number_float() && !it->is_number_integer()) {
            error = std::string("field '") + key + "' must be numeric";
            return false;
        }
        target = static_cast<float>(it->get<double>());
        return true;
    };

    if (!assign_string("model_path", config.model_path) ||
        !assign_string("clip_l_path", config.clip_l_path) ||
        !assign_string("clip_g_path", config.clip_g_path) ||
        !assign_string("clip_vision_path", config.clip_vision_path) ||
        !assign_string("t5xxl_path", config.t5xxl_path) ||
        !assign_string("qwen2vl_path", config.qwen2vl_path) ||
        !assign_string("qwen2vl_vision_path", config.qwen2vl_vision_path) ||
        !assign_string("diffusion_model_path", config.diffusion_model_path) ||
        !assign_string("high_noise_diffusion_model_path", config.high_noise_diffusion_model_path) ||
        !assign_string("vae_path", config.vae_path) ||
        !assign_string("taesd_path", config.taesd_path) ||
        !assign_string("control_net_path", config.control_net_path) ||
        !assign_string("lora_model_dir", config.lora_model_dir) ||
        !assign_string("embedding_dir", config.embedding_dir) ||
        !assign_string("photo_maker_path", config.photo_maker_path)) {
        return false;
    }

    if (!assign_bool("vae_decode_only", config.vae_decode_only) ||
        !assign_bool("free_params_immediately", config.free_params_immediately) ||
        !assign_bool("offload_params_to_cpu", config.offload_params_to_cpu) ||
        !assign_bool("clip_on_cpu", config.keep_clip_on_cpu) ||
        !assign_bool("control_net_cpu", config.keep_control_net_on_cpu) ||
        !assign_bool("vae_on_cpu", config.keep_vae_on_cpu) ||
        !assign_bool("diffusion_flash_attn", config.diffusion_flash_attn) ||
        !assign_bool("diffusion_conv_direct", config.diffusion_conv_direct) ||
        !assign_bool("vae_conv_direct", config.vae_conv_direct) ||
        !assign_bool("force_sdxl_vae_conv_scale", config.force_sdxl_vae_conv_scale) ||
        !assign_bool("chroma_use_dit_mask", config.chroma_use_dit_mask) ||
        !assign_bool("chroma_use_t5_mask", config.chroma_use_t5_mask)) {
        return false;
    }

    if (!assign_int("n_threads", config.n_threads) ||
        !assign_int("chroma_t5_mask_pad", config.chroma_t5_mask_pad)) {
        return false;
    }

    if (!assign_float("flow_shift", config.flow_shift)) {
        return false;
    }

    auto rng_it = body.find("rng_type");
    if (rng_it != body.end()) {
        if (!rng_it->is_string()) {
            error = "field 'rng_type' must be a string";
            return false;
        }
        std::string value = to_lower_copy(rng_it->get<std::string>());
        rng_type_t rng = str_to_rng_type(value.c_str());
        if (rng == RNG_TYPE_COUNT) {
            error = "invalid rng_type value";
            return false;
        }
        config.rng_type = rng;
    }

    auto wtype_it = body.find("wtype");
    if (wtype_it != body.end()) {
        if (!wtype_it->is_string()) {
            error = "field 'wtype' must be a string";
            return false;
        }
        std::string value = to_lower_copy(wtype_it->get<std::string>());
        sd_type_t type = str_to_sd_type(value.c_str());
        if (type == SD_TYPE_COUNT) {
            error = "invalid wtype value";
            return false;
        }
        config.wtype = type;
    }

    auto prediction_it = body.find("prediction");
    if (prediction_it != body.end()) {
        if (!prediction_it->is_string()) {
            error = "field 'prediction' must be a string";
            return false;
        }
        std::string value = to_lower_copy(prediction_it->get<std::string>());
        prediction_t prediction = str_to_prediction(value.c_str());
        if (prediction == PREDICTION_COUNT) {
            error = "invalid prediction value";
            return false;
        }
        config.prediction = prediction;
    }

    return true;
}

bool parse_generation_request(const json& body, GenerationRequest& request, std::string& error) {
    sd_easycache_params_init(&request.easycache);
    request.easycache_provided = false;
    auto prompt_it = body.find("prompt");
    if (prompt_it == body.end() || !prompt_it->is_string()) {
        error = "field 'prompt' is required";
        return false;
    }
    request.prompt = prompt_it->get<std::string>();

    if (request.prompt.empty()) {
        error = "prompt must not be empty";
        return false;
    }

    auto neg_it = body.find("negative_prompt");
    if (neg_it != body.end()) {
        if (!neg_it->is_string()) {
            error = "field 'negative_prompt' must be a string";
            return false;
        }
        request.negative_prompt = neg_it->get<std::string>();
    }

    auto loras_it = body.find("loras");
    auto weights_it = body.find("lora_weights");
    if (loras_it != body.end() || weights_it != body.end()) {
        if (loras_it == body.end() || !loras_it->is_string()) {
            error = "field 'loras' must be a string";
            return false;
        }
        if (weights_it == body.end() || !weights_it->is_string()) {
            error = "field 'lora_weights' must be a string";
            return false;
        }

        const std::string loras_value = loras_it->get<std::string>();
        const std::string weights_value = weights_it->get<std::string>();

        request.lora_paths = split_and_trim(loras_value, ',');
        std::vector<std::string> weight_tokens = split_and_trim(weights_value, ',');

        if (request.lora_paths.empty()) {
            error = "field 'loras' must contain at least one path";
            return false;
        }
        if (weight_tokens.size() != request.lora_paths.size()) {
            error = "fields 'loras' and 'lora_weights' must have the same number of entries";
            return false;
        }

        request.lora_weights.clear();
        request.lora_weights.reserve(weight_tokens.size());

        for (size_t i = 0; i < request.lora_paths.size(); ++i) {
            if (request.lora_paths[i].empty()) {
                error = "lora path entries must not be empty";
                return false;
            }
            const std::string& weight_token = weight_tokens[i];
            if (weight_token.empty()) {
                error = "lora weight entries must not be empty";
                return false;
            }
            size_t pos = 0;
            float weight = 0.0f;
            try {
                weight = std::stof(weight_token, &pos);
            } catch (const std::exception&) {
                error = std::string("invalid lora weight value: '") + weight_token + "'";
                return false;
            }
            if (pos != weight_token.size()) {
                error = std::string("invalid lora weight value: '") + weight_token + "'";
                return false;
            }
            request.lora_weights.push_back(weight);
        }

        for (size_t i = 0; i < request.lora_paths.size(); ++i) {
            request.prompt += "<lora:" + request.lora_paths[i] + ":" + weight_tokens[i] + ">";
        }
    }

    auto clip_skip_it = body.find("clip_skip");
    if (clip_skip_it != body.end()) {
        if (!clip_skip_it->is_number_integer()) {
            error = "field 'clip_skip' must be an integer";
            return false;
        }
        request.clip_skip = static_cast<int>(clip_skip_it->get<int64_t>());
    }

    auto width_it = body.find("width");
    if (width_it != body.end()) {
        if (!width_it->is_number_integer()) {
            error = "field 'width' must be an integer";
            return false;
        }
        request.width = static_cast<int>(width_it->get<int64_t>());
    }
    if (request.width <= 0) {
        error = "width must be greater than 0";
        return false;
    }

    auto height_it = body.find("height");
    if (height_it != body.end()) {
        if (!height_it->is_number_integer()) {
            error = "field 'height' must be an integer";
            return false;
        }
        request.height = static_cast<int>(height_it->get<int64_t>());
    }
    if (request.height <= 0) {
        error = "height must be greater than 0";
        return false;
    }

    auto steps_it = body.find("sample_steps");
    if (steps_it != body.end()) {
        if (!steps_it->is_number_integer()) {
            error = "field 'sample_steps' must be an integer";
            return false;
        }
        request.sample_steps = static_cast<int>(steps_it->get<int64_t>());
    }
    if (request.sample_steps <= 0) {
        error = "sample_steps must be greater than 0";
        return false;
    }

    bool easycache_flag_explicit = false;
    auto apply_easycache_values = [&](float threshold, float start, float end) -> bool {
        if (threshold < 0.0f) {
            error = "easycache threshold must be non-negative";
            return false;
        }
        if (start < 0.0f || start >= 1.0f || end <= 0.0f || end > 1.0f || start >= end) {
            error = "easycache start/end percents must satisfy 0.0 <= start < end <= 1.0";
            return false;
        }
        request.easycache.reuse_threshold = threshold;
        request.easycache.start_percent = start;
        request.easycache.end_percent = end;
        request.easycache_provided = true;
        return true;
    };

    auto parse_easycache_triplet = [&](const std::string& value) -> bool {
        float parsed[3] = {0.0f, 0.0f, 0.0f};
        std::stringstream ss(value);
        std::string token;
        int idx = 0;
        auto trim = [](std::string& s) {
            const char* whitespace = " \t\r\n";
            auto start = s.find_first_not_of(whitespace);
            if (start == std::string::npos) {
                s.clear();
                return;
            }
            auto end = s.find_last_not_of(whitespace);
            s = s.substr(start, end - start + 1);
        };
        while (std::getline(ss, token, ',')) {
            trim(token);
            if (token.empty()) {
                error = "invalid easycache value";
                return false;
            }
            if (idx >= 3) {
                error = "easycache expects exactly 3 comma-separated values";
                return false;
            }
            try {
                parsed[idx] = std::stof(token);
            } catch (const std::exception&) {
                error = "invalid easycache value";
                return false;
            }
            idx++;
        }
        if (idx != 3) {
            error = "easycache expects exactly 3 comma-separated values";
            return false;
        }
        return apply_easycache_values(parsed[0], parsed[1], parsed[2]);
    };

    auto easycache_it = body.find("easycache");
    if (easycache_it != body.end()) {
        if (easycache_it->is_boolean()) {
            request.easycache.enabled = easycache_it->get<bool>();
            request.easycache_provided = true;
            easycache_flag_explicit = true;
        } else if (easycache_it->is_string()) {
            if (!parse_easycache_triplet(easycache_it->get<std::string>())) {
                return false;
            }
            request.easycache.enabled = true;
        } else if (easycache_it->is_array()) {
            if (easycache_it->size() != 3) {
                error = "field 'easycache' array must contain exactly 3 values";
                return false;
            }
            float threshold = 0.0f;
            float start = 0.0f;
            float end = 0.0f;
            for (size_t i = 0; i < 3; ++i) {
                if (!(*easycache_it)[i].is_number_float() && !(*easycache_it)[i].is_number_integer()) {
                    error = "field 'easycache' array must contain numeric values";
                    return false;
                }
            }
            threshold = static_cast<float>((*easycache_it)[0].get<double>());
            start = static_cast<float>((*easycache_it)[1].get<double>());
            end = static_cast<float>((*easycache_it)[2].get<double>());
            if (!apply_easycache_values(threshold, start, end)) {
                return false;
            }
            request.easycache.enabled = true;
        } else {
            error = "field 'easycache' must be boolean, string, or array";
            return false;
        }
    }

    auto easycache_threshold_it = body.find("easycache_threshold");
    if (easycache_threshold_it != body.end()) {
        if (!easycache_threshold_it->is_number_float() && !easycache_threshold_it->is_number_integer()) {
            error = "field 'easycache_threshold' must be numeric";
            return false;
        }
        float value = static_cast<float>(easycache_threshold_it->get<double>());
        if (value < 0.0f) {
            error = "easycache threshold must be non-negative";
            return false;
        }
        request.easycache.reuse_threshold = value;
        request.easycache_provided = true;
        if (!easycache_flag_explicit) {
            request.easycache.enabled = true;
        }
    }

    auto easycache_start_it = body.find("easycache_start_percent");
    if (easycache_start_it != body.end()) {
        if (!easycache_start_it->is_number_float() && !easycache_start_it->is_number_integer()) {
            error = "field 'easycache_start_percent' must be numeric";
            return false;
        }
        float value = static_cast<float>(easycache_start_it->get<double>());
        if (value < 0.0f || value >= 1.0f) {
            error = "easycache_start_percent must be between 0.0 and 1.0";
            return false;
        }
        request.easycache.start_percent = value;
        request.easycache_provided = true;
        if (!easycache_flag_explicit) {
            request.easycache.enabled = true;
        }
    }

    auto easycache_end_it = body.find("easycache_end_percent");
    if (easycache_end_it != body.end()) {
        if (!easycache_end_it->is_number_float() && !easycache_end_it->is_number_integer()) {
            error = "field 'easycache_end_percent' must be numeric";
            return false;
        }
        float value = static_cast<float>(easycache_end_it->get<double>());
        if (value <= 0.0f || value > 1.0f) {
            error = "easycache_end_percent must be between 0.0 and 1.0";
            return false;
        }
        request.easycache.end_percent = value;
        request.easycache_provided = true;
        if (!easycache_flag_explicit) {
            request.easycache.enabled = true;
        }
    }

    if (request.easycache_provided && request.easycache.start_percent >= request.easycache.end_percent) {
        error = "easycache_start_percent must be less than easycache_end_percent";
        return false;
    }

    auto cfg_it = body.find("cfg_scale");
    if (cfg_it != body.end()) {
        if (!cfg_it->is_number_float() && !cfg_it->is_number_integer()) {
            error = "field 'cfg_scale' must be numeric";
            return false;
        }
        request.cfg_scale = static_cast<float>(cfg_it->get<double>());
    }

    auto img_cfg_it = body.find("img_cfg_scale");
    if (img_cfg_it != body.end()) {
        if (!img_cfg_it->is_number_float() && !img_cfg_it->is_number_integer()) {
            error = "field 'img_cfg_scale' must be numeric";
            return false;
        }
        request.img_cfg_scale = static_cast<float>(img_cfg_it->get<double>());
        request.has_img_cfg_scale = true;
    }

    auto guidance_it = body.find("guidance");
    if (guidance_it == body.end()) {
        guidance_it = body.find("distilled_guidance");
    }
    if (guidance_it != body.end()) {
        if (!guidance_it->is_number_float() && !guidance_it->is_number_integer()) {
            error = "field 'guidance' must be numeric";
            return false;
        }
        request.distilled_guidance = static_cast<float>(guidance_it->get<double>());
    }

    auto slg_scale_it = body.find("slg_scale");
    if (slg_scale_it != body.end()) {
        if (!slg_scale_it->is_number_float() && !slg_scale_it->is_number_integer()) {
            error = "field 'slg_scale' must be numeric";
            return false;
        }
        request.slg_scale = static_cast<float>(slg_scale_it->get<double>());
    }

    auto slg_start_it = body.find("skip_layer_start");
    if (slg_start_it == body.end()) {
        slg_start_it = body.find("slg_layer_start");
    }
    if (slg_start_it != body.end()) {
        if (!slg_start_it->is_number_float() && !slg_start_it->is_number_integer()) {
            error = "field 'skip_layer_start' must be numeric";
            return false;
        }
        request.slg_layer_start = static_cast<float>(slg_start_it->get<double>());
    }

    auto slg_end_it = body.find("skip_layer_end");
    if (slg_end_it == body.end()) {
        slg_end_it = body.find("slg_layer_end");
    }
    if (slg_end_it != body.end()) {
        if (!slg_end_it->is_number_float() && !slg_end_it->is_number_integer()) {
            error = "field 'skip_layer_end' must be numeric";
            return false;
        }
        request.slg_layer_end = static_cast<float>(slg_end_it->get<double>());
    }

    auto skip_layers_it = body.find("skip_layers");
    if (skip_layers_it != body.end()) {
        std::vector<int> layers;
        if (skip_layers_it->is_array()) {
            for (const auto& item : *skip_layers_it) {
                if (!item.is_number_integer()) {
                    error = "elements of 'skip_layers' must be integers";
                    return false;
                }
                layers.push_back(static_cast<int>(item.get<int64_t>()));
            }
        } else if (skip_layers_it->is_string()) {
            std::string value = trim_copy(skip_layers_it->get<std::string>());
            if (!value.empty()) {
                if (value.front() == '[' && value.back() == ']') {
                    value = value.substr(1, value.size() - 2);
                }
                std::vector<std::string> tokens = split_and_trim(value, ',');
                for (const auto& token : tokens) {
                    if (token.empty()) {
                        continue;
                    }
                    try {
                        layers.push_back(std::stoi(token));
                    } catch (const std::exception&) {
                        error = "failed to parse 'skip_layers'";
                        return false;
                    }
                }
            }
        } else {
            error = "field 'skip_layers' must be an array or string";
            return false;
        }
        if (!layers.empty()) {
            request.slg_layers = std::move(layers);
        }
    }

    auto eta_it = body.find("eta");
    if (eta_it != body.end()) {
        if (!eta_it->is_number_float() && !eta_it->is_number_integer()) {
            error = "field 'eta' must be numeric";
            return false;
        }
        request.eta = static_cast<float>(eta_it->get<double>());
        request.has_eta = true;
    }

    auto shift_it = body.find("timestep_shift");
    const char* shift_field_name = "timestep_shift";
    if (shift_it == body.end()) {
        shift_it = body.find("shifted_timestep");
        shift_field_name = "shifted_timestep";
    }
    if (shift_it != body.end()) {
        if (!shift_it->is_number_integer()) {
            error = std::string("field '") + shift_field_name + "' must be an integer";
            return false;
        }
        int value = static_cast<int>(shift_it->get<int64_t>());
        if (value < 0 || value > 1000) {
            error = std::string("field '") + shift_field_name + "' must be between 0 and 1000";
            return false;
        }
        request.shifted_timestep = value;
    }

    auto strength_it = body.find("strength");
    if (strength_it != body.end()) {
        if (!strength_it->is_number_float() && !strength_it->is_number_integer()) {
            error = "field 'strength' must be numeric";
            return false;
        }
        request.strength = static_cast<float>(strength_it->get<double>());
    }

    auto control_strength_it = body.find("control_strength");
    if (control_strength_it != body.end()) {
        if (!control_strength_it->is_number_float() && !control_strength_it->is_number_integer()) {
            error = "field 'control_strength' must be numeric";
            return false;
        }
        request.control_strength = static_cast<float>(control_strength_it->get<double>());
    }

    auto auto_resize_it = body.find("auto_resize_ref_image");
    if (auto_resize_it != body.end()) {
        if (!auto_resize_it->is_boolean()) {
            error = "field 'auto_resize_ref_image' must be a boolean";
            return false;
        }
        request.auto_resize_ref_image = auto_resize_it->get<bool>();
    }

    auto increase_it = body.find("increase_ref_index");
    if (increase_it != body.end()) {
        if (!increase_it->is_boolean()) {
            error = "field 'increase_ref_index' must be a boolean";
            return false;
        }
        request.increase_ref_index = increase_it->get<bool>();
    }

    auto canny_it = body.find("canny_preprocess");
    if (canny_it != body.end()) {
        if (!canny_it->is_boolean()) {
            error = "field 'canny_preprocess' must be a boolean";
            return false;
        }
        request.canny_preprocess = canny_it->get<bool>();
    }

    auto init_image_it = body.find("init_image_path");
    if (init_image_it == body.end()) {
        init_image_it = body.find("init_image");
    }
    if (init_image_it != body.end()) {
        if (!init_image_it->is_string()) {
            error = "field 'init_image_path' must be a string";
            return false;
        }
        request.init_image_path = trim_copy(init_image_it->get<std::string>());
    }

    auto mask_image_it = body.find("mask_image_path");
    if (mask_image_it == body.end()) {
        mask_image_it = body.find("mask_path");
    }
    if (mask_image_it != body.end()) {
        if (!mask_image_it->is_string()) {
            error = "field 'mask_image_path' must be a string";
            return false;
        }
        request.mask_image_path = trim_copy(mask_image_it->get<std::string>());
    }

    auto control_image_it = body.find("control_image_path");
    if (control_image_it == body.end()) {
        control_image_it = body.find("control_image");
    }
    if (control_image_it != body.end()) {
        if (!control_image_it->is_string()) {
            error = "field 'control_image_path' must be a string";
            return false;
        }
        request.control_image_path = trim_copy(control_image_it->get<std::string>());
    }

    auto ref_images_it = body.find("ref_image_paths");
    if (ref_images_it == body.end()) {
        ref_images_it = body.find("ref_images");
    }
    if (ref_images_it != body.end()) {
        request.ref_image_paths.clear();
        if (ref_images_it->is_array()) {
            for (const auto& item : *ref_images_it) {
                if (!item.is_string()) {
                    error = "elements of 'ref_image_paths' must be strings";
                    return false;
                }
                std::string path = trim_copy(item.get<std::string>());
                if (!path.empty()) {
                    request.ref_image_paths.push_back(path);
                }
            }
        } else if (ref_images_it->is_string()) {
            std::string value = trim_copy(ref_images_it->get<std::string>());
            if (!value.empty()) {
                auto paths = split_and_trim(value, ',');
                for (auto& path : paths) {
                    if (!path.empty()) {
                        request.ref_image_paths.push_back(path);
                    }
                }
            }
        } else {
            error = "field 'ref_image_paths' must be an array or string";
            return false;
        }
    }

    auto pm_dir_it = body.find("pm_id_images_dir");
    if (pm_dir_it != body.end()) {
        if (!pm_dir_it->is_string()) {
            error = "field 'pm_id_images_dir' must be a string";
            return false;
        }
        request.pm_id_images_dir = trim_copy(pm_dir_it->get<std::string>());
    }

    auto pm_embed_it = body.find("pm_id_embed_path");
    if (pm_embed_it != body.end()) {
        if (!pm_embed_it->is_string()) {
            error = "field 'pm_id_embed_path' must be a string";
            return false;
        }
        request.pm_id_embed_path = trim_copy(pm_embed_it->get<std::string>());
    }

    auto pm_style_it = body.find("pm_style_strength");
    if (pm_style_it != body.end()) {
        if (!pm_style_it->is_number_float() && !pm_style_it->is_number_integer()) {
            error = "field 'pm_style_strength' must be numeric";
            return false;
        }
        request.pm_style_strength = static_cast<float>(pm_style_it->get<double>());
    }

    auto batch_it = body.find("batch_count");
    if (batch_it != body.end()) {
        if (!batch_it->is_number_integer()) {
            error = "field 'batch_count' must be an integer";
            return false;
        }
        request.batch_count = static_cast<int>(batch_it->get<int64_t>());
    }
    if (request.batch_count <= 0) {
        error = "batch_count must be greater than 0";
        return false;
    }

    auto seed_it = body.find("seed");
    if (seed_it != body.end()) {
        if (!seed_it->is_number_integer()) {
            error = "field 'seed' must be an integer";
            return false;
        }
        request.seed = seed_it->get<int64_t>();
    }

    auto method_it = body.find("sample_method");
    if (method_it != body.end()) {
        if (!method_it->is_string()) {
            error = "field 'sample_method' must be a string";
            return false;
        }
        std::string value = to_lower_copy(method_it->get<std::string>());
        sample_method_t method = str_to_sample_method(value.c_str());
        if (method == SAMPLE_METHOD_COUNT) {
            error = "invalid sample_method value";
            return false;
        }
        request.sample_method = method;
        request.override_sample_method = true;
    }

    auto scheduler_it = body.find("scheduler");
    if (scheduler_it != body.end()) {
        if (!scheduler_it->is_string()) {
            error = "field 'scheduler' must be a string";
            return false;
        }
        std::string value = to_lower_copy(scheduler_it->get<std::string>());
        scheduler_t scheduler = str_to_schedule(value.c_str());
        if (scheduler == SCHEDULE_COUNT) {
            error = "invalid scheduler value";
            return false;
        }
        request.scheduler = scheduler;
        request.override_scheduler = true;
    }

    auto tiling_it = body.find("vae_tiling");
    if (tiling_it != body.end()) {
        if (!tiling_it->is_object()) {
            error = "field 'vae_tiling' must be an object";
            return false;
        }
        sd_tiling_params_t tiling = request.vae_tiling_params;
        bool modified = false;
        const json& tiling_obj = *tiling_it;

        auto tiling_enabled = tiling_obj.find("enabled");
        if (tiling_enabled != tiling_obj.end()) {
            if (!tiling_enabled->is_boolean()) {
                error = "field 'vae_tiling.enabled' must be a boolean";
                return false;
            }
            tiling.enabled = tiling_enabled->get<bool>();
            modified = true;
        }

        auto tiling_x = tiling_obj.find("tile_size_x");
        if (tiling_x != tiling_obj.end()) {
            if (!tiling_x->is_number_integer()) {
                error = "field 'vae_tiling.tile_size_x' must be an integer";
                return false;
            }
            tiling.tile_size_x = static_cast<int>(tiling_x->get<int64_t>());
            modified = true;
        }

        auto tiling_y = tiling_obj.find("tile_size_y");
        if (tiling_y != tiling_obj.end()) {
            if (!tiling_y->is_number_integer()) {
                error = "field 'vae_tiling.tile_size_y' must be an integer";
                return false;
            }
            tiling.tile_size_y = static_cast<int>(tiling_y->get<int64_t>());
            modified = true;
        }

        auto overlap = tiling_obj.find("target_overlap");
        if (overlap != tiling_obj.end()) {
            if (!overlap->is_number_float() && !overlap->is_number_integer()) {
                error = "field 'vae_tiling.target_overlap' must be numeric";
                return false;
            }
            tiling.target_overlap = static_cast<float>(overlap->get<double>());
            modified = true;
        }

        auto rel_x = tiling_obj.find("rel_size_x");
        if (rel_x != tiling_obj.end()) {
            if (!rel_x->is_number_float() && !rel_x->is_number_integer()) {
                error = "field 'vae_tiling.rel_size_x' must be numeric";
                return false;
            }
            tiling.rel_size_x = static_cast<float>(rel_x->get<double>());
            modified = true;
        }

        auto rel_y = tiling_obj.find("rel_size_y");
        if (rel_y != tiling_obj.end()) {
            if (!rel_y->is_number_float() && !rel_y->is_number_integer()) {
                error = "field 'vae_tiling.rel_size_y' must be numeric";
                return false;
            }
            tiling.rel_size_y = static_cast<float>(rel_y->get<double>());
            modified = true;
        }

        if (modified) {
            request.vae_tiling_params = tiling;
            request.has_vae_tiling_override = true;
        }
    }

    return true;
}

bool prepare_generation_inputs(GenerationRequest& request, std::string& error) {
    request.has_init_image = false;
    request.has_mask_image = false;
    request.has_control_image = false;
    request.ref_images.clear();
    request.pm_id_images.clear();

    if (!request.init_image_path.empty()) {
        OwnedImage init;
        if (!load_image_file(request.init_image_path, request.width, request.height, 3, init, error)) {
            return false;
        }
        request.init_image = std::move(init);
        request.has_init_image = true;
    }

    if (!request.mask_image_path.empty()) {
        OwnedImage mask;
        if (!load_image_file(request.mask_image_path, request.width, request.height, 1, mask, error)) {
            return false;
        }
        request.mask_image = std::move(mask);
        request.has_mask_image = true;
    }

    if (!request.control_image_path.empty()) {
        OwnedImage control;
        if (!load_image_file(request.control_image_path, request.width, request.height, 3, control, error)) {
            return false;
        }
        request.control_image = std::move(control);
        request.has_control_image = true;
        if (request.canny_preprocess) {
            sd_image_t control_view = request.control_image.as_sd_image();
            if (!preprocess_canny(control_view, 0.08f, 0.08f, 0.8f, 1.0f, false)) {
                error = "failed to run canny preprocessor on control image";
                return false;
            }
        }
    }

    if (!request.ref_image_paths.empty()) {
        request.ref_images.reserve(request.ref_image_paths.size());
        for (const auto& path : request.ref_image_paths) {
            OwnedImage reference;
            if (!load_image_file(path, 0, 0, 3, reference, error)) {
                return false;
            }
            request.ref_images.push_back(std::move(reference));
        }
    }

    if (!request.pm_id_images_dir.empty()) {
        if (!load_images_from_directory(request.pm_id_images_dir, 0, 0, 3, 0, request.pm_id_images, error)) {
            return false;
        }
    }

    return true;
}

std::optional<double> parse_duration_token_ms(const std::string& text) {
    std::size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    std::size_t start = i;
    bool has_digit = false;
    while (i < text.size() && (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '.')) {
        has_digit = true;
        ++i;
    }
    if (!has_digit) {
        return std::nullopt;
    }
    double value = std::stod(text.substr(start, i - start));
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i >= text.size()) {
        return std::nullopt;
    }
    if (text.compare(i, 2, "ms") == 0) {
        return value;
    }
    if (text[i] == 's') {
        return value * 1000.0;
    }
    return std::nullopt;
}

std::optional<double> extract_duration_ms(const std::string& message) {
    const std::string taking_marker = "taking ";
    auto taking_pos = message.find(taking_marker);
    if (taking_pos != std::string::npos) {
        auto value = parse_duration_token_ms(message.substr(taking_pos + taking_marker.size()));
        if (value) {
            return value;
        }
    }
    const std::string completed_in_marker = "completed in ";
    auto completed_in_pos = message.find(completed_in_marker);
    if (completed_in_pos != std::string::npos) {
        auto value = parse_duration_token_ms(message.substr(completed_in_pos + completed_in_marker.size()));
        if (value) {
            return value;
        }
    }
    return std::nullopt;
}

std::map<std::string, double> extract_duration_breakdown_ms(const std::string& message) {
    std::map<std::string, double> breakdown;
    auto open = message.find('(');
    auto close = message.find(')', open);
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return breakdown;
    }
    std::string inside = message.substr(open + 1, close - open - 1);
    std::stringstream ss(inside);
    std::string part;
    while (std::getline(ss, part, ',')) {
        part = trim_copy(part);
        if (part.empty()) {
            continue;
        }
        auto colon = part.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = trim_copy(part.substr(0, colon));
        std::string value_text = trim_copy(part.substr(colon + 1));
        auto value = parse_duration_token_ms(value_text);
        if (value) {
            breakdown[key] = *value;
        }
    }
    return breakdown;
}

std::string normalize_path_for_compare(const std::string& path) {
    std::string normalized = to_lower_copy(path);
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

std::string path_stem(const std::string& path) {
    std::size_t pos = path.find_last_of("/\\");
    std::string basename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    std::size_t dot = basename.find_last_of('.');
    if (dot == std::string::npos) {
        return basename;
    }
    return basename.substr(0, dot);
}

json breakdown_to_json(const std::map<std::string, double>& breakdown) {
    json result = json::object();
    for (const auto& kv : breakdown) {
        result[kv.first + "_ms"] = kv.second;
    }
    return result;
}

json make_telemetry(const LogCollector& collector,
                    const GenerationRequest& request,
                    const CtxConfig& config,
                    int64_t elapsed_ms,
                    int64_t effective_seed) {
    struct LoraData {
        std::string primary_path;
        std::vector<std::string> load_paths;
        std::vector<double> load_ms;
        std::vector<std::map<std::string, double>> breakdowns;
        double applied_ms = 0.0;
        bool applied_recorded = false;
    };

    struct EmbeddingData {
        std::string primary_path;
        std::vector<std::string> load_paths;
        std::vector<double> load_ms;
        std::vector<std::map<std::string, double>> breakdowns;
        int custom_embedding_count = -1;
    };

    std::map<std::string, LoraData> lora_map;
    std::vector<std::string> lora_order;
    std::map<std::string, EmbeddingData> embedding_map;
    std::vector<std::string> embedding_order;
    std::set<std::string> embedding_paths;
    std::set<std::string> embedding_compare_paths;
    struct TensorSourceInfo {
        std::string path;
        std::string component;
    };
    std::deque<TensorSourceInfo> pending_tensor_sources;
    std::map<std::string, std::string> component_by_path;
    std::optional<json> model_summary;

    std::vector<double> condition_graph_ms;
    std::vector<double> learned_condition_ms;
    std::vector<double> sampling_ms;
    struct VaeTiming {
        int latent_index = 0;
        double duration_ms = 0.0;
    };
    std::vector<VaeTiming> vae_timings;
    std::optional<double> generate_image_ms;
    json component_loads = json::array();
    auto record_component_load = [&](const std::string& component_name,
                                     const std::string& component_path,
                                     const std::optional<double>& duration,
                                     const std::map<std::string, double>& breakdown) {
        json entry = json::object();
        entry["component"] = component_name;
        if (!component_path.empty()) {
            entry["path"] = component_path;
        }
        if (duration) {
            entry["duration_ms"] = *duration;
        }
        json breakdown_json = breakdown_to_json(breakdown);
        if (!breakdown_json.empty()) {
            entry["breakdown_ms"] = breakdown_json;
        }
        component_loads.push_back(std::move(entry));
    };

    struct SpanRecord {
        std::string span_id;
        std::string parent_span_id;
        std::string name;
        std::string kind;
        double duration_ms = 0.0;
        json attributes;
        std::vector<SpanRecord> subspans;
    };

    const std::string root_span_id = "span-0";
    int span_counter = 1;
    SpanRecord root_span;
    root_span.span_id = root_span_id;
    root_span.kind = "INTERNAL";
    root_span.duration_ms = static_cast<double>(elapsed_ms);

    auto next_span_id = [&]() {
        return "span-" + std::to_string(span_counter++);
    };

    auto add_subspan = [&](SpanRecord& parent, const std::string& name, const std::optional<double>& duration, json attributes) -> SpanRecord* {
        if (!duration) {
            return nullptr;
        }
        SpanRecord record;
        record.span_id = next_span_id();
        record.parent_span_id = parent.span_id;
        record.name = name;
        record.kind = "INTERNAL";
        record.duration_ms = *duration;
        record.attributes = std::move(attributes);
        parent.subspans.push_back(std::move(record));
        return &parent.subspans.back();
    };

    const std::string model_compare_path = config.model_path.empty() ? std::string() : normalize_path_for_compare(config.model_path);
    const std::vector<std::pair<std::string, std::string>> component_markers = {
        {"loading model from '", "model.core"},
        {"loading diffusion model from '", "model.diffusion"},
        {"loading high noise diffusion model from '", "model.diffusion_high_noise"},
        {"loading clip_l from '", "encoder.clip_l"},
        {"loading clip_g from '", "encoder.clip_g"},
        {"loading clip_vision from '", "encoder.clip_vision"},
        {"loading t5xxl from '", "encoder.t5xxl"},
        {"loading qwen2vl from '", "encoder.qwen2vl"},
        {"loading qwen2vl vision from '", "encoder.qwen2vl_vision"},
        {"loading vae from '", "vae.decoder"},
        {"loading stacked ID embedding (PHOTOMAKER) model file from '", "photomaker.embedding"}
    };

    for (const auto& entry : collector.entries) {
        const std::string& message = entry.message;

        bool component_logged = false;
        for (const auto& marker : component_markers) {
            auto pos = message.find(marker.first);
            if (pos != std::string::npos) {
                std::size_t path_start = pos + marker.first.size();
                std::size_t path_end = message.find("'", path_start);
                if (path_end != std::string::npos && path_end > path_start) {
                    std::string path = message.substr(path_start, path_end - path_start);
                    std::string normalized = normalize_path_for_compare(path);
                    if (!normalized.empty()) {
                        component_by_path[normalized] = marker.second;
                    }
                    if (!path.empty()) {
                        component_by_path[path] = marker.second;
                    }
                }
                component_logged = true;
                break;
            }
        }
        if (component_logged) {
            continue;
        }

        const std::string embed_tag = "<embed:";
        std::size_t embed_search_pos = 0;
        while (true) {
            std::size_t tag_pos = message.find(embed_tag, embed_search_pos);
            if (tag_pos == std::string::npos) {
                break;
            }
            std::size_t path_start = tag_pos + embed_tag.size();
            std::size_t path_end = message.find(">", path_start);
            if (path_end == std::string::npos) {
                break;
            }
            std::string embed_token = trim_copy(message.substr(path_start, path_end - path_start));
            if (!embed_token.empty()) {
                std::size_t weight_sep = embed_token.find(":");
                if (weight_sep != std::string::npos) {
                    std::string tail = embed_token.substr(weight_sep + 1);
                    if (tail.find('/') == std::string::npos && tail.find('\\') == std::string::npos) {
                        embed_token = embed_token.substr(0, weight_sep);
                    }
                }
                bool looks_like_path = embed_token.find('/') != std::string::npos || embed_token.find('\\') != std::string::npos;
                if (looks_like_path) {
                    std::string normalized_embed = normalize_path_for_compare(embed_token);
                    if (!normalized_embed.empty()) {
                        embedding_compare_paths.insert(normalized_embed);
                    }
                    embedding_paths.insert(embed_token);
                }
            }
            embed_search_pos = path_end + 1;
        }

        const std::string loading_from_marker = "loading tensors from ";
        auto from_pos = message.find(loading_from_marker);
        if (from_pos != std::string::npos) {
            std::string path = trim_copy(message.substr(from_pos + loading_from_marker.size()));
            if (!path.empty()) {
                TensorSourceInfo info;
                info.path = path;
                std::string normalized = normalize_path_for_compare(path);
                auto comp_it = component_by_path.find(normalized);
                if (comp_it != component_by_path.end()) {
                    info.component = comp_it->second;
                } else {
                    auto raw_it = component_by_path.find(path);
                    if (raw_it != component_by_path.end()) {
                        info.component = raw_it->second;
                    }
                }
                pending_tensor_sources.push_back(std::move(info));
            }
            continue;
        }

        if (message.find("loading tensors completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            std::string path;
            std::string component;
            if (!pending_tensor_sources.empty()) {
                TensorSourceInfo info = std::move(pending_tensor_sources.front());
                pending_tensor_sources.pop_front();
                path = std::move(info.path);
                component = std::move(info.component);
            }
            auto breakdown = extract_duration_breakdown_ms(message);

            std::string normalized_path;
            bool is_model = false;
            bool is_lora = false;
            bool is_embedding = false;
            if (!path.empty()) {
                normalized_path = normalize_path_for_compare(path);
                if (!model_compare_path.empty() && normalized_path == model_compare_path) {
                    is_model = true;
                }
                if (component.empty()) {
                    auto comp_it = component_by_path.find(normalized_path);
                    if (comp_it != component_by_path.end()) {
                        component = comp_it->second;
                    } else {
                        auto raw_it = component_by_path.find(path);
                        if (raw_it != component_by_path.end()) {
                            component = raw_it->second;
                        }
                    }
                }
                if (!normalized_path.empty() && embedding_compare_paths.find(normalized_path) != embedding_compare_paths.end()) {
                    is_embedding = true;
                }
                if (!is_embedding && embedding_paths.find(path) != embedding_paths.end()) {
                    is_embedding = true;
                }
                std::string lowered_path = to_lower_copy(path);
                if (lowered_path.find("lora") != std::string::npos) {
                    is_lora = true;
                }
                if (is_embedding) {
                    is_lora = false;
                }
            }

            json attributes = json::object();
            if (duration) {
                attributes["duration.ms"] = *duration;
            }
            if (!path.empty()) {
                attributes["gen_ai.artifact.path"] = path;
            }

            std::string span_name = "gen_ai.artifact.load";
            if (is_lora) {
                std::string lora_id = path_stem(path);
                auto it = lora_map.find(lora_id);
                if (it == lora_map.end()) {
                    lora_order.push_back(lora_id);
                }
                LoraData& data = lora_map[lora_id];
                if (data.primary_path.empty()) {
                    data.primary_path = path;
                }
                data.load_paths.push_back(path);
                data.load_ms.push_back(duration.value_or(0.0));
                data.breakdowns.push_back(breakdown);

                attributes["gen_ai.artifact.id"] = lora_id;
                attributes["gen_ai.artifact.type"] = "lora";
                attributes["gen_ai.operation.stage"] = "lora.load";
                attributes["sdcpp.lora.phase_index"] = static_cast<int>(data.load_ms.size()) - 1;
                if (data.load_ms.size() == 1) {
                    attributes["sdcpp.lora.phase"] = "prefetch";
                } else if (data.load_ms.size() == 2) {
                    attributes["sdcpp.lora.phase"] = "load";
                } else {
                    attributes["sdcpp.lora.phase"] = "load_extra";
                }
                span_name = "gen_ai.artifact.lora.load";
            } else if (is_embedding) {
                std::string embedding_id = path_stem(path);
                auto it = embedding_map.find(embedding_id);
                if (it == embedding_map.end()) {
                    embedding_order.push_back(embedding_id);
                }
                EmbeddingData& data = embedding_map[embedding_id];
                if (data.primary_path.empty()) {
                    data.primary_path = path;
                }
                data.load_paths.push_back(path);
                data.load_ms.push_back(duration.value_or(0.0));
                data.breakdowns.push_back(breakdown);
                if (!normalized_path.empty()) {
                    embedding_compare_paths.insert(normalized_path);
                }
                if (!path.empty()) {
                    embedding_paths.insert(path);
                }

                attributes["gen_ai.artifact.id"] = embedding_id;
                attributes["gen_ai.artifact.type"] = "embedding";
                attributes["gen_ai.operation.stage"] = "embedding.load";
                attributes["sdcpp.embedding.phase_index"] = static_cast<int>(data.load_ms.size()) - 1;
                attributes["sdcpp.embedding.phase"] = "load";
                span_name = "gen_ai.artifact.embedding.load";
            } else if (is_model) {
                if (component.empty()) {
                    component = "model.core";
                }
                attributes["gen_ai.artifact.type"] = "model";
                if (!component.empty()) {
                    attributes["sdcpp.artifact.component"] = component;
                }
                attributes["gen_ai.operation.stage"] = "model.load";
                span_name = "gen_ai.model.load";
                if (duration) {
                    json model = json::object();
                    model["path"] = path;
                    model["duration_ms"] = *duration;
                    json breakdown_json = breakdown_to_json(breakdown);
                    if (!breakdown_json.empty()) {
                        model["breakdown_ms"] = breakdown_json;
                    }
                    model_summary = model;
                }
                if (!component.empty()) {
                    record_component_load(component, path, duration, breakdown);
                }
            } else if (!component.empty()) {
                attributes["gen_ai.artifact.type"] = "component";
                attributes["gen_ai.operation.stage"] = "component.load";
                attributes["sdcpp.artifact.component"] = component;
                span_name = "gen_ai.component.load";
                record_component_load(component, path, duration, breakdown);
            } else {
                attributes["gen_ai.operation.stage"] = "artifact.load";
            }

            json breakdown_json = breakdown_to_json(breakdown);
            if (!breakdown_json.empty()) {
                attributes["sdcpp.stage.breakdown_ms"] = breakdown_json;
            }

            add_subspan(root_span, span_name, duration, std::move(attributes));
            continue;
        }
        const std::string lora_marker = "lora '";
        auto lora_pos = message.find(lora_marker);
        if (lora_pos != std::string::npos && message.find(" applied") != std::string::npos) {
            std::size_t id_start = lora_pos + lora_marker.size();
            std::size_t id_end = message.find("'", id_start);
            if (id_end != std::string::npos) {
                std::string lora_id = message.substr(id_start, id_end - id_start);
                auto duration = extract_duration_ms(message);
                if (duration) {
                    auto it = lora_map.find(lora_id);
                    if (it == lora_map.end()) {
                        lora_order.push_back(lora_id);
                    }
                    LoraData& data = lora_map[lora_id];
                    data.applied_ms = *duration;
                    data.applied_recorded = true;

                    double load_total = std::accumulate(data.load_ms.begin(), data.load_ms.end(), 0.0);
                    double compute_ms = *duration - load_total;

                    json attributes = json::object();
                    attributes["gen_ai.artifact.id"] = lora_id;
                    attributes["gen_ai.artifact.type"] = "lora";
                    attributes["duration.ms"] = *duration;
                    attributes["gen_ai.operation.stage"] = "lora.apply";
                    attributes["sdcpp.lora.load_total_ms"] = load_total;
                    attributes["sdcpp.lora.compute_ms"] = compute_ms;
                    if (!data.primary_path.empty()) {
                        attributes["gen_ai.artifact.path"] = data.primary_path;
                    }

                    add_subspan(root_span, "gen_ai.artifact.lora.apply", duration, std::move(attributes));
                }
            }
            continue;
        }

        const std::string embedding_marker = "embedding '";
        auto embedding_pos = message.find(embedding_marker);
        if (embedding_pos != std::string::npos && message.find(" applied") != std::string::npos) {
            std::size_t path_start = embedding_pos + embedding_marker.size();
            std::size_t path_end = message.find("'", path_start);
            if (path_end != std::string::npos) {
                std::string embedding_path = message.substr(path_start, path_end - path_start);
                std::string normalized_embedding = normalize_path_for_compare(embedding_path);
                if (!embedding_path.empty()) {
                    embedding_paths.insert(embedding_path);
                }
                if (!normalized_embedding.empty()) {
                    embedding_compare_paths.insert(normalized_embedding);
                }
                std::string embedding_id = path_stem(embedding_path);
                auto it = embedding_map.find(embedding_id);
                if (it == embedding_map.end()) {
                    embedding_order.push_back(embedding_id);
                }
                EmbeddingData& data = embedding_map[embedding_id];
                if (data.primary_path.empty()) {
                    data.primary_path = embedding_path;
                }
                const std::string custom_marker = "custom embeddings:";
                auto custom_pos = message.find(custom_marker, path_end);
                if (custom_pos != std::string::npos) {
                    std::string count_text = trim_copy(message.substr(custom_pos + custom_marker.size()));
                    try {
                        data.custom_embedding_count = std::stoi(count_text);
                    } catch (const std::exception&) {
                        // ignore parse errors
                    }
                }
            }
            continue;
        }

        if (message.find("computing condition graph completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            if (duration) {
                condition_graph_ms.push_back(*duration);
                json attributes = json::object();
                attributes["duration.ms"] = *duration;
                attributes["gen_ai.operation.stage"] = "conditioning.graph";
                add_subspan(root_span, "gen_ai.conditioner.graph", duration, std::move(attributes));
            }
            continue;
        }

        if (message.find("get_learned_condition completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            if (duration) {
                learned_condition_ms.push_back(*duration);
                json attributes = json::object();
                attributes["duration.ms"] = *duration;
                attributes["gen_ai.operation.stage"] = "conditioning.get_learned_condition";
                add_subspan(root_span, "gen_ai.conditioner.get_learned_condition", duration, std::move(attributes));
            }
            continue;
        }

        if (message.find("sampling completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            if (duration) {
                sampling_ms.push_back(*duration);
                json attributes = json::object();
                attributes["duration.ms"] = *duration;
                attributes["gen_ai.operation.stage"] = "sampling";
                add_subspan(root_span, "gen_ai.sampling", duration, std::move(attributes));
            }
            continue;
        }

        auto latent_pos = message.find("latent ");
        auto decoded_pos = message.find(" decoded");
        if (latent_pos != std::string::npos && decoded_pos != std::string::npos && latent_pos < decoded_pos) {
            const std::size_t latent_prefix_len = 7;
            std::size_t idx_start = latent_pos + latent_prefix_len;
            std::size_t idx_end = idx_start;
            while (idx_end < message.size() && std::isdigit(static_cast<unsigned char>(message[idx_end]))) {
                ++idx_end;
            }
            if (idx_end > idx_start) {
                int latent_index = std::stoi(message.substr(idx_start, idx_end - idx_start));
                auto duration = extract_duration_ms(message);
                if (duration) {
                    vae_timings.push_back({latent_index, *duration});
                    json attributes = json::object();
                    attributes["duration.ms"] = *duration;
                    attributes["gen_ai.operation.stage"] = "vae.decode";
                    attributes["sdcpp.latent.index"] = latent_index;
                    add_subspan(root_span, "gen_ai.vae.decode", duration, std::move(attributes));
                }
            }
            continue;
        }

        if (message.find("generate_image completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            if (duration) {
                generate_image_ms = *duration;
                json attributes = json::object();
                attributes["duration.ms"] = *duration;
                attributes["gen_ai.operation.stage"] = "inference.complete";
                add_subspan(root_span, "gen_ai.generate_image", duration, std::move(attributes));
            }
            continue;
        }
    }

    json summary = json::object();
    summary["elapsed_ms"] = elapsed_ms;
    if (model_summary) {
        summary["model_load"] = *model_summary;
    }

    json lora_summary = json::array();
    for (const auto& id : lora_order) {
        auto it = lora_map.find(id);
        if (it == lora_map.end()) {
            continue;
        }
        const LoraData& data = it->second;
        json entry = json::object();
        entry["id"] = id;
        if (!data.primary_path.empty()) {
            entry["path"] = data.primary_path;
        }
        if (!data.load_ms.empty()) {
            double load_total = std::accumulate(data.load_ms.begin(), data.load_ms.end(), 0.0);
            entry["load_total_ms"] = load_total;
            json phases = json::array();
            for (std::size_t i = 0; i < data.load_ms.size(); ++i) {
                json phase = json::object();
                phase["duration_ms"] = data.load_ms[i];
                phase["phase_index"] = static_cast<int>(i);
                if (i == 0) {
                    phase["phase"] = "prefetch";
                } else if (i == 1) {
                    phase["phase"] = "load";
                } else {
                    phase["phase"] = "load_extra";
                }
                if (i < data.load_paths.size() && !data.load_paths[i].empty()) {
                    phase["path"] = data.load_paths[i];
                }
                json breakdown_json = breakdown_to_json(data.breakdowns[i]);
                if (!breakdown_json.empty()) {
                    phase["breakdown_ms"] = breakdown_json;
                }
                phases.push_back(std::move(phase));
            }
            entry["load_phases"] = std::move(phases);
            if (data.applied_recorded) {
                double compute_ms = data.applied_ms - load_total;
                entry["apply_ms"] = data.applied_ms;
                entry["compute_ms"] = compute_ms;
            }
        } else {
            entry["load_total_ms"] = 0.0;
            if (data.applied_recorded) {
                entry["apply_ms"] = data.applied_ms;
                entry["compute_ms"] = data.applied_ms;
            }
        }
        lora_summary.push_back(std::move(entry));
    }
    if (!lora_summary.empty()) {
        summary["loras"] = std::move(lora_summary);
    }

    json embedding_summary = json::array();
    for (const auto& id : embedding_order) {
        auto it = embedding_map.find(id);
        if (it == embedding_map.end()) {
            continue;
        }
        const EmbeddingData& data = it->second;
        json entry = json::object();
        entry["id"] = id;
        if (!data.primary_path.empty()) {
            entry["path"] = data.primary_path;
        }
        if (!data.load_ms.empty()) {
            double load_total = std::accumulate(data.load_ms.begin(), data.load_ms.end(), 0.0);
            entry["load_total_ms"] = load_total;
            json phases = json::array();
            for (std::size_t i = 0; i < data.load_ms.size(); ++i) {
                json phase = json::object();
                phase["duration_ms"] = data.load_ms[i];
                phase["phase_index"] = static_cast<int>(i);
                phase["phase"] = "load";
                if (i < data.load_paths.size() && !data.load_paths[i].empty()) {
                    phase["path"] = data.load_paths[i];
                }
                json breakdown_json = breakdown_to_json(data.breakdowns[i]);
                if (!breakdown_json.empty()) {
                    phase["breakdown_ms"] = breakdown_json;
                }
                phases.push_back(std::move(phase));
            }
            entry["load_phases"] = std::move(phases);
        } else {
            entry["load_total_ms"] = 0.0;
        }
        if (data.custom_embedding_count >= 0) {
            entry["custom_embeddings"] = data.custom_embedding_count;
        }
        embedding_summary.push_back(std::move(entry));
    }
    if (!embedding_summary.empty()) {
        summary["embeddings"] = std::move(embedding_summary);
    }

    auto vector_to_json_array = [](const std::vector<double>& values) {
        json arr = json::array();
        for (double value : values) {
            arr.push_back(value);
        }
        return arr;
    };

    if (!condition_graph_ms.empty() || !learned_condition_ms.empty()) {
        json conditioning = json::object();
        if (!condition_graph_ms.empty()) {
            json compute_graph = json::object();
            compute_graph["durations_ms"] = vector_to_json_array(condition_graph_ms);
            compute_graph["total_ms"] = std::accumulate(condition_graph_ms.begin(), condition_graph_ms.end(), 0.0);
            conditioning["compute_graph"] = std::move(compute_graph);
        }
        if (!learned_condition_ms.empty()) {
            json learned = json::object();
            learned["durations_ms"] = vector_to_json_array(learned_condition_ms);
            learned["total_ms"] = std::accumulate(learned_condition_ms.begin(), learned_condition_ms.end(), 0.0);
            conditioning["get_learned_condition"] = std::move(learned);
        }
        summary["conditioning"] = std::move(conditioning);
    }

    if (!sampling_ms.empty()) {
        json sampling = json::object();
        sampling["durations_ms"] = vector_to_json_array(sampling_ms);
        sampling["total_ms"] = std::accumulate(sampling_ms.begin(), sampling_ms.end(), 0.0);
        summary["sampling"] = std::move(sampling);
    }

    if (!vae_timings.empty()) {
        json latents = json::array();
        double total = 0.0;
        for (const auto& timing : vae_timings) {
            total += timing.duration_ms;
            json latent_entry = json::object();
            latent_entry["latent_index"] = timing.latent_index;
            latent_entry["duration_ms"] = timing.duration_ms;
            latents.push_back(std::move(latent_entry));
        }
        json vae = json::object();
        vae["latents"] = std::move(latents);
        vae["total_ms"] = total;
        summary["vae_decode"] = std::move(vae);
    }

    if (generate_image_ms) {
        json generate = json::object();
        generate["duration_ms"] = *generate_image_ms;
        summary["generate_image"] = std::move(generate);
    }
    if (!component_loads.empty()) {
        summary["component_loads"] = component_loads;
    }

    json span_attributes = json::object();
    span_attributes["gen_ai.operation.name"] = "image_generation";
    span_attributes["gen_ai.output.type"] = "image";
    span_attributes["gen_ai.provider.name"] = "stable_diffusion_cpp";
    if (!config.model_path.empty()) {
        span_attributes["gen_ai.request.model"] = path_stem(config.model_path);
        span_attributes["sdcpp.model.path"] = config.model_path;
    }
    if (effective_seed >= 0) {
        span_attributes["gen_ai.request.seed"] = effective_seed;
    }
    if (request.seed >= 0 && request.seed != effective_seed) {
        span_attributes["sdcpp.request.seed_requested"] = request.seed;
    }
    if (request.seed < 0) {
        span_attributes["sdcpp.request.random_seed"] = true;
    }
    span_attributes["sdcpp.request.batch_count"] = request.batch_count;
    span_attributes["sdcpp.request.width"] = request.width;
    span_attributes["sdcpp.request.height"] = request.height;
    span_attributes["sdcpp.request.sample_steps"] = request.sample_steps;
    span_attributes["sdcpp.request.cfg_scale"] = request.cfg_scale;
    span_attributes["sdcpp.request.clip_skip"] = request.clip_skip;
    if (request.has_eta) {
        span_attributes["sdcpp.request.eta"] = request.eta;
    }
    span_attributes["sdcpp.request.shifted_timestep"] = request.shifted_timestep;
    span_attributes["sdcpp.request.strength"] = request.strength;
    span_attributes["sdcpp.request.control_strength"] = request.control_strength;
    span_attributes["sdcpp.request.auto_resize_ref_image"] = request.auto_resize_ref_image;
    span_attributes["sdcpp.request.increase_ref_index"] = request.increase_ref_index;
    span_attributes["sdcpp.request.has_init_image"] = request.has_init_image;
    span_attributes["sdcpp.request.has_mask_image"] = request.has_mask_image;
    span_attributes["sdcpp.request.has_control_image"] = request.has_control_image;
    if (!request.init_image_path.empty()) {
        span_attributes["sdcpp.request.init_image_path"] = request.init_image_path;
    }
    if (!request.mask_image_path.empty()) {
        span_attributes["sdcpp.request.mask_image_path"] = request.mask_image_path;
    }
    if (!request.control_image_path.empty()) {
        span_attributes["sdcpp.request.control_image_path"] = request.control_image_path;
    }
    if (!request.ref_image_paths.empty()) {
        span_attributes["sdcpp.request.ref_image_count"] = static_cast<int>(request.ref_image_paths.size());
    }
    if (!request.pm_id_images_dir.empty()) {
        span_attributes["sdcpp.request.pm_id_images_dir"] = request.pm_id_images_dir;
    }
    if (!request.pm_id_embed_path.empty()) {
        span_attributes["sdcpp.request.pm_id_embed_path"] = request.pm_id_embed_path;
    }
    span_attributes["sdcpp.request.pm_style_strength"] = request.pm_style_strength;
    if (request.has_img_cfg_scale) {
        span_attributes["sdcpp.request.img_cfg_scale"] = request.img_cfg_scale;
    }
    if (!request.override_sample_method && request.sample_method != SAMPLE_METHOD_DEFAULT) {
        span_attributes["sdcpp.request.sample_method"] = sd_sample_method_name(request.sample_method);
    } else if (request.override_sample_method) {
        span_attributes["sdcpp.request.sample_method"] = sd_sample_method_name(request.sample_method);
    }
    if (request.override_scheduler) {
        span_attributes["sdcpp.request.scheduler"] = sd_schedule_name(request.scheduler);
    }
    span_attributes["sdcpp.request.distilled_guidance"] = request.distilled_guidance;
    span_attributes["sdcpp.request.slg_scale"] = request.slg_scale;
    span_attributes["sdcpp.request.slg_layer_start"] = request.slg_layer_start;
    span_attributes["sdcpp.request.slg_layer_end"] = request.slg_layer_end;
    span_attributes["gen_ai.response.latency_ms"] = elapsed_ms;
    span_attributes["sdcpp.context.diffusion_flash_attn"] = config.diffusion_flash_attn;
    span_attributes["sdcpp.context.diffusion_conv_direct"] = config.diffusion_conv_direct;
    span_attributes["sdcpp.context.vae_conv_direct"] = config.vae_conv_direct;
    span_attributes["sdcpp.context.force_sdxl_vae_conv_scale"] = config.force_sdxl_vae_conv_scale;
    span_attributes["sdcpp.context.offload_params_to_cpu"] = config.offload_params_to_cpu;
    span_attributes["sdcpp.context.keep_clip_on_cpu"] = config.keep_clip_on_cpu;
    span_attributes["sdcpp.context.keep_control_net_on_cpu"] = config.keep_control_net_on_cpu;
    span_attributes["sdcpp.context.keep_vae_on_cpu"] = config.keep_vae_on_cpu;
    span_attributes["sdcpp.context.rng_type"] = sd_rng_type_name(config.rng_type);
    if (config.wtype != SD_TYPE_COUNT) {
        span_attributes["sdcpp.context.weight_type"] = sd_type_name(config.wtype);
    }
    if (!std::isfinite(config.flow_shift)) {
        span_attributes["sdcpp.context.flow_shift"] = "auto";
    } else {
        span_attributes["sdcpp.context.flow_shift"] = config.flow_shift;
    }
    span_attributes["sdcpp.context.chroma_use_dit_mask"] = config.chroma_use_dit_mask;
    span_attributes["sdcpp.context.chroma_use_t5_mask"] = config.chroma_use_t5_mask;
    span_attributes["sdcpp.context.chroma_t5_mask_pad"] = config.chroma_t5_mask_pad;
    span_attributes["sdcpp.context.prediction"] = sd_prediction_name(config.prediction);
    if (!config.clip_l_path.empty()) {
        span_attributes["sdcpp.context.clip_l_path"] = config.clip_l_path;
    }
    if (!config.clip_g_path.empty()) {
        span_attributes["sdcpp.context.clip_g_path"] = config.clip_g_path;
    }
    if (!config.clip_vision_path.empty()) {
        span_attributes["sdcpp.context.clip_vision_path"] = config.clip_vision_path;
    }
    if (!config.t5xxl_path.empty()) {
        span_attributes["sdcpp.context.t5xxl_path"] = config.t5xxl_path;
    }
    if (!config.qwen2vl_path.empty()) {
        span_attributes["sdcpp.context.qwen2vl_path"] = config.qwen2vl_path;
    }
    if (!config.qwen2vl_vision_path.empty()) {
        span_attributes["sdcpp.context.qwen2vl_vision_path"] = config.qwen2vl_vision_path;
    }
    if (!config.vae_path.empty()) {
        span_attributes["sdcpp.context.vae_path"] = config.vae_path;
    }
    if (!config.diffusion_model_path.empty()) {
        span_attributes["sdcpp.context.diffusion_model_path"] = config.diffusion_model_path;
    }
    if (!config.high_noise_diffusion_model_path.empty()) {
        span_attributes["sdcpp.context.high_noise_diffusion_model_path"] = config.high_noise_diffusion_model_path;
    }
    if (!config.photo_maker_path.empty()) {
        span_attributes["sdcpp.context.photo_maker_path"] = config.photo_maker_path;
    }

    root_span.name = "gen_ai.inference.image_generation";
    root_span.duration_ms = static_cast<double>(elapsed_ms);
    root_span.attributes = span_attributes;

    std::function<json(const SpanRecord&)> span_to_json;
    span_to_json = [&](const SpanRecord& record) {
        json span = json::object();
        span["name"] = record.name;
        span["span_id"] = record.span_id;
        span["kind"] = record.kind;
        span["duration_ms"] = record.duration_ms;
        if (!record.parent_span_id.empty()) {
            span["parent_span_id"] = record.parent_span_id;
        }
        if (!record.attributes.empty()) {
            span["attributes"] = record.attributes;
        }
        if (!record.subspans.empty()) {
            json subspans = json::array();
            for (const auto& child : record.subspans) {
                subspans.push_back(span_to_json(child));
            }
            span["subspans"] = std::move(subspans);
        }
        return span;
    };

    json telemetry = json::object();
    telemetry["summary"] = std::move(summary);
    json spans = json::array();
    spans.push_back(span_to_json(root_span));
    telemetry["spans"] = std::move(spans);

    return telemetry;
}
json logs_to_json(const LogCollector& collector) {
    json entries = json::array();
    for (const auto& entry : collector.entries) {
        entries.push_back({{"level", log_level_to_string(entry.level)}, {"message", entry.message}});
    }
    return entries;
}

bool ensure_context(ServerState& state, const CtxConfig& desired, std::string& error_message) {
    if (desired.model_path.empty() && desired.diffusion_model_path.empty()) {
        error_message = "model_path or diffusion_model_path must be provided";
        return false;
    }

    bool needs_reload = (state.ctx == nullptr) || (desired != state.ctx_config);
    if (!needs_reload) {
        return true;
    }

    if (state.ctx == nullptr) {
        state.ctx_config = desired;
        sd_ctx_params_t params = state.ctx_config.to_sd_params();
        sd_ctx_t* new_ctx = new_sd_ctx(&params);
        if (new_ctx == nullptr) {
            error_message = "failed to create Stable Diffusion context";
            return false;
        }
        state.ctx = new_ctx;
        return true;
    }

    const CtxConfig& current = state.ctx_config;

    auto needs_full_reload = [&]() {
        return current.model_path != desired.model_path ||
               current.n_threads != desired.n_threads ||
               current.wtype != desired.wtype ||
               current.rng_type != desired.rng_type ||
               current.offload_params_to_cpu != desired.offload_params_to_cpu ||
               current.keep_clip_on_cpu != desired.keep_clip_on_cpu ||
               current.keep_control_net_on_cpu != desired.keep_control_net_on_cpu ||
               current.keep_vae_on_cpu != desired.keep_vae_on_cpu ||
               current.vae_decode_only != desired.vae_decode_only ||
               current.free_params_immediately != desired.free_params_immediately ||
               current.taesd_path != desired.taesd_path ||
               current.control_net_path != desired.control_net_path ||
               current.lora_model_dir != desired.lora_model_dir ||
               current.embedding_dir != desired.embedding_dir ||
               current.photo_maker_path != desired.photo_maker_path ||
               current.clip_vision_path != desired.clip_vision_path ||
               current.force_sdxl_vae_conv_scale != desired.force_sdxl_vae_conv_scale ||
               current.chroma_use_dit_mask != desired.chroma_use_dit_mask ||
               current.chroma_use_t5_mask != desired.chroma_use_t5_mask ||
               current.chroma_t5_mask_pad != desired.chroma_t5_mask_pad ||
               current.flow_shift != desired.flow_shift ||
               current.prediction != desired.prediction;
    };

    if (needs_full_reload()) {
        CtxConfig previous_config = state.ctx_config;
        sd_ctx_t* previous_ctx = state.ctx;
        if (previous_ctx != nullptr) {
            free_sd_ctx(previous_ctx);
            state.ctx = nullptr;
        }

        state.ctx_config = desired;
        sd_ctx_params_t params = state.ctx_config.to_sd_params();
        sd_ctx_t* new_ctx = new_sd_ctx(&params);
        if (new_ctx == nullptr) {
            error_message = "failed to create Stable Diffusion context";
            state.ctx_config = previous_config;
            if (!previous_config.model_path.empty() || !previous_config.diffusion_model_path.empty()) {
                sd_ctx_params_t restore_params = state.ctx_config.to_sd_params();
                state.ctx = new_sd_ctx(&restore_params);
            }
            return false;
        }

        state.ctx = new_ctx;
        return true;
    }

    bool diffusion_changed = current.diffusion_model_path != desired.diffusion_model_path ||
                             current.high_noise_diffusion_model_path != desired.high_noise_diffusion_model_path ||
                             current.diffusion_flash_attn != desired.diffusion_flash_attn ||
                             current.diffusion_conv_direct != desired.diffusion_conv_direct;

    bool vae_changed = current.vae_path != desired.vae_path ||
                       current.vae_conv_direct != desired.vae_conv_direct;

    bool text_encoders_changed = current.clip_l_path != desired.clip_l_path ||
                                 current.clip_g_path != desired.clip_g_path ||
                                 current.t5xxl_path != desired.t5xxl_path ||
                                 current.qwen2vl_path != desired.qwen2vl_path ||
                                 current.qwen2vl_vision_path != desired.qwen2vl_vision_path;

    if (diffusion_changed) {
        if (!sd_reload_diffusion_model(state.ctx,
                                       desired.diffusion_model_path.c_str(),
                                       desired.high_noise_diffusion_model_path.c_str(),
                                       desired.diffusion_flash_attn,
                                       desired.diffusion_conv_direct)) {
            error_message = "failed to reload diffusion model";
            return false;
        }
    }

    if (vae_changed) {
        if (!sd_reload_vae(state.ctx,
                          desired.vae_path.c_str(),
                          desired.vae_conv_direct)) {
            error_message = "failed to reload VAE";
            return false;
        }
    }

    if (text_encoders_changed) {
        if (!sd_reload_text_encoders(state.ctx,
                                     desired.clip_l_path.c_str(),
                                     desired.clip_g_path.c_str(),
                                     desired.t5xxl_path.c_str(),
                                     desired.qwen2vl_path.c_str(),
                                     desired.qwen2vl_vision_path.c_str())) {
            error_message = "failed to reload text encoders";
            return false;
        }
    }

    state.ctx_config = desired;
    return true;
}

void sd_server_log_callback(sd_log_level_t level, const char* text, void* user_data) {
    if (!text || !user_data) {
        return;
    }

    ServerState* state = static_cast<ServerState*>(user_data);
    std::string message;
    bool only_partial = false;

    {
        std::lock_guard<std::mutex> guard(state->log_mutex);

        std::string combined = state->pending_log_fragment;
        combined.append(text);

        Utf8SplitResult sanitized = extract_complete_utf8(combined);
        state->pending_log_fragment = std::move(sanitized.remainder);

        message = std::move(sanitized.valid);
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
            message.pop_back();
        }

        only_partial = message.empty() && !state->pending_log_fragment.empty();
        if (only_partial) {
            return;
        }

        if (state->active_collector != nullptr) {
            state->active_collector->add(level, message);
            return;
        }
    }

    if (state->verbose) {
        if (level == SD_LOG_ERROR) {
            std::cerr << '[' << log_level_tag(level) << "] " << message << std::endl;
        } else {
            std::cout << '[' << log_level_tag(level) << "] " << message << std::endl;
        }
    }
}

json make_error_response(const std::string& message, const LogCollector& collector) {
    json response;
    response["success"] = false;
    response["error"] = message;
    response["logs"] = logs_to_json(collector);
    return response;
}

json make_success_response(const json& images,
                           int64_t elapsed_ms,
                           const GenerationRequest& request,
                           const CtxConfig& config,
                           const LogCollector& collector,
                           int64_t effective_seed) {
    json response;
    response["success"] = true;
    response["logs"] = logs_to_json(collector);
    response["model_path"] = config.model_path;
    response["batch_count"] = static_cast<int>(images.size());
    response["requested_seed"] = request.seed;
    response["elapsed_ms"] = elapsed_ms;
    response["telemetry"] = make_telemetry(collector, request, config, elapsed_ms, effective_seed);
    response["images"] = images;
    return response;
}

}  // namespace
int main(int argc, char** argv) {
    CLIOptions options;
    bool show_help = false;
    std::string error_message;
    if (!parse_arguments(argc, argv, options, show_help, error_message)) {
        if (show_help) {
            print_usage();
            return 0;
        }
        std::cerr << "Error: " << error_message << std::endl;
        print_usage();
        return 1;
    }

    ServerState state;
    state.verbose = options.verbose;
    state.ctx_config.model_path = options.model_path;
    state.ctx_config.clip_l_path = options.clip_l_path;
    state.ctx_config.clip_g_path = options.clip_g_path;
    state.ctx_config.clip_vision_path = options.clip_vision_path;
    state.ctx_config.t5xxl_path = options.t5xxl_path;
    state.ctx_config.qwen2vl_path = options.qwen2vl_path;
    state.ctx_config.qwen2vl_vision_path = options.qwen2vl_vision_path;
    state.ctx_config.diffusion_model_path = options.diffusion_model_path;
    state.ctx_config.high_noise_diffusion_model_path = options.high_noise_diffusion_model_path;
    state.ctx_config.vae_path = options.vae_path;
    state.ctx_config.taesd_path = options.taesd_path;
    state.ctx_config.control_net_path = options.control_net_path;
    state.ctx_config.lora_model_dir = options.lora_model_dir;
    state.ctx_config.embedding_dir = options.embedding_dir;
    state.ctx_config.photo_maker_path = options.photo_maker_path;
    state.ctx_config.vae_decode_only = true;
    state.ctx_config.free_params_immediately = false;
    state.ctx_config.n_threads = options.n_threads;
    state.ctx_config.wtype = options.wtype;
    state.ctx_config.rng_type = options.rng_type;
    state.ctx_config.offload_params_to_cpu = options.offload_params_to_cpu;
    state.ctx_config.keep_control_net_on_cpu = options.control_net_cpu;
    state.ctx_config.keep_clip_on_cpu = options.clip_on_cpu;
    state.ctx_config.keep_vae_on_cpu = options.vae_on_cpu;
    state.ctx_config.diffusion_flash_attn = options.diffusion_flash_attn;
    state.ctx_config.diffusion_conv_direct = options.diffusion_conv_direct;
    state.ctx_config.vae_conv_direct = options.vae_conv_direct;
    state.ctx_config.force_sdxl_vae_conv_scale = options.force_sdxl_vae_conv_scale;
    state.ctx_config.chroma_use_dit_mask = options.chroma_use_dit_mask;
    state.ctx_config.chroma_use_t5_mask = options.chroma_use_t5_mask;
    state.ctx_config.chroma_t5_mask_pad = options.chroma_t5_mask_pad;
    state.ctx_config.flow_shift = options.flow_shift;
    state.ctx_config.prediction = options.prediction;
    state.default_config = state.ctx_config;
    sd_easycache_params_init(&state.default_easycache);
    if (options.easycache_provided) {
        state.default_easycache = options.easycache_params;
    } else {
        state.default_easycache.enabled = false;
    }

    sd_set_log_callback(sd_server_log_callback, &state);

    {
        std::unique_lock<std::mutex> lock(state.mutex);
        LogCollector collector;
        LogCaptureScope capture(state, collector);
        std::string init_error;
        if (!ensure_context(state, state.ctx_config, init_error)) {
            std::cerr << "failed to initialize context: " << init_error << std::endl;
            for (const auto& entry : collector.entries) {
                std::cerr << '[' << log_level_tag(entry.level) << "] " << entry.message << std::endl;
            }
            return 1;
        }
    }

    httplib::Server server;

    server.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string message = "internal server error";
        try {
            if (ep) {
                std::rethrow_exception(ep);
            }
        } catch (const std::exception& ex) {
            message = ex.what();
        }
        json response = {{"success", false}, {"error", message}};
        res.status = 500;
        res.set_content(response.dump(), "application/json");
    });

    server.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        auto collector_ptr = std::make_shared<LogCollector>();
        LogCollector& collector = *collector_ptr;

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& ex) {
            auto response = make_error_response(std::string("invalid JSON payload: ") + ex.what(), collector);
            res.status = 400;
            res.set_content(response.dump(), "application/json");
            return;
        }

        GenerationRequest request_params;
        std::string parse_error;
        if (!parse_generation_request(body, request_params, parse_error)) {
            auto response = make_error_response(parse_error, collector);
            res.status = 400;
            res.set_content(response.dump(), "application/json");
            return;
        }

        std::string prepare_error;
        if (!prepare_generation_inputs(request_params, prepare_error)) {
            auto response = make_error_response(prepare_error, collector);
            res.status = 400;
            res.set_content(response.dump(), "application/json");
            return;
        }

        if (!request_params.easycache_provided) {
            if (state.default_easycache.enabled) {
                request_params.easycache = state.default_easycache;
                request_params.easycache_provided = true;
            } else {
                request_params.easycache.enabled = false;
            }
        }

        std::unique_lock<std::mutex> lock(state.mutex);
        auto capture_scope = std::make_unique<LogCaptureScope>(state, collector);

        CtxConfig desired_config = state.ctx_config;
        if (desired_config.model_path.empty()) {
            desired_config = state.default_config;
        }

        const bool has_vae_override = body.find("vae_path") != body.end();
        if (!has_vae_override) {
            desired_config.vae_path = state.default_config.vae_path;
        }

        std::string context_error;
        if (!apply_context_overrides(body, desired_config, context_error)) {
            auto response = make_error_response(context_error, collector);
            res.status = 400;
            res.set_content(response.dump(), "application/json");
            return;
        }

        if (!ensure_context(state, desired_config, context_error)) {
            auto response = make_error_response(context_error, collector);
            res.status = 500;
            res.set_content(response.dump(), "application/json");
            return;
        }

        const bool random_seed_requested = request_params.seed < 0;
        int64_t effective_seed = request_params.seed;
        if (random_seed_requested) {
            // Interpret sentinel seeds (e.g. -1) as a request for a fresh random seed.
            effective_seed = generate_random_seed();
        }

        GenerationRequest streaming_request = std::move(request_params);
        CtxConfig active_config = state.ctx_config;
        auto streaming_responder = std::make_shared<StreamingImageResponder>(state,
                                                                              std::move(lock),
                                                                              std::move(capture_scope),
                                                                              collector_ptr,
                                                                              std::move(streaming_request),
                                                                              std::move(active_config),
                                                                              random_seed_requested,
                                                                              effective_seed);
        res.status = 200;
        res.set_chunked_content_provider(
            "application/json",
            [streaming_responder](size_t, httplib::DataSink& sink) {
                return streaming_responder->next(sink);
            },
            [streaming_responder](bool) {
                streaming_responder->cancel();
            });
        return;
    });

    server.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content("health: ok", "text/plain");
    });

    server.Post("/free", [&](const httplib::Request&, httplib::Response& res) {
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.ctx != nullptr) {
            free_sd_ctx(state.ctx);
            state.ctx = nullptr;
        }
        json response = {{"success", true}, {"message", "context released"}, {"model_path", state.ctx_config.model_path}};
        res.status = 200;
        res.set_content(response.dump(), "application/json");
    });

    std::cout << "sd-server listening on port " << options.port << std::endl;
    if (!server.listen("0.0.0.0", options.port)) {
        std::cerr << "failed to start HTTP server on port " << options.port << std::endl;
        return 1;
    }

    if (state.ctx != nullptr) {
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
    }

    return 0;
}
