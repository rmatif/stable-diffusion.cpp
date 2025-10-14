#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "stable-diffusion.h"

#include "httplib.h"
#include "json.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

using json = nlohmann::json;

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

struct CLIOptions {
    std::string model_path;
    int port = 8000;
    int n_threads = -1;
    bool verbose = false;
    bool diffusion_flash_attn = false;
    bool diffusion_conv_direct = false;
    bool vae_conv_direct = false;
};

void print_usage() {
    std::cout
        << "Usage: sd-server -m <model_path> [-t <threads>] [--port <port>] [--verbose] [--diffusion-fa]"
        << " [--diffusion-conv-direct] [--vae-conv-direct]" << std::endl;
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
            if (options.n_threads <= 0) {
                error = "thread count must be greater than 0";
                return false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--diffusion-fa") {
            options.diffusion_flash_attn = true;
        } else if (arg == "--diffusion-conv-direct") {
            options.diffusion_conv_direct = true;
        } else if (arg == "--vae-conv-direct") {
            options.vae_conv_direct = true;
        } else if (arg == "-h" || arg == "--help") {
            show_help = true;
            return false;
        } else {
            error = "unknown argument: " + arg;
            return false;
        }
    }

    if (options.model_path.empty()) {
        error = "model path is required (-m/--model)";
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
    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask = false;
    int chroma_t5_mask_pad = 1;
    float flow_shift = std::numeric_limits<float>::infinity();

    bool operator==(const CtxConfig& other) const {
        return model_path == other.model_path &&
               clip_l_path == other.clip_l_path &&
               clip_g_path == other.clip_g_path &&
               clip_vision_path == other.clip_vision_path &&
               t5xxl_path == other.t5xxl_path &&
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
               chroma_use_dit_mask == other.chroma_use_dit_mask &&
               chroma_use_t5_mask == other.chroma_use_t5_mask &&
               chroma_t5_mask_pad == other.chroma_t5_mask_pad &&
               flow_shift == other.flow_shift;
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
        params.chroma_use_dit_mask     = chroma_use_dit_mask;
        params.chroma_use_t5_mask      = chroma_use_t5_mask;
        params.chroma_t5_mask_pad      = chroma_t5_mask_pad;
        params.flow_shift              = flow_shift;

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

    return true;
}

bool parse_generation_request(const json& body, GenerationRequest& request, std::string& error) {
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
    std::deque<std::string> pending_tensor_sources;
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

    for (const auto& entry : collector.entries) {
        const std::string& message = entry.message;

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
                pending_tensor_sources.push_back(path);
            }
            continue;
        }

        if (message.find("loading tensors completed") != std::string::npos) {
            auto duration = extract_duration_ms(message);
            std::string path;
            if (!pending_tensor_sources.empty()) {
                path = pending_tensor_sources.front();
                pending_tensor_sources.pop_front();
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
    if (request.has_img_cfg_scale) {
        span_attributes["sdcpp.request.img_cfg_scale"] = request.img_cfg_scale;
    }
    span_attributes["gen_ai.response.latency_ms"] = elapsed_ms;

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
    state.ctx_config.vae_decode_only = true;
    state.ctx_config.free_params_immediately = false;
    state.ctx_config.n_threads = options.n_threads;
    state.ctx_config.diffusion_flash_attn = options.diffusion_flash_attn;
    state.ctx_config.diffusion_conv_direct = options.diffusion_conv_direct;
    state.ctx_config.vae_conv_direct = options.vae_conv_direct;
    state.default_config = state.ctx_config;

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
        LogCollector collector;

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

        std::unique_lock<std::mutex> lock(state.mutex);
        LogCaptureScope capture(state, collector);

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

        sd_img_gen_params_t img_params;
        sd_img_gen_params_init(&img_params);

        img_params.prompt           = request_params.prompt.c_str();
        img_params.negative_prompt  = request_params.negative_prompt.c_str();
        img_params.clip_skip        = request_params.clip_skip;
        img_params.width            = request_params.width;
        img_params.height           = request_params.height;
        img_params.batch_count      = request_params.batch_count;
        img_params.seed             = effective_seed;
        if (request_params.has_vae_tiling_override) {
            img_params.vae_tiling_params = request_params.vae_tiling_params;
        }

        sd_sample_params_t& sample_params = img_params.sample_params;
        sample_params.sample_steps = request_params.sample_steps;
        sample_params.guidance.txt_cfg = request_params.cfg_scale;
        if (request_params.has_img_cfg_scale) {
            sample_params.guidance.img_cfg = request_params.img_cfg_scale;
        }
        if (!std::isfinite(sample_params.guidance.img_cfg)) {
            sample_params.guidance.img_cfg = sample_params.guidance.txt_cfg;
        }
        if (request_params.override_sample_method) {
            sample_params.sample_method = request_params.sample_method;
        }
        if (sample_params.sample_method == SAMPLE_METHOD_DEFAULT) {
            sample_params.sample_method = sd_get_default_sample_method(state.ctx);
        }
        if (request_params.override_scheduler) {
            sample_params.scheduler = request_params.scheduler;
        }
        if (request_params.has_eta) {
            sample_params.eta = request_params.eta;
        }
        sample_params.shifted_timestep = request_params.shifted_timestep;

        auto start_time = std::chrono::steady_clock::now();
        sd_image_t* results = generate_image(state.ctx, &img_params);
        if (results == nullptr) {
            auto response = make_error_response("image generation failed", collector);
            res.status = 500;
            res.set_content(response.dump(), "application/json");
            return;
        }

        ImageResultGuard guard{results, img_params.batch_count};

        json images = json::array();
        for (int i = 0; i < img_params.batch_count; ++i) {
            sd_image_t& image = results[i];
            if (image.data == nullptr) {
                auto response = make_error_response("image data is empty", collector);
                res.status = 500;
                res.set_content(response.dump(), "application/json");
                return;
            }
            int png_size = 0;
            unsigned char* png_data = stbi_write_png_to_mem(image.data, 0, image.width, image.height, image.channel, &png_size, nullptr);
            if (png_data == nullptr) {
                auto response = make_error_response("failed to encode PNG", collector);
                res.status = 500;
                res.set_content(response.dump(), "application/json");
                return;
            }
            std::string encoded = base64_encode(png_data, static_cast<size_t>(png_size));
            STBIW_FREE(png_data);

            // Preserve the legacy -1 seed while still reporting the concrete seed that was used.
            int64_t actual_seed = random_seed_requested ? (effective_seed + i) : (request_params.seed + i);
            int64_t reported_seed = random_seed_requested ? -1 : actual_seed;
            images.push_back({{"index", i},
                              {"seed", reported_seed},
                              {"actual_seed", actual_seed},
                              {"width", image.width},
                              {"height", image.height},
                              {"format", "png"},
                              {"mime_type", "image/png"},
                              {"data", std::move(encoded)}});
        }

        auto end_time = std::chrono::steady_clock::now();
        int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        auto response = make_success_response(images, elapsed_ms, request_params, state.ctx_config, collector, effective_seed);
        response["applied_seed"] = effective_seed;
        response["random_seed_requested"] = random_seed_requested;
        res.status = 200;
        res.set_content(response.dump(), "application/json");
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
