#ifndef __LORA_HPP__
#define __LORA_HPP__

#include <mutex>
#include "ggml_extend.hpp"

#define LORA_GRAPH_BASE_SIZE 10240

struct LoraModel : public GGMLRunner {
    std::string lora_id;
    enum lora_t {
        REGULAR      = 0,
        DIFFUSERS    = 1,
        DIFFUSERS_2  = 2,
        DIFFUSERS_3  = 3,
        TRANSFORMERS = 4,
        LORA_TYPE_COUNT
    };

    const std::string lora_ups[LORA_TYPE_COUNT] = {
        ".lora_up",
        "_lora.up",
        ".lora_B",
        ".lora.up",
        ".lora_linear_layer.up",
    };

    const std::string lora_downs[LORA_TYPE_COUNT] = {
        ".lora_down",
        "_lora.down",
        ".lora_A",
        ".lora.down",
        ".lora_linear_layer.down",
    };

    const std::string lora_pre[LORA_TYPE_COUNT] = {
        "lora.",
        "",
        "",
        "",
        "",
    };

    const std::map<std::string, std::string> alt_names = {
        // mmdit
        {"final_layer.adaLN_modulation.1", "norm_out.linear"},
        {"pos_embed", "pos_embed.proj"},
        {"final_layer.linear", "proj_out"},
        {"y_embedder.mlp.0", "time_text_embed.text_embedder.linear_1"},
        {"y_embedder.mlp.2", "time_text_embed.text_embedder.linear_2"},
        {"t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1"},
        {"t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2"},
        {"x_block.mlp.fc1", "ff.net.0.proj"},
        {"x_block.mlp.fc2", "ff.net.2"},
        {"context_block.mlp.fc1", "ff_context.net.0.proj"},
        {"context_block.mlp.fc2", "ff_context.net.2"},
        {"x_block.adaLN_modulation.1", "norm1.linear"},
        {"context_block.adaLN_modulation.1", "norm1_context.linear"},
        {"context_block.attn.proj", "attn.to_add_out"},
        {"x_block.attn.proj", "attn.to_out.0"},
        {"x_block.attn2.proj", "attn2.to_out.0"},
        // flux
        {"img_in", "x_embedder"},
        // singlestream
        {"linear2", "proj_out"},
        {"modulation.lin", "norm.linear"},
        // doublestream
        {"txt_attn.proj", "attn.to_add_out"},
        {"img_attn.proj", "attn.to_out.0"},
        {"txt_mlp.0", "ff_context.net.0.proj"},
        {"txt_mlp.2", "ff_context.net.2"},
        {"img_mlp.0", "ff.net.0.proj"},
        {"img_mlp.2", "ff.net.2"},
        {"txt_mod.lin", "norm1_context.linear"},
        {"img_mod.lin", "norm1.linear"},
    };

    const std::map<std::string, std::string> qkv_prefixes = {
        // mmdit
        {"context_block.attn.qkv", "attn.add_"},  // suffix "_proj"
        {"x_block.attn.qkv", "attn.to_"},
        {"x_block.attn2.qkv", "attn2.to_"},
        // flux
        // doublestream
        {"txt_attn.qkv", "attn.add_"},  // suffix "_proj"
        {"img_attn.qkv", "attn.to_"},
    };
    const std::map<std::string, std::string> qkvm_prefixes = {
        // flux
        // singlestream
        {"linear1", ""},
    };

    const std::string* type_fingerprints = lora_ups;

    float multiplier = 1.0f;
    std::unordered_map<std::string, ggml_tensor*> lora_tensors;
    std::map<ggml_tensor*, ggml_tensor*> original_tensor_to_final_tensor;
    std::set<std::string> applied_lora_tensors;
    std::string file_path;
    ModelLoader model_loader;
    bool load_failed         = false;
    bool applied             = false;
    bool tensor_preprocessed = false;
    std::vector<int> zero_index_vec = {0};
    ggml_tensor* zero_index         = nullptr;
    enum lora_t type                = REGULAR;

    typedef std::function<bool(const std::string&)> filter_t;

    LoraModel(const std::string& lora_id,
              ggml_backend_t backend,
              const std::string& file_path = "",
              std::string prefix           = "",
              SDVersion version            = VERSION_COUNT,
              bool convert_names           = false)
        : lora_id(lora_id), file_path(file_path), GGMLRunner(backend, false) {
        bool ok = false;
        if (convert_names) {
            prefix = "lora." + prefix;
            ok     = model_loader.init_from_file_and_convert_name(file_path, prefix, version);
        } else {
            ok = model_loader.init_from_file(file_path, prefix);
        }
        if (!ok) {
            load_failed = true;
        }
    }

    std::string get_desc() override {
        return "lora";
    }

    bool load_from_file(int n_threads, filter_t filter = nullptr) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        std::unordered_map<std::string, TensorStorage> tensors_to_create;
        std::mutex lora_mutex;
        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            if (dry_run) {
                const std::string& name = tensor_storage.name;

                if (filter && !filter(name)) {
                    return true;
                }

                {
                    std::lock_guard<std::mutex> lock(lora_mutex);
                    for (int i = 0; i < LORA_TYPE_COUNT; i++) {
                        if (name.find(type_fingerprints[i]) != std::string::npos) {
                            type = (lora_t)i;
                            break;
                        }
                    }
                    tensors_to_create[name] = tensor_storage;
                }
            } else {
                const std::string& name = tensor_storage.name;
                auto iter               = lora_tensors.find(name);
                if (iter != lora_tensors.end()) {
                    *dst_tensor = iter->second;
                }
            }
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, n_threads);

        if (tensors_to_create.empty()) {
            return true;
        }

        for (const auto& pair : tensors_to_create) {
            const auto& name   = pair.first;
            const auto& ts     = pair.second;
            ggml_tensor* real  = ggml_new_tensor(params_ctx,
                                                 ts.type,
                                                 ts.n_dims,
                                                 ts.ne);
            lora_tensors[name] = real;
        }

        alloc_params_buffer();

        dry_run = false;
        model_loader.load_tensors(on_new_tensor_cb, n_threads);

        LOG_DEBUG("lora type: \"%s\"/\"%s\"", lora_downs[type].c_str(), lora_ups[type].c_str());

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    void preprocess_lora_tensors(const std::map<std::string, ggml_tensor*>& model_tensors) {
        if (tensor_preprocessed) {
            return;
        }
        tensor_preprocessed = true;
        // I really hate these hardcoded processes.
        if (model_tensors.find("cond_stage_model.1.transformer.text_model.encoder.layers.0.self_attn.in_proj.weight") != model_tensors.end()) {
            std::map<std::string, ggml_tensor*> new_lora_tensors;
            for (auto& [old_name, tensor] : lora_tensors) {
                std::string new_name = old_name;

                if (contains(new_name, "cond_stage_model.1.transformer.text_model.encoder.layers")) {
                    std::vector<std::pair<std::string, std::string>> qkv_name_map = {
                        {"self_attn.q_proj.weight", "self_attn.in_proj.weight"},
                        {"self_attn.q_proj.bias", "self_attn.in_proj.bias"},
                        {"self_attn.k_proj.weight", "self_attn.in_proj.weight.1"},
                        {"self_attn.k_proj.bias", "self_attn.in_proj.bias.1"},
                        {"self_attn.v_proj.weight", "self_attn.in_proj.weight.2"},
                        {"self_attn.v_proj.bias", "self_attn.in_proj.bias.2"},
                    };
                    for (auto kv : qkv_name_map) {
                        size_t pos = new_name.find(kv.first);
                        if (pos != std::string::npos) {
                            new_name.replace(pos, kv.first.size(), kv.second);
                        }
                    }
                }

                new_lora_tensors[new_name] = tensor;
            }

            lora_tensors = std::move(new_lora_tensors);
        }
    }

    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* a) {
        auto out = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out      = ggml_get_rows(ctx, out, zero_index);
        out      = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    }

    std::vector<std::string> to_lora_keys(std::string blk_name, SDVersion version) {
        std::vector<std::string> keys;
        // if (!sd_version_is_sd3(version) || blk_name != "model.diffusion_model.pos_embed") {
        size_t k_pos = blk_name.find(".weight");
        if (k_pos == std::string::npos) {
            return keys;
        }
        blk_name = blk_name.substr(0, k_pos);
        // }
        keys.push_back(blk_name);
        keys.push_back("lora." + blk_name);
        if (sd_version_is_dit(version)) {
            if (blk_name.find("model.diffusion_model") != std::string::npos) {
                blk_name.replace(blk_name.find("model.diffusion_model"), sizeof("model.diffusion_model") - 1, "transformer");
            }

            if (blk_name.find(".single_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".single_blocks"), sizeof(".single_blocks") - 1, ".single_transformer_blocks");
            }
            if (blk_name.find(".double_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".double_blocks"), sizeof(".double_blocks") - 1, ".transformer_blocks");
            }

            if (blk_name.find(".joint_blocks") != std::string::npos) {
                blk_name.replace(blk_name.find(".joint_blocks"), sizeof(".joint_blocks") - 1, ".transformer_blocks");
            }

            if (blk_name.find("text_encoders.clip_l") != std::string::npos) {
                blk_name.replace(blk_name.find("text_encoders.clip_l"), sizeof("text_encoders.clip_l") - 1, "cond_stage_model");
            }

            for (const auto& item : alt_names) {
                size_t match = blk_name.find(item.first);
                if (match != std::string::npos) {
                    blk_name = blk_name.substr(0, match) + item.second;
                }
            }
            for (const auto& prefix : qkv_prefixes) {
                size_t match = blk_name.find(prefix.first);
                if (match != std::string::npos) {
                    std::string split_blk = "SPLIT|" + blk_name.substr(0, match) + prefix.second;
                    keys.push_back(split_blk);
                }
            }
            for (const auto& prefix : qkvm_prefixes) {
                size_t match = blk_name.find(prefix.first);
                if (match != std::string::npos) {
                    std::string split_blk = "SPLIT_L|" + blk_name.substr(0, match) + prefix.second;
                    keys.push_back(split_blk);
                }
            }
            keys.push_back(blk_name);
        }

        std::vector<std::string> ret;
        for (std::string& key : keys) {
            ret.push_back(key);
            replace_all_chars(key, '.', '_');
            // fix for some sdxl lora, like lcm-lora-xl
            if (key == "model_diffusion_model_output_blocks_2_2_conv") {
                ret.push_back("model_diffusion_model_output_blocks_2_1_conv");
            }
            ret.push_back(key);
        }
        return ret;
    }

    ggml_tensor* get_lora_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string lora_down_name = "lora." + key + ".lora_down";
            std::string lora_up_name   = "lora." + key + ".lora_up";
            std::string lora_mid_name  = "lora." + key + ".lora_mid";
            std::string scale_name     = "lora." + key + ".scale";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lora_up   = nullptr;
            ggml_tensor* lora_mid  = nullptr;
            ggml_tensor* lora_down = nullptr;

            auto iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = ggml_ext_cast_f32(ctx, iter->second);
            }

            if (lora_up == nullptr || lora_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            if (lora_mid) {
                applied_lora_tensors.insert(lora_mid_name);
            }

            float scale_value = 1.0f;

            int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
            iter         = lora_tensors.find(scale_name);
            if (iter != lora_tensors.end()) {
                scale_value = ggml_ext_backend_tensor_get_f32(iter->second);
                applied_lora_tensors.insert(scale_name);
            } else {
                iter = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                    applied_lora_tensors.insert(alpha_name);
                }
            }
            scale_value *= multiplier;

            auto curr_updown = ggml_ext_merge_lora(ctx, lora_down, lora_up, lora_mid);
            curr_updown      = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_raw_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string diff_name = "lora." + key + ".diff";

            ggml_tensor* curr_updown = nullptr;

            auto iter = lora_tensors.find(diff_name);
            if (iter != lora_tensors.end()) {
                curr_updown = ggml_ext_cast_f32(ctx, iter->second);
            } else {
                break;
            }

            applied_lora_tensors.insert(diff_name);

            float scale_value = 1.0f;
            scale_value *= multiplier;

            curr_updown = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_loha_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string hada_1_down_name = "lora." + key + ".hada_w1_b";
            std::string hada_1_mid_name  = "lora." + key + ".hada_t1";
            std::string hada_1_up_name   = "lora." + key + ".hada_w1_a";
            std::string hada_2_down_name = "lora." + key + ".hada_w2_b";
            std::string hada_2_mid_name  = "lora." + key + ".hada_t2";
            std::string hada_2_up_name   = "lora." + key + ".hada_w2_a";
            std::string alpha_name       = "lora." + key + ".alpha";

            ggml_tensor* hada_1_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_1_up   = nullptr;
            ggml_tensor* hada_1_down = nullptr;

            ggml_tensor* hada_2_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_2_up   = nullptr;
            ggml_tensor* hada_2_down = nullptr;

            auto iter = lora_tensors.find(hada_1_down_name);
            if (iter != lora_tensors.end()) {
                hada_1_down = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_up_name);
            if (iter != lora_tensors.end()) {
                hada_1_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_1_mid_name);
            if (iter != lora_tensors.end()) {
                hada_1_mid = ggml_ext_cast_f32(ctx, iter->second);
                hada_1_up  = ggml_cont(ctx, ggml_transpose(ctx, hada_1_up));
            }

            iter = lora_tensors.find(hada_2_down_name);
            if (iter != lora_tensors.end()) {
                hada_2_down = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_up_name);
            if (iter != lora_tensors.end()) {
                hada_2_up = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(hada_2_mid_name);
            if (iter != lora_tensors.end()) {
                hada_2_mid = ggml_ext_cast_f32(ctx, iter->second);
                hada_2_up  = ggml_cont(ctx, ggml_transpose(ctx, hada_2_up));
            }

            if (hada_1_up == nullptr || hada_1_down == nullptr || hada_2_up == nullptr || hada_2_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(hada_1_down_name);
            applied_lora_tensors.insert(hada_1_up_name);
            applied_lora_tensors.insert(hada_2_down_name);
            applied_lora_tensors.insert(hada_2_up_name);
            applied_lora_tensors.insert(alpha_name);

            if (hada_1_mid) {
                applied_lora_tensors.insert(hada_1_mid_name);
            }

            if (hada_2_mid) {
                applied_lora_tensors.insert(hada_2_mid_name);
            }

            float scale_value = 1.0f;

            // calc_scale
            // TODO: .dora_scale?
            int64_t rank = hada_1_down->ne[ggml_n_dims(hada_1_down) - 1];
            iter         = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }
            scale_value *= multiplier;

            ggml_tensor* updown_1 = ggml_ext_merge_lora(ctx, hada_1_down, hada_1_up, hada_1_mid);
            ggml_tensor* updown_2 = ggml_ext_merge_lora(ctx, hada_2_down, hada_2_up, hada_2_mid);
            auto curr_updown      = ggml_mul_inplace(ctx, updown_1, updown_2);
            curr_updown           = ggml_ext_scale(ctx, curr_updown, scale_value, true);
            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_lokr_weight_diff(const std::string& model_tensor_name, ggml_context* ctx) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string lokr_w1_name   = "lora." + key + ".lokr_w1";
            std::string lokr_w1_a_name = "lora." + key + ".lokr_w1_a";
            std::string lokr_w1_b_name = "lora." + key + ".lokr_w1_b";
            std::string lokr_w2_name   = "lora." + key + ".lokr_w2";
            std::string lokr_w2_a_name = "lora." + key + ".lokr_w2_a";
            std::string lokr_w2_b_name = "lora." + key + ".lokr_w2_b";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lokr_w1   = nullptr;
            ggml_tensor* lokr_w1_a = nullptr;
            ggml_tensor* lokr_w1_b = nullptr;
            ggml_tensor* lokr_w2   = nullptr;
            ggml_tensor* lokr_w2_a = nullptr;
            ggml_tensor* lokr_w2_b = nullptr;

            auto iter = lora_tensors.find(lokr_w1_name);
            if (iter != lora_tensors.end()) {
                lokr_w1 = ggml_ext_cast_f32(ctx, iter->second);
            }

            iter = lora_tensors.find(lokr_w2_name);
            if (iter != lora_tensors.end()) {
                lokr_w2 = ggml_ext_cast_f32(ctx, iter->second);
            }

            int64_t rank = 1;
            if (lokr_w1 == nullptr) {
                iter = lora_tensors.find(lokr_w1_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = ggml_ext_cast_f32(ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = ggml_ext_cast_f32(ctx, iter->second);
                }

                if (lokr_w1_a == nullptr || lokr_w1_b == nullptr) {
                    break;
                }

                rank = lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];

                lokr_w1 = ggml_ext_merge_lora(ctx, lokr_w1_b, lokr_w1_a);
            }

            if (lokr_w2 == nullptr) {
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = ggml_ext_cast_f32(ctx, iter->second);
                }

                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = ggml_ext_cast_f32(ctx, iter->second);
                }

                if (lokr_w2_a == nullptr || lokr_w2_b == nullptr) {
                    break;
                }

                rank = lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];

                lokr_w2 = ggml_ext_merge_lora(ctx, lokr_w2_b, lokr_w2_a);
            }

            if (!lokr_w1_a) {
                applied_lora_tensors.insert(lokr_w1_name);
            } else {
                applied_lora_tensors.insert(lokr_w1_a_name);
                applied_lora_tensors.insert(lokr_w1_b_name);
            }

            if (!lokr_w2_a) {
                applied_lora_tensors.insert(lokr_w2_name);
            } else {
                applied_lora_tensors.insert(lokr_w2_a_name);
                applied_lora_tensors.insert(lokr_w2_b_name);
            }

            float scale_value = 1.0f;
            iter              = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }

            if (rank == 1) {
                scale_value = 1.0f;
            }

            scale_value *= multiplier;

            auto curr_updown = ggml_ext_kronecker(ctx, lokr_w1, lokr_w2);
            curr_updown      = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_tensor* model_tensor, bool with_lora_and_lokr = true) {
        // lora
        ggml_tensor* diff = nullptr;
        if (with_lora_and_lokr) {
            diff = get_lora_weight_diff(model_tensor_name, ctx);
        }
        // diff
        if (diff == nullptr) {
            diff = get_raw_weight_diff(model_tensor_name, ctx);
        }
        // loha
        if (diff == nullptr) {
            diff = get_loha_weight_diff(model_tensor_name, ctx);
        }
        // lokr
        if (diff == nullptr && with_lora_and_lokr) {
            diff = get_lokr_weight_diff(model_tensor_name, ctx);
        }
        if (diff != nullptr) {
            if (ggml_nelements(diff) < ggml_nelements(model_tensor)) {
                if (ggml_n_dims(diff) == 2 && ggml_n_dims(model_tensor) == 2 && diff->ne[0] == model_tensor->ne[0]) {
                    LOG_WARN("pad for %s", model_tensor_name.c_str());
                    auto pad_tensor = ggml_ext_zeros(ctx, diff->ne[0], model_tensor->ne[1] - diff->ne[1], 1, 1);
                    diff            = ggml_concat(ctx, diff, pad_tensor, 1);
                }
            }

            GGML_ASSERT(ggml_nelements(diff) == ggml_nelements(model_tensor));
            diff = ggml_reshape(ctx, diff, model_tensor);
        }
        return diff;
    }

    ggml_tensor* get_out_diff(ggml_context* ctx,
                              ggml_tensor* x,
                              WeightAdapter::ForwardParams forward_params,
                              const std::string& model_tensor_name) {
        ggml_tensor* out_diff = nullptr;
        int index             = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            bool is_conv2d = forward_params.op_type == WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;

            std::string lokr_w1_name   = "lora." + key + ".lokr_w1";
            std::string lokr_w1_a_name = "lora." + key + ".lokr_w1_a";
            // if either of these is found, then we have a lokr lora
            auto iter   = lora_tensors.find(lokr_w1_name);
            auto iter_a = lora_tensors.find(lokr_w1_a_name);
            if (iter != lora_tensors.end() || iter_a != lora_tensors.end()) {
                std::string lokr_w1_b_name = "lora." + key + ".lokr_w1_b";
                std::string lokr_w2_name   = "lora." + key + ".lokr_w2";
                std::string lokr_w2_a_name = "lora." + key + ".lokr_w2_a";
                std::string lokr_w2_b_name = "lora." + key + ".lokr_w2_b";
                std::string alpha_name     = "lora." + key + ".alpha";

                ggml_tensor* lokr_w1   = nullptr;
                ggml_tensor* lokr_w1_a = nullptr;
                ggml_tensor* lokr_w1_b = nullptr;
                ggml_tensor* lokr_w2   = nullptr;
                ggml_tensor* lokr_w2_a = nullptr;
                ggml_tensor* lokr_w2_b = nullptr;

                if (iter != lora_tensors.end()) {
                    lokr_w1 = iter->second;
                }
                iter = iter_a;
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = iter->second;
                }
                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = iter->second;
                }

                iter = lora_tensors.find(lokr_w2_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2 = iter->second;
                    if (is_conv2d && lokr_w2->type != GGML_TYPE_F16) {
                        lokr_w2 = ggml_cast(ctx, lokr_w2, GGML_TYPE_F16);
                    }
                }
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = iter->second;
                    if (is_conv2d && lokr_w2_a->type != GGML_TYPE_F16) {
                        lokr_w2_a = ggml_cast(ctx, lokr_w2_a, GGML_TYPE_F16);
                    }
                }
                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = iter->second;
                    if (is_conv2d && lokr_w2_b->type != GGML_TYPE_F16) {
                        lokr_w2_b = ggml_cast(ctx, lokr_w2_b, GGML_TYPE_F16);
                    }
                }

                int rank = 1;
                if (lokr_w1_b) {
                    rank = (int)lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];
                }
                if (lokr_w2_b) {
                    rank = (int)lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];
                }

                float scale_value = 1.0f;
                iter              = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    applied_lora_tensors.insert(alpha_name);
                }

                if (rank == 1) {
                    scale_value = 1.0f;
                }
                scale_value *= multiplier;

                auto curr_out_diff = ggml_ext_lokr_forward(ctx, x, lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, is_conv2d, forward_params.conv2d, scale_value);
                if (out_diff == nullptr) {
                    out_diff = curr_out_diff;
                } else {
                    out_diff = ggml_concat(ctx, out_diff, curr_out_diff, 0);
                }

                if (lokr_w1)
                    applied_lora_tensors.insert(lokr_w1_name);
                if (lokr_w1_a)
                    applied_lora_tensors.insert(lokr_w1_a_name);
                if (lokr_w1_b)
                    applied_lora_tensors.insert(lokr_w1_b_name);
                if (lokr_w2)
                    applied_lora_tensors.insert(lokr_w2_name);
                if (lokr_w2_a)
                    applied_lora_tensors.insert(lokr_w2_name);
                if (lokr_w2_b)
                    applied_lora_tensors.insert(lokr_w2_b_name);
                applied_lora_tensors.insert(alpha_name);

                index++;
                continue;
            }

            // not a lokr, normal lora path

            std::string lora_down_name = "lora." + key + ".lora_down";
            std::string lora_up_name   = "lora." + key + ".lora_up";
            std::string lora_mid_name  = "lora." + key + ".lora_mid";
            std::string scale_name     = "lora." + key + ".scale";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lora_up   = nullptr;
            ggml_tensor* lora_mid  = nullptr;
            ggml_tensor* lora_down = nullptr;

            iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = iter->second;
                if (is_conv2d && lora_up->type != GGML_TYPE_F16) {
                    lora_up = ggml_cast(ctx, lora_up, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = iter->second;
                if (is_conv2d && lora_mid->type != GGML_TYPE_F16) {
                    lora_mid = ggml_cast(ctx, lora_mid, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = iter->second;
                if (is_conv2d && lora_down->type != GGML_TYPE_F16) {
                    lora_down = ggml_cast(ctx, lora_down, GGML_TYPE_F16);
                }
            }

            if (lora_up == nullptr || lora_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            if (lora_mid) {
                applied_lora_tensors.insert(lora_mid_name);
            }

            float scale_value = 1.0f;

            int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
            iter         = lora_tensors.find(scale_name);
            if (iter != lora_tensors.end()) {
                scale_value = ggml_ext_backend_tensor_get_f32(iter->second);
                applied_lora_tensors.insert(scale_name);
            } else {
                iter = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                    applied_lora_tensors.insert(alpha_name);
                }
            }
            scale_value *= multiplier;

            ggml_tensor* lx;
            if (!is_conv2d) {
                lx = ggml_ext_linear(ctx, x, lora_down, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                if (lora_mid) {
                    lx = ggml_ext_linear(ctx, lx, lora_mid, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                }
                lx = ggml_ext_linear(ctx, lx, lora_up, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
            } else {  // OP_CONV2D
                lx = ggml_ext_conv_2d(ctx,
                                      x,
                                      lora_down,
                                      nullptr,
                                      forward_params.conv2d.s0,
                                      forward_params.conv2d.s1,
                                      forward_params.conv2d.p0,
                                      forward_params.conv2d.p1,
                                      forward_params.conv2d.d0,
                                      forward_params.conv2d.d1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.circular_x,
                                      forward_params.conv2d.circular_y,
                                      forward_params.conv2d.scale);
                if (lora_mid) {
                    lx = ggml_ext_conv_2d(ctx,
                                          lx,
                                          lora_mid,
                                          nullptr,
                                          1,
                                          1,
                                          0,
                                          0,
                                          1,
                                          1,
                                          forward_params.conv2d.direct,
                                          forward_params.conv2d.circular_x,
                                          forward_params.conv2d.circular_y,
                                          forward_params.conv2d.scale);
                }
                lx = ggml_ext_conv_2d(ctx,
                                      lx,
                                      lora_up,
                                      nullptr,
                                      1,
                                      1,
                                      0,
                                      0,
                                      1,
                                      1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.circular_x,
                                      forward_params.conv2d.circular_y,
                                      forward_params.conv2d.scale);
            }

            auto curr_out_diff = ggml_ext_scale(ctx, lx, scale_value, true);

            if (out_diff == nullptr) {
                out_diff = curr_out_diff;
            } else {
                out_diff = ggml_concat(ctx, out_diff, curr_out_diff, 0);
            }

            index++;
        }
        return out_diff;
    }

    ggml_cgraph* build_lora_graph(const std::map<std::string, ggml_tensor*>& model_tensors, SDVersion version) {
        size_t lora_graph_size = LORA_GRAPH_BASE_SIZE + lora_tensors.size() * 10;
        ggml_cgraph* gf        = ggml_new_graph_custom(compute_ctx, lora_graph_size, false);

        preprocess_lora_tensors(model_tensors);

        zero_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        set_backend_tensor_data(zero_index, zero_index_vec.data());
        ggml_build_forward_expand(gf, zero_index);

        original_tensor_to_final_tensor.clear();
        applied_lora_tensors.clear();

        for (auto it : model_tensors) {
            std::string model_tensor_name = it.first;
            ggml_tensor* model_tensor     = it.second;

            std::vector<std::string> keys = to_lora_keys(model_tensor_name, version);
            bool is_bias                  = ends_with(model_tensor_name, ".bias");
            if (keys.size() == 0) {
                if (is_bias) {
                    keys.push_back(model_tensor_name.substr(0, model_tensor_name.size() - 5));  // remove .bias
                } else {
                    continue;
                }
            }

            for (auto& key : keys) {
                bool is_qkv_split = starts_with(key, "SPLIT|");
                if (is_qkv_split) {
                    key = key.substr(sizeof("SPLIT|") - 1);
                }
                bool is_qkvm_split = starts_with(key, "SPLIT_L|");
                if (is_qkvm_split) {
                    key = key.substr(sizeof("SPLIT_L|") - 1);
                }
                struct ggml_tensor* updown = nullptr;
                float scale_value          = 1.0f;
                std::string full_key       = lora_pre[type] + key;
                if (is_bias) {
                    if (lora_tensors.find(full_key + ".diff_b") != lora_tensors.end()) {
                        std::string diff_name = full_key + ".diff_b";
                        ggml_tensor* diff     = lora_tensors[diff_name];
                        updown                = to_f32(compute_ctx, diff);
                        applied_lora_tensors.insert(diff_name);
                    } else {
                        continue;
                    }
                } else if (lora_tensors.find(full_key + ".diff") != lora_tensors.end()) {
                    std::string diff_name = full_key + ".diff";
                    ggml_tensor* diff     = lora_tensors[diff_name];
                    updown                = to_f32(compute_ctx, diff);
                    applied_lora_tensors.insert(diff_name);
                } else if (lora_tensors.find(full_key + ".hada_w1_a") != lora_tensors.end()) {
                    // LoHa mode

                    // TODO: split qkv convention for LoHas (is it ever used?)
                    if (is_qkv_split || is_qkvm_split) {
                        LOG_ERROR("Split qkv isn't supported for LoHa models.");
                        break;
                    }
                    std::string alpha_name = "";

                    ggml_tensor* hada_1_mid  = nullptr;  // tau for tucker decomposition
                    ggml_tensor* hada_1_up   = nullptr;
                    ggml_tensor* hada_1_down = nullptr;

                    ggml_tensor* hada_2_mid  = nullptr;  // tau for tucker decomposition
                    ggml_tensor* hada_2_up   = nullptr;
                    ggml_tensor* hada_2_down = nullptr;

                    std::string hada_1_mid_name  = "";
                    std::string hada_1_down_name = "";
                    std::string hada_1_up_name   = "";

                    std::string hada_2_mid_name  = "";
                    std::string hada_2_down_name = "";
                    std::string hada_2_up_name   = "";

                    hada_1_down_name = full_key + ".hada_w1_b";
                    hada_1_up_name   = full_key + ".hada_w1_a";
                    hada_1_mid_name  = full_key + ".hada_t1";
                    if (lora_tensors.find(hada_1_down_name) != lora_tensors.end()) {
                        hada_1_down = to_f32(compute_ctx, lora_tensors[hada_1_down_name]);
                    }
                    if (lora_tensors.find(hada_1_up_name) != lora_tensors.end()) {
                        hada_1_up = to_f32(compute_ctx, lora_tensors[hada_1_up_name]);
                    }
                    if (lora_tensors.find(hada_1_mid_name) != lora_tensors.end()) {
                        hada_1_mid = to_f32(compute_ctx, lora_tensors[hada_1_mid_name]);
                        applied_lora_tensors.insert(hada_1_mid_name);
                        hada_1_up = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_1_up));
                    }

                    hada_2_down_name = full_key + ".hada_w2_b";
                    hada_2_up_name   = full_key + ".hada_w2_a";
                    hada_2_mid_name  = full_key + ".hada_t2";
                    if (lora_tensors.find(hada_2_down_name) != lora_tensors.end()) {
                        hada_2_down = to_f32(compute_ctx, lora_tensors[hada_2_down_name]);
                    }
                    if (lora_tensors.find(hada_2_up_name) != lora_tensors.end()) {
                        hada_2_up = to_f32(compute_ctx, lora_tensors[hada_2_up_name]);
                    }
                    if (lora_tensors.find(hada_2_mid_name) != lora_tensors.end()) {
                        hada_2_mid = to_f32(compute_ctx, lora_tensors[hada_2_mid_name]);
                        applied_lora_tensors.insert(hada_2_mid_name);
                        hada_2_up = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, hada_2_up));
                    }

                    alpha_name = full_key + ".alpha";

                    applied_lora_tensors.insert(hada_1_down_name);
                    applied_lora_tensors.insert(hada_1_up_name);
                    applied_lora_tensors.insert(hada_2_down_name);
                    applied_lora_tensors.insert(hada_2_up_name);

                    applied_lora_tensors.insert(alpha_name);
                    if (hada_1_up == nullptr || hada_1_down == nullptr || hada_2_up == nullptr || hada_2_down == nullptr) {
                        continue;
                    }

                    struct ggml_tensor* updown_1 = ggml_ext_merge_lora(compute_ctx, hada_1_down, hada_1_up, hada_1_mid);
                    struct ggml_tensor* updown_2 = ggml_ext_merge_lora(compute_ctx, hada_2_down, hada_2_up, hada_2_mid);
                    updown                       = ggml_mul_inplace(compute_ctx, updown_1, updown_2);

                    // calc_scale
                    // TODO: .dora_scale?
                    int64_t rank = hada_1_down->ne[ggml_n_dims(hada_1_down) - 1];
                    if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                        float alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[alpha_name]);
                        scale_value = alpha / rank;
                    }
                } else if (lora_tensors.find(full_key + ".lokr_w1") != lora_tensors.end() || lora_tensors.find(full_key + ".lokr_w1_a") != lora_tensors.end()) {
                    // LoKr mode

                    // TODO: split qkv convention for LoKrs (is it ever used?)
                    if (is_qkv_split || is_qkvm_split) {
                        LOG_ERROR("Split qkv isn't supported for LoKr models.");
                        break;
                    }

                    std::string alpha_name = full_key + ".alpha";

                    ggml_tensor* lokr_w1 = nullptr;
                    ggml_tensor* lokr_w2 = nullptr;

                    std::string lokr_w1_name = "";
                    std::string lokr_w2_name = "";

                    lokr_w1_name = full_key + ".lokr_w1";
                    lokr_w2_name = full_key + ".lokr_w2";

                    if (lora_tensors.find(lokr_w1_name) != lora_tensors.end()) {
                        lokr_w1 = to_f32(compute_ctx, lora_tensors[lokr_w1_name]);
                        applied_lora_tensors.insert(lokr_w1_name);
                    } else {
                        ggml_tensor* down     = nullptr;
                        ggml_tensor* up       = nullptr;
                        std::string down_name = lokr_w1_name + "_b";
                        std::string up_name   = lokr_w1_name + "_a";
                        if (lora_tensors.find(down_name) != lora_tensors.end()) {
                            // w1 should not be low rank normally, sometimes w1 and w2 are swapped
                            down = to_f32(compute_ctx, lora_tensors[down_name]);
                            applied_lora_tensors.insert(down_name);

                            int64_t rank = down->ne[ggml_n_dims(down) - 1];
                            if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                                float alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[alpha_name]);
                                scale_value = alpha / rank;
                            }
                        }
                        if (lora_tensors.find(up_name) != lora_tensors.end()) {
                            up = to_f32(compute_ctx, lora_tensors[up_name]);
                            applied_lora_tensors.insert(up_name);
                        }
                        lokr_w1 = ggml_ext_merge_lora(compute_ctx, down, up);
                    }
                    if (lora_tensors.find(lokr_w2_name) != lora_tensors.end()) {
                        lokr_w2 = to_f32(compute_ctx, lora_tensors[lokr_w2_name]);
                        applied_lora_tensors.insert(lokr_w2_name);
                    } else {
                        ggml_tensor* down     = nullptr;
                        ggml_tensor* up       = nullptr;
                        std::string down_name = lokr_w2_name + "_b";
                        std::string up_name   = lokr_w2_name + "_a";
                        if (lora_tensors.find(down_name) != lora_tensors.end()) {
                            down = to_f32(compute_ctx, lora_tensors[down_name]);
                            applied_lora_tensors.insert(down_name);

                            int64_t rank = down->ne[ggml_n_dims(down) - 1];
                            if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                                float alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[alpha_name]);
                                scale_value = alpha / rank;
                            }
                        }
                        if (lora_tensors.find(up_name) != lora_tensors.end()) {
                            up = to_f32(compute_ctx, lora_tensors[up_name]);
                            applied_lora_tensors.insert(up_name);
                        }
                        lokr_w2 = ggml_ext_merge_lora(compute_ctx, down, up);
                    }

                    // Technically it might be unused, but I believe it's the expected behavior
                    applied_lora_tensors.insert(alpha_name);

                    updown = ggml_ext_kronecker(compute_ctx, lokr_w1, lokr_w2);

                } else {
                    // LoRA mode
                    ggml_tensor* lora_mid  = nullptr;  // tau for tucker decomposition
                    ggml_tensor* lora_up   = nullptr;
                    ggml_tensor* lora_down = nullptr;

                    std::string alpha_name         = "";
                    std::string scale_name         = "";
                    std::string split_q_scale_name = "";
                    std::string lora_mid_name      = "";
                    std::string lora_down_name     = "";
                    std::string lora_up_name       = "";

                    if (is_qkv_split) {
                        std::string suffix  = "";
                        auto split_q_d_name = full_key + "q" + suffix + lora_downs[type] + ".weight";

                        if (lora_tensors.find(split_q_d_name) == lora_tensors.end()) {
                            suffix         = "_proj";
                            split_q_d_name = full_key + "q" + suffix + lora_downs[type] + ".weight";
                        }
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = full_key + "k" + suffix + lora_downs[type] + ".weight";
                            auto split_v_d_name = full_key + "v" + suffix + lora_downs[type] + ".weight";

                            auto split_q_u_name = full_key + "q" + suffix + lora_ups[type] + ".weight";
                            auto split_k_u_name = full_key + "k" + suffix + lora_ups[type] + ".weight";
                            auto split_v_u_name = full_key + "v" + suffix + lora_ups[type] + ".weight";

                            auto split_q_scale_name = full_key + "q" + suffix + ".scale";
                            auto split_k_scale_name = full_key + "k" + suffix + ".scale";
                            auto split_v_scale_name = full_key + "v" + suffix + ".scale";

                            auto split_q_alpha_name = full_key + "q" + suffix + ".alpha";
                            auto split_k_alpha_name = full_key + "k" + suffix + ".alpha";
                            auto split_v_alpha_name = full_key + "v" + suffix + ".alpha";

                            ggml_tensor* lora_q_down = nullptr;
                            ggml_tensor* lora_q_up   = nullptr;
                            ggml_tensor* lora_k_down = nullptr;
                            ggml_tensor* lora_k_up   = nullptr;
                            ggml_tensor* lora_v_down = nullptr;
                            ggml_tensor* lora_v_up   = nullptr;

                            lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            float q_rank = lora_q_up->ne[0];
                            float k_rank = lora_k_up->ne[0];
                            float v_rank = lora_v_up->ne[0];

                            float lora_q_scale = 1;
                            float lora_k_scale = 1;
                            float lora_v_scale = 1;

                            if (lora_tensors.find(split_q_scale_name) != lora_tensors.end()) {
                                lora_q_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_q_scale_name]);
                                applied_lora_tensors.insert(split_q_scale_name);
                            }
                            if (lora_tensors.find(split_k_scale_name) != lora_tensors.end()) {
                                lora_k_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_k_scale_name]);
                                applied_lora_tensors.insert(split_k_scale_name);
                            }
                            if (lora_tensors.find(split_v_scale_name) != lora_tensors.end()) {
                                lora_v_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_v_scale_name]);
                                applied_lora_tensors.insert(split_v_scale_name);
                            }

                            if (lora_tensors.find(split_q_alpha_name) != lora_tensors.end()) {
                                float lora_q_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_q_alpha_name]);
                                applied_lora_tensors.insert(split_q_alpha_name);
                                lora_q_scale = lora_q_alpha / q_rank;
                            }
                            if (lora_tensors.find(split_k_alpha_name) != lora_tensors.end()) {
                                float lora_k_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_k_alpha_name]);
                                applied_lora_tensors.insert(split_k_alpha_name);
                                lora_k_scale = lora_k_alpha / k_rank;
                            }
                            if (lora_tensors.find(split_v_alpha_name) != lora_tensors.end()) {
                                float lora_v_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_v_alpha_name]);
                                applied_lora_tensors.insert(split_v_alpha_name);
                                lora_v_scale = lora_v_alpha / v_rank;
                            }

                            ggml_scale_inplace(compute_ctx, lora_q_down, lora_q_scale);
                            ggml_scale_inplace(compute_ctx, lora_k_down, lora_k_scale);
                            ggml_scale_inplace(compute_ctx, lora_v_down, lora_v_scale);

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);    //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);    //[R, 3072, 1, 1]

                            // these need to be stitched together this way:
                            //                          |q_up,0   ,0   |
                            //                          |0   ,k_up,0   |
                            //                          |0   ,0   ,v_up|
                            // (q_down,k_down,v_down) . (q   ,k   ,v)

                            // up_concat will be [9216, R*3, 1, 1]
                            // down_concat will be [R*3, 3072, 1, 1]
                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), lora_v_down, 1);

                            ggml_tensor* z = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, lora_q_up, zz, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), z, 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, zz, lora_v_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 9216, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 9216, 1, 1]
                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), v_up, 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*3, 9216, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                        }
                    } else if (is_qkvm_split) {
                        auto split_q_d_name = full_key + "attn.to_q" + lora_downs[type] + ".weight";
                        if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                            // print_ggml_tensor(it.second, true);  //[3072, 21504, 1, 1]
                            // find qkv and mlp up parts in LoRA model
                            auto split_k_d_name = full_key + "attn.to_k" + lora_downs[type] + ".weight";
                            auto split_v_d_name = full_key + "attn.to_v" + lora_downs[type] + ".weight";

                            auto split_q_u_name = full_key + "attn.to_q" + lora_ups[type] + ".weight";
                            auto split_k_u_name = full_key + "attn.to_k" + lora_ups[type] + ".weight";
                            auto split_v_u_name = full_key + "attn.to_v" + lora_ups[type] + ".weight";

                            auto split_m_d_name = full_key + "proj_mlp" + lora_downs[type] + ".weight";
                            auto split_m_u_name = full_key + "proj_mlp" + lora_ups[type] + ".weight";

                            auto split_q_scale_name = full_key + "attn.to_q" + ".scale";
                            auto split_k_scale_name = full_key + "attn.to_k" + ".scale";
                            auto split_v_scale_name = full_key + "attn.to_v" + ".scale";
                            auto split_m_scale_name = full_key + "proj_mlp" + ".scale";

                            auto split_q_alpha_name = full_key + "attn.to_q" + ".alpha";
                            auto split_k_alpha_name = full_key + "attn.to_k" + ".alpha";
                            auto split_v_alpha_name = full_key + "attn.to_v" + ".alpha";
                            auto split_m_alpha_name = full_key + "proj_mlp" + ".alpha";

                            ggml_tensor* lora_q_down = nullptr;
                            ggml_tensor* lora_q_up   = nullptr;
                            ggml_tensor* lora_k_down = nullptr;
                            ggml_tensor* lora_k_up   = nullptr;
                            ggml_tensor* lora_v_down = nullptr;
                            ggml_tensor* lora_v_up   = nullptr;

                            ggml_tensor* lora_m_down = nullptr;
                            ggml_tensor* lora_m_up   = nullptr;

                            lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);

                            if (lora_tensors.find(split_q_d_name) != lora_tensors.end()) {
                                lora_q_down = to_f32(compute_ctx, lora_tensors[split_q_d_name]);
                            }

                            if (lora_tensors.find(split_q_u_name) != lora_tensors.end()) {
                                lora_q_up = to_f32(compute_ctx, lora_tensors[split_q_u_name]);
                            }

                            if (lora_tensors.find(split_k_d_name) != lora_tensors.end()) {
                                lora_k_down = to_f32(compute_ctx, lora_tensors[split_k_d_name]);
                            }

                            if (lora_tensors.find(split_k_u_name) != lora_tensors.end()) {
                                lora_k_up = to_f32(compute_ctx, lora_tensors[split_k_u_name]);
                            }

                            if (lora_tensors.find(split_v_d_name) != lora_tensors.end()) {
                                lora_v_down = to_f32(compute_ctx, lora_tensors[split_v_d_name]);
                            }

                            if (lora_tensors.find(split_v_u_name) != lora_tensors.end()) {
                                lora_v_up = to_f32(compute_ctx, lora_tensors[split_v_u_name]);
                            }

                            if (lora_tensors.find(split_m_d_name) != lora_tensors.end()) {
                                lora_m_down = to_f32(compute_ctx, lora_tensors[split_m_d_name]);
                            }

                            if (lora_tensors.find(split_m_u_name) != lora_tensors.end()) {
                                lora_m_up = to_f32(compute_ctx, lora_tensors[split_m_u_name]);
                            }

                            float q_rank = lora_q_up->ne[0];
                            float k_rank = lora_k_up->ne[0];
                            float v_rank = lora_v_up->ne[0];
                            float m_rank = lora_v_up->ne[0];

                            float lora_q_scale = 1;
                            float lora_k_scale = 1;
                            float lora_v_scale = 1;
                            float lora_m_scale = 1;

                            if (lora_tensors.find(split_q_scale_name) != lora_tensors.end()) {
                                lora_q_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_q_scale_name]);
                                applied_lora_tensors.insert(split_q_scale_name);
                            }
                            if (lora_tensors.find(split_k_scale_name) != lora_tensors.end()) {
                                lora_k_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_k_scale_name]);
                                applied_lora_tensors.insert(split_k_scale_name);
                            }
                            if (lora_tensors.find(split_v_scale_name) != lora_tensors.end()) {
                                lora_v_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_v_scale_name]);
                                applied_lora_tensors.insert(split_v_scale_name);
                            }
                            if (lora_tensors.find(split_m_scale_name) != lora_tensors.end()) {
                                lora_m_scale = ggml_ext_backend_tensor_get_f32(lora_tensors[split_m_scale_name]);
                                applied_lora_tensors.insert(split_m_scale_name);
                            }

                            if (lora_tensors.find(split_q_alpha_name) != lora_tensors.end()) {
                                float lora_q_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_q_alpha_name]);
                                applied_lora_tensors.insert(split_q_alpha_name);
                                lora_q_scale = lora_q_alpha / q_rank;
                            }
                            if (lora_tensors.find(split_k_alpha_name) != lora_tensors.end()) {
                                float lora_k_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_k_alpha_name]);
                                applied_lora_tensors.insert(split_k_alpha_name);
                                lora_k_scale = lora_k_alpha / k_rank;
                            }
                            if (lora_tensors.find(split_v_alpha_name) != lora_tensors.end()) {
                                float lora_v_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_v_alpha_name]);
                                applied_lora_tensors.insert(split_v_alpha_name);
                                lora_v_scale = lora_v_alpha / v_rank;
                            }
                            if (lora_tensors.find(split_m_alpha_name) != lora_tensors.end()) {
                                float lora_m_alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[split_m_alpha_name]);
                                applied_lora_tensors.insert(split_m_alpha_name);
                                lora_m_scale = lora_m_alpha / m_rank;
                            }

                            ggml_scale_inplace(compute_ctx, lora_q_down, lora_q_scale);
                            ggml_scale_inplace(compute_ctx, lora_k_down, lora_k_scale);
                            ggml_scale_inplace(compute_ctx, lora_v_down, lora_v_scale);
                            ggml_scale_inplace(compute_ctx, lora_m_down, lora_m_scale);

                            // print_ggml_tensor(lora_q_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_k_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_v_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_m_down, true);  //[3072, R, 1, 1]
                            // print_ggml_tensor(lora_q_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_k_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_v_up, true);  //[R, 3072, 1, 1]
                            // print_ggml_tensor(lora_m_up, true);  //[R, 12288, 1, 1]

                            // these need to be stitched together this way:
                            //                                 |q_up,0   ,0   ,0   |
                            //                                 |0   ,k_up,0   ,0   |
                            //                                 |0   ,0   ,v_up,0   |
                            //                                 |0   ,0   ,0   ,m_up|
                            // (q_down,k_down,v_down,m_down) . (q   ,k   ,v   ,m)

                            // up_concat will be [21504, R*4, 1, 1]
                            // down_concat will be [R*4, 3072, 1, 1]

                            ggml_tensor* lora_down_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_down, lora_k_down, 1), ggml_concat(compute_ctx, lora_v_down, lora_m_down, 1), 1);
                            // print_ggml_tensor(lora_down_concat, true);  //[3072, R*4, 1, 1]

                            // this also means that if rank is bigger than 672, it is less memory efficient to do it this way (should be fine)
                            // print_ggml_tensor(lora_q_up, true);  //[3072, R, 1, 1]
                            ggml_tensor* z     = ggml_dup_tensor(compute_ctx, lora_q_up);
                            ggml_tensor* mlp_z = ggml_dup_tensor(compute_ctx, lora_m_up);
                            ggml_scale(compute_ctx, z, 0);
                            ggml_scale(compute_ctx, mlp_z, 0);
                            ggml_tensor* zz = ggml_concat(compute_ctx, z, z, 1);

                            ggml_tensor* q_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, lora_q_up, zz, 1), mlp_z, 1);
                            ggml_tensor* k_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, z, lora_k_up, 1), ggml_concat(compute_ctx, z, mlp_z, 1), 1);
                            ggml_tensor* v_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, lora_v_up, 1), mlp_z, 1);
                            ggml_tensor* m_up = ggml_concat(compute_ctx, ggml_concat(compute_ctx, zz, z, 1), lora_m_up, 1);
                            // print_ggml_tensor(q_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(k_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(v_up, true);  //[R, 21504, 1, 1]
                            // print_ggml_tensor(m_up, true);  //[R, 21504, 1, 1]

                            ggml_tensor* lora_up_concat = ggml_concat(compute_ctx, ggml_concat(compute_ctx, q_up, k_up, 0), ggml_concat(compute_ctx, v_up, m_up, 0), 0);
                            // print_ggml_tensor(lora_up_concat, true);  //[R*4, 21504, 1, 1]

                            lora_down = ggml_cont(compute_ctx, lora_down_concat);
                            lora_up   = ggml_cont(compute_ctx, lora_up_concat);

                            applied_lora_tensors.insert(split_q_u_name);
                            applied_lora_tensors.insert(split_k_u_name);
                            applied_lora_tensors.insert(split_v_u_name);
                            applied_lora_tensors.insert(split_m_u_name);

                            applied_lora_tensors.insert(split_q_d_name);
                            applied_lora_tensors.insert(split_k_d_name);
                            applied_lora_tensors.insert(split_v_d_name);
                            applied_lora_tensors.insert(split_m_d_name);
                        }
                    } else {
                        lora_up_name   = full_key + lora_ups[type] + ".weight";
                        lora_down_name = full_key + lora_downs[type] + ".weight";
                        lora_mid_name  = full_key + ".lora_mid.weight";

                        alpha_name = full_key + ".alpha";
                        scale_name = full_key + ".scale";

                        if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                            lora_up = to_f32(compute_ctx, lora_tensors[lora_up_name]);
                            applied_lora_tensors.insert(lora_up_name);
                        }

                        if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                            lora_down = to_f32(compute_ctx, lora_tensors[lora_down_name]);
                            applied_lora_tensors.insert(lora_down_name);
                        }

                        if (lora_tensors.find(lora_mid_name) != lora_tensors.end()) {
                            lora_mid = to_f32(compute_ctx, lora_tensors[lora_mid_name]);
                            applied_lora_tensors.insert(lora_mid_name);
                        }
                    }

                    if (lora_up == nullptr || lora_down == nullptr) {
                        continue;
                    }
                    // calc_scale
                    // TODO: .dora_scale?
                    int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
                    if (lora_tensors.find(scale_name) != lora_tensors.end()) {
                        scale_value = ggml_ext_backend_tensor_get_f32(lora_tensors[scale_name]);
                        applied_lora_tensors.insert(scale_name);
                    } else if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                        float alpha = ggml_ext_backend_tensor_get_f32(lora_tensors[alpha_name]);
                        scale_value = alpha / rank;
                        // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                        applied_lora_tensors.insert(alpha_name);
                    }

                    updown = ggml_ext_merge_lora(compute_ctx, lora_down, lora_up, lora_mid);
                }
                if (updown == NULL) {
                    continue;
                }

                const int64_t model_elems = ggml_nelements(model_tensor);
                const int64_t lora_elems  = ggml_nelements(updown);
                if (model_elems != lora_elems) {
                    LOG_WARN("LoRA '%s' tensor '%s' is unsupported: element count mismatch (LoRA=%lld, model=%lld); skipping",
                             file_path.c_str(),
                             model_tensor_name.c_str(),
                             (long long) lora_elems,
                             (long long) model_elems);
                    continue;
                }

                scale_value *= multiplier;
                ggml_tensor* original_tensor = model_tensor;
                if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                    model_tensor = ggml_dup_tensor(compute_ctx, model_tensor);
                    set_backend_tensor_data(model_tensor, original_tensor->data);
                }
                updown = ggml_reshape(compute_ctx, updown, model_tensor);
                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(model_tensor));
                updown = ggml_scale_inplace(compute_ctx, updown, scale_value);
                ggml_tensor* final_tensor;
                if (model_tensor->type != GGML_TYPE_F32 && model_tensor->type != GGML_TYPE_F16) {
                    final_tensor = to_f32(compute_ctx, model_tensor);
                    final_tensor = ggml_add_inplace(compute_ctx, final_tensor, updown);
                    final_tensor = ggml_cpy(compute_ctx, final_tensor, model_tensor);
                } else {
                    final_tensor = ggml_add_inplace(compute_ctx, model_tensor, updown);
                }
                ggml_build_forward_expand(gf, final_tensor);
                if (!ggml_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                    original_tensor_to_final_tensor[original_tensor] = final_tensor;
                }
                break;
            }
        }
        return gf;
    }

    void apply(std::map<std::string, ggml_tensor*> model_tensors, SDVersion version, int n_threads) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_lora_graph(model_tensors, version);
        };
        GGMLRunner::compute(get_graph, n_threads, false);
        stat();
        for (auto item : original_tensor_to_final_tensor) {
            ggml_tensor* original_tensor = item.first;
            ggml_tensor* final_tensor    = item.second;

            ggml_backend_tensor_copy(final_tensor, original_tensor);
        }
        original_tensor_to_final_tensor.clear();
        GGMLRunner::free_compute_buffer();
    }

    void stat(bool at_runntime = false) {
        size_t total_lora_tensors_count   = 0;
        size_t applied_lora_tensors_count = 0;

        for (auto& kv : lora_tensors) {
            total_lora_tensors_count++;
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                if (!at_runntime) {
                    //LOG_WARN("unused lora tensor |%s|", kv.first.c_str());
                    //print_ggml_tensor(kv.second, true);
                }
            } else {
                applied_lora_tensors_count++;
            }
        }
        /* Don't worry if this message shows up twice in the logs per LoRA,
         * this function is called once to calculate the required buffer size
         * and then again to actually generate a graph to be used */
        if (!at_runntime && applied_lora_tensors_count != total_lora_tensors_count) {
            LOG_WARN("Only (%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, total_lora_tensors_count, file_path.c_str());
        } else {
            LOG_INFO("(%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, total_lora_tensors_count, file_path.c_str());
        }
    }
};

struct MultiLoraAdapter : public WeightAdapter {
protected:
    std::vector<std::shared_ptr<LoraModel>> lora_models;

public:
    explicit MultiLoraAdapter(const std::vector<std::shared_ptr<LoraModel>>& lora_models)
        : lora_models(lora_models) {
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_tensor* weight, const std::string& weight_name, bool with_lora_and_lokr) {
        for (auto& lora_model : lora_models) {
            ggml_tensor* diff = lora_model->get_weight_diff(weight_name, ctx, weight, with_lora_and_lokr);
            if (diff == nullptr) {
                continue;
            }

            if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                weight = ggml_ext_cast_f32(ctx, weight);
            }
            weight = ggml_add(ctx, weight, diff);
        }
        return weight;
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_tensor* weight, const std::string& weight_name) override {
        return patch_weight(ctx, weight, weight_name, true);
    }

    ggml_tensor* forward_with_lora(ggml_context* ctx,
                                   ggml_tensor* x,
                                   ggml_tensor* w,
                                   ggml_tensor* b,
                                   const std::string& prefix,
                                   WeightAdapter::ForwardParams forward_params) override {
        w = patch_weight(ctx, w, prefix + "weight", false);
        if (b) {
            b = patch_weight(ctx, b, prefix + "bias", false);
        }
        ggml_tensor* out;
        if (forward_params.op_type == ForwardParams::op_type_t::OP_LINEAR) {
            out = ggml_ext_linear(ctx, x, w, b, forward_params.linear.force_prec_f32, forward_params.linear.scale);
        } else {  // OP_CONV2D
            out = ggml_ext_conv_2d(ctx,
                                   x,
                                   w,
                                   b,
                                   forward_params.conv2d.s0,
                                   forward_params.conv2d.s1,
                                   forward_params.conv2d.p0,
                                   forward_params.conv2d.p1,
                                   forward_params.conv2d.d0,
                                   forward_params.conv2d.d1,
                                   forward_params.conv2d.direct,
                                   forward_params.conv2d.circular_x,
                                   forward_params.conv2d.circular_y,
                                   forward_params.conv2d.scale);
        }
        for (auto& lora_model : lora_models) {
            ggml_tensor* out_diff = lora_model->get_out_diff(ctx, x, forward_params, prefix + "weight");
            if (out_diff == nullptr) {
                continue;
            }
            out = ggml_add_inplace(ctx, out, out_diff);
        }
        return out;
    }

    size_t get_extra_graph_size() override {
        size_t lora_tensor_num = 0;
        for (auto& lora_model : lora_models) {
            lora_tensor_num += lora_model->lora_tensors.size();
        }
        return LORA_GRAPH_BASE_SIZE + lora_tensor_num * 10;
    }
};

#endif  // __LORA_HPP__
