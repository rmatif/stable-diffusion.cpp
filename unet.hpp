#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"
#include "model.h"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 10240

class SpatialVideoTransformer : public SpatialTransformer {
protected:
    int64_t time_depth;
    int64_t max_time_embed_period;

public:
    SpatialVideoTransformer(int64_t in_channels,
                            int64_t n_head,
                            int64_t d_head,
                            int64_t depth,
                            int64_t context_dim,
                            int64_t time_depth            = 1,
                            int64_t max_time_embed_period = 10000)
        : SpatialTransformer(in_channels, n_head, d_head, depth, context_dim),
          max_time_embed_period(max_time_embed_period) {
        // We will convert unet transformer linear to conv2d 1x1 when loading the weights, so use_linear is always False
        // use_spatial_context is always True
        // merge_strategy is always learned_with_images
        // merge_factor is loaded from weights
        // time_context_dim is always None
        // ff_in is always True
        // disable_self_attn is always False
        // disable_temporal_crossattention is always False

        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(depth == time_depth);
        GGML_ASSERT(in_channels == inner_dim);

        int64_t time_mix_d_head    = d_head;
        int64_t n_time_mix_heads   = n_head;
        int64_t time_mix_inner_dim = time_mix_d_head * n_time_mix_heads;  // equal to inner_dim
        int64_t time_context_dim   = context_dim;

        for (int i = 0; i < time_depth; i++) {
            std::string name = "time_stack." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim,
                                                                                    n_time_mix_heads,
                                                                                    time_mix_d_head,
                                                                                    time_context_dim,
                                                                                    true));
        }

        int64_t time_embed_dim     = in_channels * 4;
        blocks["time_pos_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim));
        // time_pos_embed.1 is nn.SiLU()
        blocks["time_pos_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, in_channels));

        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* context,
                                int timesteps) {
        // x: [N, in_channels, h, w] aka [b*t, in_channels, h, w], t == timesteps
        // context: [N, max_position(aka n_context), hidden_size(aka context_dim)] aka [b*t, n_context, context_dim], t == timesteps
        // t_emb: [N, in_channels] aka [b*t, in_channels]
        // timesteps is num_frames
        // time_context is always None
        // image_only_indicator is always tensor([0.])
        // transformer_options is not used
        // GGML_ASSERT(ggml_n_dims(context) == 3);

        auto norm             = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in          = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out         = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);
        auto time_pos_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.0"]);
        auto time_pos_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["time_pos_embed.2"]);
        auto time_mixer       = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        GGML_ASSERT(n == timesteps);  // We compute cond and uncond separately, so batch_size==1

        auto time_context    = context;  // [b*t, n_context, context_dim]
        auto spatial_context = context;
        // time_context_first_timestep = time_context[::timesteps]
        auto time_context_first_timestep = ggml_view_3d(ctx,
                                                        time_context,
                                                        time_context->ne[0],
                                                        time_context->ne[1],
                                                        1,
                                                        time_context->nb[1],
                                                        time_context->nb[2],
                                                        0);  // [b, n_context, context_dim]
        time_context                     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                              time_context_first_timestep->ne[0],
                                                              time_context_first_timestep->ne[1],
                                                              time_context_first_timestep->ne[2] * h * w);
        time_context                     = ggml_repeat(ctx, time_context_first_timestep, time_context);  // [b*h*w, n_context, context_dim]

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n);      // [N, h * w, inner_dim]

        auto num_frames = ggml_arange(ctx, 0, timesteps, 1);
        // since b is 1, no need to do repeat
        auto t_emb = ggml_nn_timestep_embedding(ctx, num_frames, in_channels, max_time_embed_period);  // [N, in_channels]

        auto emb = time_pos_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = time_pos_embed_2->forward(ctx, emb);                   // [N, in_channels]
        emb      = ggml_reshape_3d(ctx, emb, emb->ne[0], 1, emb->ne[1]);  // [N, 1, in_channels]

        for (int i = 0; i < depth; i++) {
            std::string transformer_name = "transformer_blocks." + std::to_string(i);
            std::string time_stack_name  = "time_stack." + std::to_string(i);

            auto block     = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[transformer_name]);
            auto mix_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[time_stack_name]);

            x = block->forward(ctx, x, spatial_context);  // [N, h * w, inner_dim]

            // in_channels == inner_dim
            auto x_mix = x;
            x_mix      = ggml_add(ctx, x_mix, emb);  // [N, h * w, inner_dim]

            int64_t N = x_mix->ne[2];
            int64_t T = timesteps;
            int64_t B = N / T;
            int64_t S = x_mix->ne[1];
            int64_t C = x_mix->ne[0];

            x_mix = ggml_reshape_4d(ctx, x_mix, C, S, T, B);               // (b t) s c -> b t s c
            x_mix = ggml_cont(ctx, ggml_permute(ctx, x_mix, 0, 2, 1, 3));  // b t s c -> b s t c
            x_mix = ggml_reshape_3d(ctx, x_mix, C, T, S * B);              // b s t c -> (b s) t c

            x_mix = mix_block->forward(ctx, x_mix, time_context);  // [B * h * w, T, inner_dim]

            x_mix = ggml_reshape_4d(ctx, x_mix, C, T, S, B);               // (b s) t c -> b s t c
            x_mix = ggml_cont(ctx, ggml_permute(ctx, x_mix, 0, 2, 1, 3));  // b s t c -> b t s c
            x_mix = ggml_reshape_3d(ctx, x_mix, C, S, T * B);              // b t s c -> (b t) s c

            x = time_mixer->forward(ctx, x, x_mix);  // [N, h * w, inner_dim]
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n);       // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock : public GGMLBlock {
protected:
    static std::map<std::string, enum ggml_type> empty_tensor_types;
    SDVersion version = VERSION_SD1;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_SD2, 2048 for VERSION_SDXL

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_SDXL/SVD

    UnetModelBlock(SDVersion version = VERSION_SD1, std::map<std::string, enum ggml_type>& tensor_types = empty_tensor_types, bool flash_attn = false)
        : version(version) {
        if (sd_version_is_sd2(version)) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (sd_version_is_sdxl(version)) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
        } else if (version == VERSION_SVD) {
            in_channels       = 8;
            out_channels      = 4;
            context_dim       = 1024;
            adm_in_channels   = 768;
            num_head_channels = 64;
            num_heads         = -1;
        }
        if (sd_version_is_inpaint(version)) {
            in_channels = 9;
        }

        // dims is always 2
        // use_temporal_attention is always True for SVD

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (sd_version_is_sdxl(version) || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            // label_emb_1 is nn.SiLU()
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            if (version == VERSION_SVD) {
                return new VideoResBlock(channels, emb_channels, out_channels);
            } else {
                return new ResBlock(channels, emb_channels, out_channels);
            }
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            if (version == VERSION_SVD) {
                return new SpatialVideoTransformer(in_channels, n_head, d_head, depth, context_dim);
            } else {
                return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim, flash_attn);
            }
        };

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
        blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  transformer_depth[transformer_depth.size() - 1],
                                                                                  context_dim));
        blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch                = mult * model_channels;
                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, transformer_depth[i], context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        blocks["out.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(ch));  // ch == model_channels
        // out_1 is nn.SiLU()
        blocks["out.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* resblock_forward(std::string name,
                                         struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* emb,
                                         int num_video_frames) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<VideoResBlock>(blocks[name]);

            return block->forward(ctx, x, emb, num_video_frames);
        } else {
            auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);

            return block->forward(ctx, x, emb);
        }
    }

    struct ggml_tensor* attention_layer_forward(std::string name,
                                                struct ggml_context* ctx,
                                                struct ggml_tensor* x,
                                                struct ggml_tensor* context,
                                                int timesteps) {
        if (version == VERSION_SVD) {
            auto block = std::dynamic_pointer_cast<SpatialVideoTransformer>(blocks[name]);

            return block->forward(ctx, x, context, timesteps);
        } else {
            auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);

            return block->forward(ctx, x, context);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                 struct ggml_tensor* context,
                                 struct ggml_tensor* c_concat              = NULL,
                                 struct ggml_tensor* y                     = NULL,
                                 int num_video_frames                      = -1,
                                 std::vector<struct ggml_tensor*> controls = {},
                                 float control_strength                    = 0.f,
                                 bool clockwork_is_adaptor_pass            = false,
                                 ggml_tensor* clockwork_input_cache        = NULL,
                                 ggml_tensor** clockwork_output_cache_ptr  = NULL) {
        // x: [N, in_channels, h, w] or [N, in_channels/2, h, w]
        // timesteps: [N,]
        // context: [N, max_position, hidden_size] or [1, max_position, hidden_size]. for example, [N, 77, 768]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        // return: [N, out_channels, h, w]
        if (context != NULL) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx, context, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (c_concat != NULL) {
            if (c_concat->ne[3] != x->ne[3]) {
                c_concat = ggml_repeat(ctx, c_concat, x);
            }
            x = ggml_concat(ctx, x, c_concat, 2);
        }

        if (y != NULL) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx, y, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

        auto t_emb = ggml_nn_timestep_embedding(ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != NULL) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);

        // ControlNet application to middle block (only in full pass)
        if (!clockwork_is_adaptor_pass && controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[controls.size() - 1], control_strength);
            h       = ggml_add(ctx, h, cs);  // middle control
        }
        int control_offset = controls.size() > 0 ? (controls.size() - 2) : -1;


        // output_blocks
        int output_block_idx = 0;
        int ds_out           = 1; // ds for output blocks, effectively tracks current resolution scale factor from highest res
                                 // Starts at 1 (e.g. 64x64 for 512 input), then becomes 2 (32x32), 4 (16x16), 8 (8x8) as i decreases.
                                 // This is used to match attention_resolutions.
                                 // The ds variable from input_blocks loop is not directly usable here as it ends at max downsample.
                                 // Instead, we recalculate based on `i`. For SD1.x, last `i` is 0 (ds_full was 1).
                                 // Middle block operates at max ds (e.g., 8).
                                 // First `i` in output_blocks is `len_mults - 1`. Max `ds` is `2^(len_mults-1)`.
        ds_out = 1 << ((int)channel_mult.size() -1);


        for (int i = (int)channel_mult.size() - 1; i >= 0; i--) { // i = 3,2,1,0 for SD1.x
            // Clockwork Adaptor Pass Logic for skipping initial upsampling stages
            if (clockwork_is_adaptor_pass) {
                // Equivalent to self.unet.up_blocks = self.up_blocks[-2:]
                // This means we skip output blocks for i = len_mults-1 down to i = 2.
                // For SD1.x (len_mults=4), this skips i=3 and i=2.
                if (i > 1) { // Skips i=3 (up_blocks[0]) and i=2 (up_blocks[1])
                    if (!hs.empty()) { // Pop corresponding skip connections if they were (mistakenly) fully populated
                        for(int k=0; k < num_res_blocks + 1; ++k) if(!hs.empty()) hs.pop_back();
                    }
                    ds_out /=2; // Keep ds_out synchronized
                    output_block_idx += (num_res_blocks + 1); // Advance block index notionally
                    continue;
                }
                if (i == 1) { // This is the up_blocks[-2] slot, replaced by adaptor.
                              // h becomes the cached feature.
                    GGML_ASSERT(clockwork_input_cache != NULL);
                    h = clockwork_input_cache;
                    // Original ResBlocks, Attn, and Upsampler for this stage are skipped.
                    // Their skip connections from `hs` are also effectively skipped.
                    // Since hs only contains one element in adaptor mode, popping is not needed here for these.
                    ds_out /=2; // Keep ds_out synchronized as if upsampling occurred.
                    output_block_idx += (num_res_blocks + 1); // Advance block index notionally
                    continue; // Proceed to the next stage (i=0)
                }
            }

            // Common path for a stage (full pass, or i=0 in adaptor pass)
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back(); hs.pop_back();

                if (controls.size() > 0 && control_offset >= 0) {
                     if (!clockwork_is_adaptor_pass || (clockwork_is_adaptor_pass && i==0)) { // Apply control to non-skipped stages
                        auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
                        h_skip  = ggml_add(ctx, h_skip, cs);
                     }
                    control_offset--;
                }

                h = ggml_concat(ctx, h, h_skip, 2);

                std::string name_res = "output_blocks." + std::to_string(output_block_idx) + ".0";
                h = resblock_forward(name_res, ctx, h, emb, num_video_frames);

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds_out) != attention_resolutions.end()) {
                    std::string name_attn = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    h = attention_layer_forward(name_attn, ctx, h, context, num_video_frames);
                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) { // Upsampling for this level
                    std::string name_up = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    auto block_up       = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name_up]);
                    h = block_up->forward(ctx, h);
                }
                output_block_idx += 1;
            }

            // Cache point: After processing for i=1 (up_blocks[-2]) in a full pass
            if (!clockwork_is_adaptor_pass && i == 1 && clockwork_output_cache_ptr != NULL) {
                // Ensure *clockwork_output_cache_ptr is a tensor in a persistent context.
                // Here, we are in compute_ctx. We need to copy h to *clockwork_output_cache_ptr.
                if (*clockwork_output_cache_ptr == NULL) {
                     // This allocation should ideally happen once outside the unet call, in a persistent context.
                     // For simplicity now, assume it's pre-allocated matching h's shape & type.
                     // Or, the runner handles allocation and passes a valid tensor.
                     // Let's assume runner ensures *clockwork_output_cache_ptr is valid.
                }
                // Create a cpy operation if *clockwork_output_cache_ptr is in a different backend/context
                // For now, direct data copy if same context, or rely on runner to handle cross-context copy.
                // This is tricky without knowing the context of *clockwork_output_cache_ptr.
                // Simplest for now: if runner wants to cache, it receives `h` and dups it.
                // So, this specific assignment here is more conceptual.
                // The actual caching will be managed by UNetModelRunner using the returned `h`
                // if it was a caching step.
                // For now, this means we need a way for UnetModelBlock::forward to signal this `h`.
                // Alternative: UNetModelRunner uses ggml_graph_get_tensor(gf, "cache_point_tensor_name")
                // Let's go with the `clockwork_output_cache_ptr` being a ggml_tensor* that UNetModelBlock writes to.
                // The runner ensures this tensor is in the correct (persistent) context.
                // This means `ggml_build_forward_expand(gf, ggml_cpy(compute_ctx, h, *clockwork_output_cache_ptr))`
                // should be added to the graph by the runner after `unet.forward` returns the final `h`.
                // This tensor `h` is the one we want to cache. Name it.
                ggml_set_name(h, "clockwork_cache_point_for_up_blocks_minus_2");
            }
            if (i > 0) { // if not the highest-res layer, ds_out is halved for the next, higher-res layer
                ds_out /= 2;
            }
        }

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "unet_output_node"); // Name the final output
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }
};

class UnetModelBlock;

struct UNetModelRunner : public GGMLRunner {
    UnetModelBlock unet;

    // Clockwork Diffusion parameters
    struct ClockworkParams {
        bool is_adaptor_pass;
        ggml_tensor* input_cache; 
        ggml_tensor** output_cache_target_ptr;

        ClockworkParams() : is_adaptor_pass(false), input_cache(NULL), output_cache_target_ptr(NULL) {}
    };
    ggml_tensor* clockwork_cached_features_ = NULL; // Stores the cached features
    int clockwork_time_step_ = 0;
    int clockwork_clock_config_ = 0; // Configured clock value (0 = disabled)
    struct ggml_context* clockwork_cache_ctx_ = NULL; // Context for clockwork_cached_features_

    UNetModelRunner(ggml_backend_t backend,
                    std::map<std::string, enum ggml_type>& tensor_types,
                    const std::string prefix,
                    SDVersion version = VERSION_SD1,
                    bool flash_attn   = false,
                    int clockwork_clock_val = 0) // Added clockwork_clock_val
        : GGMLRunner(backend), unet(version, tensor_types, flash_attn), clockwork_clock_config_(clockwork_clock_val) {
        unet.init(params_ctx, tensor_types, prefix);
        if (clockwork_clock_config_ > 0) {
            // Initialize context for storing cached features if clockwork is active
            // This context needs to persist across compute calls.
            // Using params_ctx for this might bloat it if cache is large.
            // A dedicated small context is better.
            struct ggml_init_params p_params;
            p_params.mem_size   = 256 * 1024 * 1024; // Estimate for cache, adjust as needed
            p_params.mem_buffer = NULL;
            p_params.no_alloc   = false; // We need to allocate the tensor here
            clockwork_cache_ctx_ = ggml_init(p_params);
            GGML_ASSERT(clockwork_cache_ctx_ != NULL);
            // clockwork_cached_features_ will be allocated on first full pass based on actual shape.
        }
    }

    ~UNetModelRunner() {
        if (clockwork_cache_ctx_ != NULL) {
            ggml_free(clockwork_cache_ctx_);
            clockwork_cache_ctx_ = NULL;
            // clockwork_cached_features_ is part of this context, so it's freed.
        }
    }

    std::string get_desc() {
        return "unet";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                     struct ggml_tensor* c_concat              = NULL,
                                     struct ggml_tensor* y                     = NULL,
                                     int num_video_frames                      = -1,
                                     std::vector<struct ggml_tensor*> controls = {},
                                     float control_strength                    = 0.f,
                                     const ClockworkParams& clockwork_params   = ClockworkParams()) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, UNET_GRAPH_SIZE, false);

        if (num_video_frames == -1) {
            num_video_frames = x->ne[3];
        }

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);
        c_concat  = to_backend(c_concat);

        for (int i = 0; i < controls.size(); i++) {
            controls[i] = to_backend(controls[i]);
        }

        struct ggml_tensor* out = unet.forward(compute_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               c_concat,
                                                y,
                                                num_video_frames,
                                                controls,
                                                control_strength,
                                                clockwork_params.is_adaptor_pass,
                                                clockwork_params.input_cache,
                                                clockwork_params.output_cache_target_ptr);

        ggml_build_forward_expand(gf, out); // Builds graph for the main UNet output `out`

        // Clockwork: If this was a full pass and caching is enabled, add cpy op to graph
        if (clockwork_params.output_cache_target_ptr != NULL && !clockwork_params.is_adaptor_pass) {
            // struct ggml_tensor* h_to_cache_node = NULL; // Use get_tensor_from_graph instead
            struct ggml_tensor* h_to_cache_node = get_tensor_from_graph(gf, "clockwork_cache_point_for_up_blocks_minus_2");

            if (h_to_cache_node == NULL) {
                LOG_WARN("Clockwork cache point tensor 'clockwork_cache_point_for_up_blocks_minus_2' not found in graph. Caching might fail.");
            }

            if (h_to_cache_node != NULL) {
                if (*clockwork_params.output_cache_target_ptr == NULL && clockwork_cache_ctx_ != NULL) {
                    // Allocate persistent cache tensor for the first time
                    *clockwork_params.output_cache_target_ptr = ggml_dup_tensor(clockwork_cache_ctx_, h_to_cache_node);
                    ggml_set_name(*clockwork_params.output_cache_target_ptr, "clockwork_persistent_cache_tensor");
                    // If the clockwork_cache_ctx_ is associated with a non-CPU backend, 
                    // the tensor duplicated into it needs to be properly managed by that backend.
                    // ggml_dup_tensor should handle basic context association.
                    // If clockwork_cache_ctx_ is just a CPU context, no special backend handling is needed here for the tensor itself.
                    // The ggml_cpy operation will handle data transfer between backends if compute_ctx is on GPU.
                }
                GGML_ASSERT(*clockwork_params.output_cache_target_ptr != NULL && "Persistent cache tensor is NULL.");
                GGML_ASSERT(ggml_are_same_shape(h_to_cache_node, *clockwork_params.output_cache_target_ptr) && "Cache tensor shape mismatch.");

                struct ggml_tensor* cpy_to_cache_op = ggml_cpy(compute_ctx, h_to_cache_node, *clockwork_params.output_cache_target_ptr);
                // This cpy_op also needs to be part of the graph `gf` to be scheduled and computed.
                ggml_build_forward_expand(gf, cpy_to_cache_op); 
                // Now, `cpy_to_cache_op` is the last node added to the graph.
                // The `GGMLRunner::compute` logic must retrieve the main output by its specific name.
            } else {
                 LOG_WARN("Failed to find tensor for caching. Cache will not be updated this step.");
            }
        }
        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                  float control_strength                    = 0.f,
                  struct ggml_tensor** output               = NULL,
                  struct ggml_context* output_ctx           = NULL,
                  ClockworkParams clockwork_runtime_params  = ClockworkParams()) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 77, 768]) or [1, max_position, hidden_size]
        // c_concat: [N, in_channels, h, w] or [1, in_channels, h, w]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        // y: [N, adm_in_channels] or [1, adm_in_channels]

        ClockworkParams current_clockwork_params = clockwork_runtime_params; // Use passed in params

        if (clockwork_clock_config_ > 0) {
            current_clockwork_params.is_adaptor_pass = (clockwork_time_step_ % clockwork_clock_config_ != 0);
            if (current_clockwork_params.is_adaptor_pass) {
                current_clockwork_params.input_cache = clockwork_cached_features_;
                GGML_ASSERT(current_clockwork_params.input_cache != NULL);
                current_clockwork_params.output_cache_target_ptr = NULL;
            } else { // Full pass
                current_clockwork_params.input_cache = NULL;
                // output_cache_target_ptr will point to clockwork_cached_features_
                // Allocation/reallocation of clockwork_cached_features_ might be needed if shape changes or first time.
                // For now, assume shape is constant. UnetModelBlock will fill it via ggml_cpy.
                // This tensor needs to be created in clockwork_cache_ctx_.
                // The ggml_cpy op will be added to the graph by UnetModelBlock.
                // For SD, latent shape is usually fixed. Let's allocate it here if null.
                if (clockwork_cached_features_ == NULL && clockwork_cache_ctx_ != NULL) {
                    // Determine shape of up_blocks[-2] output. This is hard without running part of the model.
                    // For SD1.x (512x512 input -> 64x64 latent), up_blocks[-2] (i=1) outputs
                    // model_channels * channel_mult[1] (e.g. 320*2=640) channels, at 32x32 (latent H/2, W/2).
                    // This is an approximation. A more robust way is to get shape after first full run.
                    // For simplicity, let's assume it is pre-created or UnetModelBlock handles it.
                    // The UnetModelBlock will perform ggml_cpy into this tensor if output_cache_target_ptr is set.
                    // This tensor must exist and be in a compatible context for ggml_cpy.

                    // Placeholder: expect UnetModelBlock to handle ggml_dup into this if NULL initially,
                    // or require it to be pre-sized.
                    // For now, let UnetModelBlock ggml_dup to the target if it's a full pass and target_ptr is valid.
                    // The runner is responsible for managing the lifecycle of the *clockwork_cached_features_ tensor itself.
                    // Let's assume that if *output_cache_target_ptr is NULL, UnetModelBlock will allocate it
                    // using ggml_dup(ctx, h_to_cache); and the runner will take ownership.
                    // This means clockwork_cached_features_ might point to a tensor in compute_ctx,
                    // which is bad if compute_ctx is freed.
                    // Safest: UnetModelBlock uses output_cache_target_ptr to fill an existing tensor.
                    // Runner allocates clockwork_cached_features_ in its clockwork_cache_ctx_.
                    // TODO: This logic needs to be robust for first cache creation.
                    // For now, UnetModelBlock's ggml_cpy will write to *current_clockwork_params.output_cache_target_ptr
                    // which is &clockwork_cached_features_.
                }
                current_clockwork_params.output_cache_target_ptr = &clockwork_cached_features_;
            }
        }


        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength, current_clockwork_params);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);

        if (clockwork_clock_config_ > 0) {
            clockwork_time_step_++;
            if (!current_clockwork_params.is_adaptor_pass && current_clockwork_params.output_cache_target_ptr != NULL) {
                // If clockwork_cached_features_ was just updated (it was a full pass)
                // and if it was allocated for the first time by UnetModelBlock via dup into compute_ctx,
                // we need to ensure it's moved to clockwork_cache_ctx_.
                // The current mechanism has UnetModelBlock doing ggml_cpy to an existing tensor.
                // So, the runner needs to ensure clockwork_cached_features_ exists in clockwork_cache_ctx_.
                // This part is complex to get right for the first allocation.
                // A simpler approach: On first full pass, UnetModelBlock returns the tensor to be cached.
                // Runner dups it into clockwork_cache_ctx_ and stores it.
                // For subsequent full passes, UnetModelBlock copies into existing tensor.
                // For now, assuming the ggml_cpy in UnetModelBlock::forward handles writing to the persistent tensor.
            }
        }
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);

        {
            // CPU, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CUDA, num_video_frames = 1, x{num_video_frames, 8, 8, 8}: Pass
            // CPU, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: Wrong result
            // CUDA, num_video_frames = 3, x{num_video_frames, 8, 8, 8}: nan
            int num_video_frames = 3;

            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 8, num_video_frames);
            std::vector<float> timesteps_vec(num_video_frames, 999.f);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            ggml_set_f32(x, 0.5f);
            // print_ggml_tensor(x);

            auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 1024, 1, num_video_frames);
            ggml_set_f32(context, 0.5f);
            // print_ggml_tensor(context);

            auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 768, num_video_frames);
            ggml_set_f32(y, 0.5f);
            // print_ggml_tensor(y);

            struct ggml_tensor* out = NULL;

            int t0 = ggml_time_ms();
            compute(8, x, timesteps, context, NULL, y, num_video_frames, {}, 0.f, &out, work_ctx);
            int t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("unet test done in %dms", t1 - t0);
        }
    }
};

#endif  // __UNET_HPP__
