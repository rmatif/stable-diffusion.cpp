#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "ggml_extend.hpp"
#include <vector> // For std::vector
#include <mutex>  // For std::mutex if thread safety for banks is needed (complex)

// Forward declare ggml_tensor if not already included by ggml_extend.hpp in a way that makes it visible
struct ggml_tensor;
struct ggml_context;

// Enum for Reference Attention Mode
// To be used by U-Net blocks to determine behavior
enum RefAttnMode {
    REF_ATTN_NORMAL, // Standard operation
    REF_ATTN_WRITE,  // Write to attention bank
    REF_ATTN_READ    // Read from attention bank
};

// Struct for Reference Attention options, passed down to blocks
struct ReferenceOptions_ggml {
    float attn_style_fidelity = 0.5f;
    float attn_strength = 1.0f;
    // bool enabled = false; // Implicitly handled by passing valid ref_opts or REF_ATTN_NORMAL mode
};


class DownSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;
    bool vae_downsample;

public:
    DownSampleBlock(int channels,
                    int out_channels,
                    bool vae_downsample = false)
        : channels(channels),
          out_channels(out_channels),
          vae_downsample(vae_downsample) {
        if (vae_downsample) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {0, 0}));
        } else {
            blocks["op"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {2, 2}, {1, 1}));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_pad(ctx, x, 1, 1, 0, 0);
            x = conv->forward(ctx, x);
        } else {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["op"]);

            x = conv->forward(ctx, x);
        }
        return x;  // [N, out_channels, h/2, w/2]
    }
};

class UpSampleBlock : public GGMLBlock {
protected:
    int channels;
    int out_channels;

public:
    UpSampleBlock(int channels,
                  int out_channels)
        : channels(channels),
          out_channels(out_channels) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);    // [N, out_channels, h*2, w*2]
        return x;
    }
};

class ResBlock : public GGMLBlock {
protected:
    // network hparams
    int64_t channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int64_t emb_channels;  // time_embed_dim
    int64_t out_channels;  // mult * model_channels
    std::pair<int, int> kernel_size;
    int dims;
    bool skip_t_emb;
    bool exchange_temb_dims;

    std::shared_ptr<GGMLBlock> conv_nd(int dims,
                                       int64_t in_channels,
                                       int64_t out_channels,
                                       std::pair<int, int> kernel_size,
                                       std::pair<int, int> padding) {
        GGML_ASSERT(dims == 2 || dims == 3);
        if (dims == 3) {
            return std::shared_ptr<GGMLBlock>(new Conv3dnx1x1(in_channels, out_channels, kernel_size.first, 1, padding.first));
        } else {
            return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, {1, 1}, padding));
        }
    }

public:
    ResBlock(int64_t channels,
             int64_t emb_channels,
             int64_t out_channels,
             std::pair<int, int> kernel_size = {3, 3},
             int dims                        = 2,
             bool exchange_temb_dims         = false,
             bool skip_t_emb                 = false)
        : channels(channels),
          emb_channels(emb_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          dims(dims),
          skip_t_emb(skip_t_emb),
          exchange_temb_dims(exchange_temb_dims) {
        std::pair<int, int> padding = {kernel_size.first / 2, kernel_size.second / 2};
        blocks["in_layers.0"]       = std::shared_ptr<GGMLBlock>(new GroupNorm32(channels));
        // in_layer_1 is nn.SILU()
        blocks["in_layers.2"] = conv_nd(dims, channels, out_channels, kernel_size, padding);

        if (!skip_t_emb) {
            // emb_layer_0 is nn.SILU()
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(emb_channels, out_channels));
        }

        blocks["out_layers.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        // out_layer_1 is nn.SILU()
        // out_layer_2 is nn.Dropout(), skip for inference
        blocks["out_layers.3"] = conv_nd(dims, out_channels, out_channels, kernel_size, padding);

        if (out_channels != channels) {
            blocks["skip_connection"] = conv_nd(dims, channels, out_channels, {1, 1}, {0, 0});
        }
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb = NULL) {
        // For dims==3, we reduce dimension from 5d to 4d by merging h and w, in order not to change ggml
        // [N, c, t, h, w] => [N, c, t, h * w]
        // x: [N, channels, h, w] if dims == 2 else [N, channels, t, h, w]
        // emb: [N, emb_channels] if dims == 2 else [N, t, emb_channels]
        auto in_layers_0  = std::dynamic_pointer_cast<GroupNorm32>(blocks["in_layers.0"]);
        auto in_layers_2  = std::dynamic_pointer_cast<UnaryBlock>(blocks["in_layers.2"]);
        auto out_layers_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out_layers.0"]);
        auto out_layers_3 = std::dynamic_pointer_cast<UnaryBlock>(blocks["out_layers.3"]);

        if (emb == NULL) {
            GGML_ASSERT(skip_t_emb);
        }

        // in_layers
        auto h = in_layers_0->forward(ctx, x);
        h      = ggml_silu_inplace(ctx, h);
        h      = in_layers_2->forward(ctx, h);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]

        // emb_layers
        if (!skip_t_emb) {
            auto emb_layer_1 = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);

            auto emb_out = ggml_silu(ctx, emb);
            emb_out      = emb_layer_1->forward(ctx, emb_out);  // [N, out_channels] if dims == 2 else [N, t, out_channels]

            if (dims == 2) {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]
            } else {
                emb_out = ggml_reshape_4d(ctx, emb_out, 1, emb_out->ne[0], emb_out->ne[1], emb_out->ne[2]);  // [N, t, out_channels, 1]
                if (exchange_temb_dims) {
                    // emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                    emb_out = ggml_cont(ctx, ggml_permute(ctx, emb_out, 0, 2, 1, 3));  // [N, out_channels, t, 1]
                }
            }

            h = ggml_add(ctx, h, emb_out);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        // out_layers
        h = out_layers_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = out_layers_3->forward(ctx, h);

        // skip connection
        if (out_channels != channels) {
            auto skip_connection = std::dynamic_pointer_cast<UnaryBlock>(blocks["skip_connection"]);
            x                    = skip_connection->forward(ctx, x);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
    }
};

class GEGLU : public GGMLBlock {
protected:
    int64_t dim_in;
    int64_t dim_out;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
        enum ggml_type wtype      = (tensor_types.find(prefix + "proj.weight") != tensor_types.end()) ? tensor_types[prefix + "proj.weight"] : GGML_TYPE_F32;
        enum ggml_type bias_wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "proj.bias") != tensor_types.end()) ? tensor_types[prefix + "proj.bias"] : GGML_TYPE_F32;
        params["proj.weight"]     = ggml_new_tensor_2d(ctx, wtype, dim_in, dim_out * 2);
        params["proj.bias"]       = ggml_new_tensor_1d(ctx, bias_wtype, dim_out * 2);
    }

public:
    GEGLU(int64_t dim_in, int64_t dim_out)
        : dim_in(dim_in), dim_out(dim_out) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        struct ggml_tensor* w = params["proj.weight"];
        struct ggml_tensor* b = params["proj.bias"];

        auto x_w    = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], 0);                        // [dim_out, dim_in]
        auto x_b    = ggml_view_1d(ctx, b, b->ne[0] / 2, 0);                                            // [dim_out, dim_in]
        auto gate_w = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], w->nb[1] * w->ne[1] / 2);  // [dim_out, ]
        auto gate_b = ggml_view_1d(ctx, b, b->ne[0] / 2, b->nb[0] * b->ne[0] / 2);                      // [dim_out, ]

        auto x_in = x;
        x         = ggml_nn_linear(ctx, x_in, x_w, x_b);        // [ne3, ne2, ne1, dim_out]
        auto gate = ggml_nn_linear(ctx, x_in, gate_w, gate_b);  // [ne3, ne2, ne1, dim_out]

        gate = ggml_gelu_inplace(ctx, gate);

        x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, dim_out]

        return x;
    }
};

class FeedForward : public GGMLBlock {
public:
    FeedForward(int64_t dim,
                int64_t dim_out,
                int64_t mult = 4) {
        int64_t inner_dim = dim * mult;

        blocks["net.0"] = std::shared_ptr<GGMLBlock>(new GEGLU(dim, inner_dim));
        // net_1 is nn.Dropout(), skip for inference
        blocks["net.2"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim_out));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim]
        // return: [ne3, ne2, ne1, dim_out]

        auto net_0 = std::dynamic_pointer_cast<GEGLU>(blocks["net.0"]);
        auto net_2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

        x = net_0->forward(ctx, x);  // [ne3, ne2, ne1, inner_dim]
        x = net_2->forward(ctx, x);  // [ne3, ne2, ne1, dim_out]
        return x;
    }
};

class CrossAttention : public GGMLBlock {
protected:
    int64_t query_dim;
    int64_t context_dim; // Can be query_dim for self-attention
    int64_t n_head;
    int64_t d_head;
    bool flash_attn;

    // Reference Attention Bank (specific to this CrossAttention instance)
    // This is a simplified bank; Python version might have more complex storage per-block.
    // Stores the 'n' tensor (normalized input to self-attention) from the WRITE pass.
    // This needs to be managed carefully (cleared per step, potentially per batch item if batch_size > 1).
    // For simplicity, this example assumes batch_size = 1 for reference parts.
    // `std::vector` might not be ideal if these are large tensors; consider ggml_context for bank storage.
    std::vector<ggml_tensor*> attention_bank_n_tensors; // Stores copies of 'n'
                                                        // In a more robust system, this would be a ggml_context
                                                        // and tensors would be ggml_dup'd into it.
                                                        // For now, using a vector of pointers that need careful lifetime management.
                                                        // These should be copies, not direct pointers to compute_ctx tensors.

public:
    CrossAttention(int64_t query_dim,
                   int64_t context_dim, // For self-attention, context_dim == query_dim
                   int64_t n_head,
                   int64_t d_head,
                   bool flash_attn = false)
        : n_head(n_head),
          d_head(d_head),
          query_dim(query_dim),
          context_dim(context_dim),
          flash_attn(flash_attn) {
        int64_t inner_dim = d_head * n_head;

        blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, false));
        blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(this->context_dim, inner_dim, false));
        blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(this->context_dim, inner_dim, false));

        blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim));
        // to_out_1 is nn.Dropout(), skip for inference
    }

    // Method to clear the attention bank for this CrossAttention instance
    void clear_bank() {
        // If bank tensors were ggml_dup'd into a specific ggml_context, that context would be freed.
        // If they are just pointers or shallow copies, this is more complex.
        // For this simplified example, just clear the vector.
        // Proper memory management of banked tensors is crucial.
        for (ggml_tensor* t : attention_bank_n_tensors) {
            // If these were allocated with ggml_new_tensor in a persistent bank_ctx,
            // freeing bank_ctx would handle them. If they are copies in work_ctx,
            // they are freed when work_ctx is freed.
            // This is a placeholder; actual freeing depends on allocation strategy.
        }
        attention_bank_n_tensors.clear();
    }


    // The forward method now needs to handle RefAttnMode
    struct ggml_tensor* forward(struct ggml_context* ctx,                // Current compute context
                                struct ggml_tensor* x_query,             // Input query tensor [N, n_token, query_dim]
                                struct ggml_tensor* k_context,           // Key context tensor [N, n_k_token, context_dim_k]
                                struct ggml_tensor* v_context,           // Value context tensor [N, n_v_token, context_dim_v] (often same as k_context)
                                RefAttnMode ref_attn_mode = REF_ATTN_NORMAL, // Reference attention mode (mostly for logging/debugging here)
                                const ReferenceOptions_ggml* ref_opts = nullptr // Reference options (mostly for logging/debugging here)
                                ) {
        auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
        auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
        auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

        int64_t n_batch   = x_query->ne[2]; // Batch size
        int64_t n_token   = x_query->ne[1]; // Query sequence length
        int64_t inner_dim = d_head * n_head;

        ggml_tensor* q_proj = to_q->forward(ctx, x_query); // [N, n_token, inner_dim]

        // k_context and v_context are now passed directly
        ggml_tensor* k_proj = to_k->forward(ctx, k_context); // [N, n_k_token, inner_dim]
        ggml_tensor* v_proj = to_v->forward(ctx, v_context); // [N, n_v_token, inner_dim]


        // Reference Attention Logic (only for self-attention, when k_context and v_context originated from x_query's source)
        // The decision to augment k_context/v_context for REF_ATTN_READ happens *before* calling this CrossAttention::forward.
        // So, this block doesn't need to handle the REF_ATTN_WRITE/READ modes for banking/retrieval itself.
        // It just consumes the k_context and v_context it's given.
        // The logging/debugging for ref_attn_mode can remain if useful.
        if (ref_attn_mode == REF_ATTN_READ && ref_opts != nullptr) {
            LOG_DEBUG("CrossAttention processing in READ mode with provided K,V contexts.");
        } else if (ref_attn_mode == REF_ATTN_WRITE && ref_opts != nullptr) {
            LOG_DEBUG("CrossAttention processing in WRITE mode (standard ops, banking handled by caller).");
        }
        // This was the problematic section. It should be within the forward method.
        ggml_tensor* attn_output_calc = ggml_nn_attention_ext(ctx, q_proj, k_proj, v_proj, n_head, NULL, false, false, flash_attn); // [N, n_token, inner_dim]
        attn_output_calc = to_out_0->forward(ctx, attn_output_calc);  // [N, n_token, query_dim]
        return attn_output_calc;
    } // This is the correct closing brace for CrossAttention::forward
}; // This closing brace is for the CrossAttention class itself.

class BasicTransformerBlock : public GGMLBlock {
protected:
    int64_t n_head;
    int64_t d_head;
    bool ff_in;
    // Reference Attention: Store the 'n' (normalized x) for this block if in WRITE mode.
    // This is a conceptual per-block bank. In ComfyUI, it's part of an "injection_holder".
    // This will be managed (filled/cleared) by the UNetModel or DiffusionModel for the current step.
    // For simplicity, this pointer will be set externally if banking is active for this block.
    ggml_tensor* banked_n_for_read_mode = nullptr; // Points to the single banked tensor relevant for this block in READ mode.
                                                   // This would be fetched from a global bank.

public:
    BasicTransformerBlock(int64_t dim,
                          int64_t n_head,
                          int64_t d_head,
                          int64_t context_dim,
                          bool ff_in      = false,
                          bool flash_attn = false)
        : n_head(n_head), d_head(d_head), ff_in(ff_in) {
        blocks["attn1"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, dim, n_head, d_head, flash_attn)); // Self-attention
        blocks["attn2"] = std::shared_ptr<GGMLBlock>(new CrossAttention(dim, context_dim, n_head, d_head, flash_attn)); // Cross-attention
        blocks["ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim));
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));

        if (ff_in) {
            blocks["norm_in"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
            blocks["ff_in"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim));
        }
    }

    // Called by UnetModel to clear any banked 'n' for this specific block after a step.
    // Or to set the banked_n_for_read_mode before a read pass.
    void set_banked_n_for_read_pass(ggml_tensor* banked_n) {
        this->banked_n_for_read_mode = banked_n; // External code manages lifetime of banked_n
    }
    void clear_banked_n_after_read_pass() {
        this->banked_n_for_read_mode = nullptr;
    }


    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,         // Input tensor
                                struct ggml_tensor* context,   // Cross-attention context
                                RefAttnMode ref_attn_mode = REF_ATTN_NORMAL,
                                const ReferenceOptions_ggml* ref_opts = nullptr,
                                // This is where the 'banked_n_for_this_block' would be passed if different per call
                                // Or, it's managed by set_banked_n_for_read_pass
                                ggml_tensor** n_to_bank = nullptr // Output param for WRITE mode
                                ) {
        auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
        auto attn2 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn2"]);
        auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
        auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto norm3 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);

        if (ff_in) {
            auto norm_in = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_in"]);
            auto ff_in_block   = std::dynamic_pointer_cast<FeedForward>(blocks["ff_in"]); // Renamed

            auto x_skip = x;
            x           = norm_in->forward(ctx, x);
            x           = ff_in_block->forward(ctx, x);
            x = ggml_add(ctx, x, x_skip);
        }

        // Self-attention part (attn1)
        auto h_norm1 = norm1->forward(ctx, x); // This is 'n' in Python code

        if (ref_attn_mode == REF_ATTN_WRITE && n_to_bank != nullptr) {
            // Caller (UNetModel) wants this `h_norm1` to be banked.
            // We pass it out. Caller is responsible for `ggml_dup_tensor` into a persistent bank context.
            *n_to_bank = h_norm1; // Assign pointer, caller must DUP.
            // attn1 will proceed normally with h_norm1 for its computation in WRITE mode.
        }

        ggml_tensor* self_attn_context_k = h_norm1;
        ggml_tensor* self_attn_context_v = h_norm1;

        if (ref_attn_mode == REF_ATTN_READ && ref_opts != nullptr && banked_n_for_read_mode != nullptr) {
            LOG_DEBUG("RefAttn READ in BasicTransformerBlock: Processing with banked features.");
            // Python:
            // real_bank = [...] copy of banked features, potentially on device
            // effective_strength = ... (can be a tensor mask or float)
            // real_bank[idx] = real_bank[idx] * effective_strength + context_attn1 * (1-effective_strength)
            // k_context_augmented = torch.cat([context_attn1] + processed_real_bank, dim=1)
            // v_context_augmented = torch.cat([value_attn1] + processed_real_bank, dim=1)

            // Here, h_norm1 is context_attn1 / value_attn1.
            // banked_n_for_read_mode is one tensor from the bank (real_bank[idx]).
            // We need to apply strength and then concatenate for K and V.
            // This assumes single item in bank for simplicity now.

            ggml_tensor* processed_bank_item = ggml_dup_tensor(ctx, banked_n_for_read_mode); // Work on a copy

            // Apply strength: processed_bank_item = banked_n * strength + h_norm1 * (1 - strength)
            // This is a simplified strength application (float). Python can have tensor masks.
            ggml_tensor* term1 = ggml_scale(ctx, processed_bank_item, ref_opts->attn_strength);
            ggml_tensor* term2 = ggml_scale(ctx, h_norm1, 1.0f - ref_opts->attn_strength);
            processed_bank_item = ggml_add(ctx, term1, term2);

            // Concatenate for K and V contexts. Assuming dim 1 is sequence length.
            // Shapes: h_norm1 [N, seq_len, D], processed_bank_item [N, bank_seq_len, D]
            // Assuming bank_seq_len == seq_len for simplicity here.
            self_attn_context_k = ggml_concat(ctx, h_norm1, processed_bank_item, 1);
            self_attn_context_v = ggml_concat(ctx, h_norm1, processed_bank_item, 1);
            // Note: `ggml_concat` creates a new tensor. K and V projections in CrossAttention
            // will operate on this extended sequence length.
        }

        // Pass potentially modified K,V context to attn1
        auto h_attn1 = attn1->forward(ctx, h_norm1, self_attn_context_k, self_attn_context_v, ref_attn_mode, ref_opts);
                                     // ^ query       ^ key context       ^ value context     ^ Pass relevant ref_attn_mode & opts

        // Style Fidelity application (if in READ mode)
        // Python: n_uc = self.attn1.to_out(...) with original K,V
        //         n_c = n_uc.clone(); if uncond: n_c[uncond_idx] = self.attn1.to_out(...) with ref K,V
        //         n = style_fidelity * n_c + (1-style_fidelity) * n_uc
        // This logic is now simplified: if style_fidelity < 1.0, we re-run attn1 with original K,V.
        if (ref_attn_mode == REF_ATTN_READ && ref_opts != nullptr && banked_n_for_read_mode != nullptr) {
            if (std::abs(ref_opts->attn_style_fidelity - 1.0f) > 1e-5) { // If not full fidelity
                // Get the "unconditioned" self-attention output (n_uc) by running attn1 with original h_norm1 as K,V context.
                ggml_tensor* h_attn1_unconditioned = attn1->forward(ctx, h_norm1, h_norm1, h_norm1, REF_ATTN_NORMAL, nullptr); // Use REF_ATTN_NORMAL for unconditioned pass

                ggml_tensor* term_cond   = ggml_scale(ctx, h_attn1, ref_opts->attn_style_fidelity);
                ggml_tensor* term_uncond = ggml_scale(ctx, h_attn1_unconditioned, 1.0f - ref_opts->attn_style_fidelity);
                h_attn1 = ggml_add(ctx, term_cond, term_uncond);
            }
        }


        auto h_after_attn1 = ggml_add(ctx, x, h_attn1); // Add back to original x (skip connection)

        // Cross-attention part (attn2)
        auto h_norm2 = norm2->forward(ctx, h_after_attn1);
        // Cross-attention always uses the external `context` for K and V, and REF_ATTN_NORMAL
        auto h_attn2 = attn2->forward(ctx, h_norm2, context, context, REF_ATTN_NORMAL, nullptr);
        auto h_after_attn2 = ggml_add(ctx, h_after_attn1, h_attn2);

        // Feed-forward part
        auto h_norm3 = norm3->forward(ctx, h_after_attn2);
        auto h_ff = ff->forward(ctx, h_norm3);
        auto h_final = ggml_add(ctx, h_after_attn2, h_ff);

        return h_final;
    }
};


class SpatialTransformer : public GGMLBlock {
protected:
    int64_t in_channels;  // mult * model_channels
    int64_t n_head;
    int64_t d_head;
    int64_t depth       = 1;    // 1
    int64_t context_dim = 768;  // hidden_size, 1024 for VERSION_SD2

public:
    SpatialTransformer(int64_t in_channels,
                       int64_t n_head,
                       int64_t d_head,
                       int64_t depth,
                       int64_t context_dim,
                       bool flash_attn = false)
        : in_channels(in_channels),
          n_head(n_head),
          d_head(d_head),
          depth(depth),
          context_dim(context_dim) {
        int64_t inner_dim = n_head * d_head;  // in_channels
        blocks["norm"]    = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["proj_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, inner_dim, {1, 1}));

        for (int i = 0; i < depth; i++) {
            std::string name = "transformer_blocks." + std::to_string(i);
            // Pass flash_attn to BasicTransformerBlock constructor
            blocks[name]     = std::shared_ptr<GGMLBlock>(new BasicTransformerBlock(inner_dim, n_head, d_head, context_dim, false, flash_attn));
        }

        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(inner_dim, in_channels, {1, 1}));
    }

    // The forward method now needs to handle RefAttnMode and pass it to BasicTransformerBlocks
    virtual struct ggml_tensor* forward(struct ggml_context* ctx,
                                        struct ggml_tensor* x,
                                        struct ggml_tensor* context,
                                        RefAttnMode ref_attn_mode = REF_ATTN_NORMAL,
                                        const ReferenceOptions_ggml* ref_opts = nullptr,
                                        // Output parameter for banking 'n' from each BasicTransformerBlock
                                        // This would be a vector if depth > 1
                                        std::vector<ggml_tensor*>* ns_to_bank = nullptr
                                        ) {
        auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto proj_in  = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);

        auto x_in         = x;
        int64_t n_batch   = x->ne[3]; // N
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n_batch);      // [N, h * w, inner_dim]

        if (ns_to_bank) ns_to_bank->clear(); // Clear if we are collecting n's for banking

        for (int i = 0; i < depth; i++) {
            std::string name       = "transformer_blocks." + std::to_string(i);
            auto transformer_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);

            ggml_tensor* current_n_to_bank = nullptr;
            if (ref_attn_mode == REF_ATTN_WRITE && ns_to_bank) {
                x = transformer_block->forward(ctx, x, context, ref_attn_mode, ref_opts, &current_n_to_bank);
                if (current_n_to_bank) {
                    ns_to_bank->push_back(current_n_to_bank); // Store pointer, caller dups
                }
            } else {
                 // For READ or NORMAL mode, or if not banking this SpatialTransformer's outputs
                x = transformer_block->forward(ctx, x, context, ref_attn_mode, ref_opts, nullptr);
            }
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n_batch);       // [N, inner_dim, h, w]

        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]
        x = ggml_add(ctx, x, x_in);
        return x;
    }

    // Helper to clear banks in all contained BasicTransformerBlocks
    void clear_attention_banks_in_children() {
        for (int i = 0; i < depth; i++) {
            std::string name = "transformer_blocks." + std::to_string(i);
            auto transformer_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);
            if (transformer_block) {
                // BasicTransformerBlock would need its own clear_bank method or way to reset banked_n_for_read_mode
                 transformer_block->clear_banked_n_after_read_pass(); // Example method
            }
        }
    }
     // Helper to set banked_n in all contained BasicTransformerBlocks for a READ pass
    void set_banked_n_for_read_pass_in_children(const std::vector<ggml_tensor*>& banked_n_list) {
        if (banked_n_list.size() != depth) {
            LOG_WARN("Mismatched number of banked tensors for SpatialTransformer depth.");
            return;
        }
        for (int i = 0; i < depth; ++i) {
            std::string name = "transformer_blocks." + std::to_string(i);
            auto transformer_block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);
            if (transformer_block) {
                transformer_block->set_banked_n_for_read_pass(banked_n_list[i]);
            }
        }
    }
};

class AlphaBlender : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F32;
        params["mix_factor"] = ggml_new_tensor_1d(ctx, wtype, 1);
    }

    float get_alpha() {
        float alpha = ggml_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }

public:
    AlphaBlender() {}

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x_spatial,
                                struct ggml_tensor* x_temporal) {
        float alpha = get_alpha();
        auto x      = ggml_add(ctx,
                               ggml_scale(ctx, x_spatial, alpha),
                               ggml_scale(ctx, x_temporal, 1.0f - alpha));
        return x;
    }
};

class VideoResBlock : public ResBlock {
public:
    VideoResBlock(int channels,
                  int emb_channels,
                  int out_channels,
                  std::pair<int, int> kernel_size = {3, 3},
                  int64_t video_kernel_size       = 3,
                  int dims                        = 2)
        : ResBlock(channels, emb_channels, out_channels, kernel_size, dims) {
        blocks["time_stack"] = std::shared_ptr<GGMLBlock>(new ResBlock(out_channels, emb_channels, out_channels, kernel_size, 3, true));
        blocks["time_mixer"] = std::shared_ptr<GGMLBlock>(new AlphaBlender());
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* emb,
                                int num_video_frames) { // This parameter is now specific to VideoResBlock
        auto time_stack = std::dynamic_pointer_cast<ResBlock>(blocks["time_stack"]);
        auto time_mixer = std::dynamic_pointer_cast<AlphaBlender>(blocks["time_mixer"]);

        x = ResBlock::forward(ctx, x, emb); // Call base class forward

        int64_t T = num_video_frames;
        int64_t B = x->ne[3] / T;
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];

        x          = ggml_reshape_4d(ctx, x, W * H, C, T, B);
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        auto x_mix = x;

        emb = ggml_reshape_4d(ctx, emb, emb->ne[0], T, B, emb->ne[3]);

        x = time_stack->forward(ctx, x, emb);

        x = time_mixer->forward(ctx, x_mix, x);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx, x, W, H, C, T * B);

        return x;
    }
};

#endif  // __COMMON_HPP__