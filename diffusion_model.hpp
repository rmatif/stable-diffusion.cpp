#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "unet.hpp" // Includes common.hpp for RefAttnMode and ReferenceOptions_ggml

struct DiffusionModel {
    virtual void compute(int n_threads,
                         struct ggml_tensor* x,
                         struct ggml_tensor* timesteps,
                         struct ggml_tensor* context,
                         struct ggml_tensor* c_concat,
                         struct ggml_tensor* y,
                         struct ggml_tensor* guidance, // For Flux
                         // Reference Attention parameters
                         RefAttnMode ref_attn_mode,
                         const ReferenceOptions_ggml* ref_opts, // Add this
                         // End Reference Attention parameters
                         int num_video_frames                      = -1,
                         std::vector<struct ggml_tensor*> controls = {},
                         float control_strength                    = 0.f,
                         struct ggml_tensor** output               = NULL,
                         struct ggml_context* output_ctx           = NULL,
                         std::vector<int> skip_layers              = std::vector<int>()
                         )             = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
    // New pure virtual method for clearing attention banks
    virtual void clear_attention_banks()                                                = 0;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              std::map<std::string, enum ggml_type>& tensor_types,
              SDVersion version = VERSION_SD1,
              bool flash_attn   = false)
        : unet(backend, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    void alloc_params_buffer() {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() {
        unet.free_params_buffer();
    }

    void free_compute_buffer() {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return unet.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return unet.unet.adm_in_channels;
    }

    void clear_attention_banks() override {
        unet.clear_unet_attention_banks();
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance, // Unused by UNetModel, but part of base signature
                         // Reference Attention parameters
                         RefAttnMode ref_attn_mode,
                         const ReferenceOptions_ggml* ref_opts,
                         // End Reference Attention parameters
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()
                 ) override {
        (void)skip_layers;  // SLG doesn't work with UNet models
        (void)guidance;     // UNetModel doesn't use Flux-style guidance tensor directly
        unet.compute(n_threads, x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength, ref_attn_mode, ref_opts, output, output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               std::map<std::string, enum ggml_type>& tensor_types)
        : mmdit(backend, tensor_types, "model.diffusion_model") {
    }

    void alloc_params_buffer() {
        mmdit.alloc_params_buffer();
    }
    void free_params_buffer() {
        mmdit.free_params_buffer();
    }
    void free_compute_buffer() {
        mmdit.free_compute_buffer();
    }
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }
    size_t get_params_buffer_size() {
        return mmdit.get_params_buffer_size();
    }
    int64_t get_adm_in_channels() {
        return 768 + 1280; // Example for SD3
    }
    void clear_attention_banks() override {
        // MMDiT might have a different internal structure for attention.
        // For now, assume it might need a similar mechanism if it were to support ref_attn.
        // Or this could be a no-op if MMDiT ref_attn is handled differently or not supported.
        LOG_WARN("MMDiTModel::clear_attention_banks() not fully implemented for reference attention.");
        // mmdit.clear_mmdit_attention_banks(); // Placeholder if MMDiTRunner has such a method
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat, // MMDiT uses y for pooled prompt, context for sequence
                 struct ggml_tensor* y,        // y is pooled_prompt_embeds for MMDiT
                 struct ggml_tensor* guidance, // Unused
                         RefAttnMode ref_attn_mode,
                         const ReferenceOptions_ggml* ref_opts,
                 int num_video_frames                      = -1, // Unused
                 std::vector<struct ggml_tensor*> controls = {}, // Unused
                 float control_strength                    = 0.f, // Unused
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()
                 ) override {
        (void)c_concat; (void)num_video_frames; (void)controls; (void)control_strength; (void)guidance;
        // MMDiT does not use ref_attn_mode or ref_opts in its current C++ form.
        // If ref_attn were added to MMDiT, its forward would need to take these.
        if (ref_attn_mode != REF_ATTN_NORMAL) {
            LOG_WARN("Reference Attention not implemented for MMDiTModel in C++ yet. Running in normal mode.");
        }
        mmdit.compute(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              std::map<std::string, enum ggml_type>& tensor_types,
              SDVersion version = VERSION_FLUX,
              bool flash_attn   = false)
        : flux(backend, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    void alloc_params_buffer() {
        flux.alloc_params_buffer();
    }
    void free_params_buffer() {
        flux.free_params_buffer();
    }
    void free_compute_buffer() {
        flux.free_compute_buffer();
    }
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }
    size_t get_params_buffer_size() {
        return flux.get_params_buffer_size();
    }
    int64_t get_adm_in_channels() {
        return 768; // Placeholder, Flux uses different conditioning
    }
     void clear_attention_banks() override {
        LOG_WARN("FluxModel::clear_attention_banks() not implemented for reference attention.");
        // flux.clear_flux_attention_banks(); // Placeholder
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,  // text_context_pooled_id for Flux
                 struct ggml_tensor* c_concat, // img_context_pooled_id for Flux (if img2img/inpaint)
                 struct ggml_tensor* y,        // text_sequence for Flux
                 struct ggml_tensor* guidance, // guidance_scale_embedding for Flux
                         RefAttnMode ref_attn_mode,
                         const ReferenceOptions_ggml* ref_opts,
                 int num_video_frames                      = -1, // Unused
                 std::vector<struct ggml_tensor*> controls = {}, // Unused
                 float control_strength                    = 0.f, // Unused
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()
                 ) override {
        (void)num_video_frames; (void)controls; (void)control_strength;
         if (ref_attn_mode != REF_ATTN_NORMAL) {
            LOG_WARN("Reference Attention not implemented for FluxModel in C++ yet. Running in normal mode.");
        }
        flux.compute(n_threads, x, timesteps, context, c_concat, y, guidance, output, output_ctx, skip_layers);
    }
};

#endif // __DIFFUSION_MODEL_H__