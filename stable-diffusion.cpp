#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"
#include "stb_image_resize.h" 
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SDXL",
    "SDXL Inpaint",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill"};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD"
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

// CONCEPTUAL CHANGE IN OTHER FILE: This enum would typically be in a shared header like unet.hpp or common.hpp

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = NULL;  // general backend
    ggml_backend_t clip_backend        = NULL;
    ggml_backend_t control_net_backend = NULL;
    ggml_backend_t vae_backend         = NULL;
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type conditioner_wtype        = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;
    ggml_type vae_wtype                = GGML_TYPE_COUNT;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEmbed> pmid_id_embeds;

    std::string taesd_path;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;
    bool stacked_id           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    // Reference Attention members
    std::string reference_attn_image_path;
    ggml_tensor* reference_latent_original = NULL; // Stores the VAE encoded original reference image
    ggml_context* reference_latent_ctx = NULL;   // Context for reference_latent_original
    ReferenceOptions_ggml reference_options;
    bool reference_attn_enabled = false;


    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        rng_type_t rng_type,
                        // Reference Attention params from main
                        const std::string& ref_attn_image_p,
                        const ReferenceOptions_ggml& ref_opts)
        : n_threads(n_threads),
          vae_decode_only(vae_decode_only),
          free_params_immediately(free_params_immediately),
          lora_model_dir(lora_model_dir),
          reference_attn_image_path(ref_attn_image_p),
          reference_options(ref_opts) {
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
        if (!reference_attn_image_path.empty()) {
            reference_attn_enabled = true;
        }
    }

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);

        if (reference_latent_ctx != NULL) {
            ggml_free(reference_latent_ctx);
            reference_latent_ctx = NULL;
            reference_latent_original = NULL; // Tensor was in this context
        }
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& clip_l_path,
                        const std::string& clip_g_path,
                        const std::string& t5xxl_path,
                        const std::string& diffusion_model_path,
                        const std::string& vae_path,
                        const std::string control_net_path,
                        const std::string embeddings_path,
                        const std::string id_embeddings_path,
                        const std::string& taesd_path,
                        bool vae_tiling_,
                        ggml_type wtype,
                        schedule_t schedule,
                        bool clip_on_cpu,
                        bool control_net_cpu,
                        bool vae_on_cpu,
                        bool diffusion_flash_attn) {
        use_tiny_autoencoder = taesd_path.size() > 0;
#ifdef SD_USE_CUDA
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        ggml_log_set(ggml_log_callback_default, nullptr);
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }

        ModelLoader model_loader;

        vae_tiling = vae_tiling_;

        if (model_path.size() > 0) {
            LOG_INFO("loading model from '%s'", model_path.c_str());
            if (!model_loader.init_from_file(model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
            }
        }

        if (clip_l_path.size() > 0) {
            LOG_INFO("loading clip_l from '%s'", clip_l_path.c_str());
            if (!model_loader.init_from_file(clip_l_path, "text_encoders.clip_l.transformer.")) {
                LOG_WARN("loading clip_l from '%s' failed", clip_l_path.c_str());
            }
        }

        if (clip_g_path.size() > 0) {
            LOG_INFO("loading clip_g from '%s'", clip_g_path.c_str());
            if (!model_loader.init_from_file(clip_g_path, "text_encoders.clip_g.transformer.")) {
                LOG_WARN("loading clip_g from '%s' failed", clip_g_path.c_str());
            }
        }

        if (t5xxl_path.size() > 0) {
            LOG_INFO("loading t5xxl from '%s'", t5xxl_path.c_str());
            if (!model_loader.init_from_file(t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", t5xxl_path.c_str());
            }
        }

        if (diffusion_model_path.size() > 0) {
            LOG_INFO("loading diffusion model from '%s'", diffusion_model_path.c_str());
            if (!model_loader.init_from_file(diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", diffusion_model_path.c_str());
            }
        }

        if (vae_path.size() > 0) {
            LOG_INFO("loading vae from '%s'", vae_path.c_str());
            if (!model_loader.init_from_file(vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", vae_path.c_str());
            }
        }

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", model_path.c_str());
            return false;
        }

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        if (wtype == GGML_TYPE_COUNT) {
            model_wtype = model_loader.get_sd_wtype();
            if (model_wtype == GGML_TYPE_COUNT) {
                model_wtype = GGML_TYPE_F32;
                LOG_WARN("can not get mode wtype frome weight, use f32");
            }
            conditioner_wtype = model_loader.get_conditioner_wtype();
            if (conditioner_wtype == GGML_TYPE_COUNT) {
                conditioner_wtype = wtype;
            }
            diffusion_model_wtype = model_loader.get_diffusion_model_wtype();
            if (diffusion_model_wtype == GGML_TYPE_COUNT) {
                diffusion_model_wtype = wtype;
            }
            vae_wtype = model_loader.get_vae_wtype();

            if (vae_wtype == GGML_TYPE_COUNT) {
                vae_wtype = wtype;
            }
        } else {
            model_wtype           = wtype;
            conditioner_wtype     = wtype;
            diffusion_model_wtype = wtype;
            vae_wtype             = wtype;
            model_loader.set_wtype_override(wtype);
        }

        if (sd_version_is_sdxl(version)) {
            vae_wtype = GGML_TYPE_F32;
            model_loader.set_wtype_override(GGML_TYPE_F32, "vae.");
        }

        LOG_INFO("Weight type:                 %s", model_wtype != SD_TYPE_COUNT ? ggml_type_name(model_wtype) : "??");
        LOG_INFO("Conditioner weight type:     %s", conditioner_wtype != SD_TYPE_COUNT ? ggml_type_name(conditioner_wtype) : "??");
        LOG_INFO("Diffusion model weight type: %s", diffusion_model_wtype != SD_TYPE_COUNT ? ggml_type_name(diffusion_model_wtype) : "??");
        LOG_INFO("VAE weight type:             %s", vae_wtype != SD_TYPE_COUNT ? ggml_type_name(vae_wtype) : "??");

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_version_is_sdxl(version)) {
            scale_factor = 0.13025f;
            if (vae_path.size() == 0 && taesd_path.size() == 0) {
                LOG_WARN(
                    "!!!It looks like you are using SDXL model. "
                    "If you find that the generated images are completely black, "
                    "try specifying SDXL VAE FP16 Fix with the --vae parameter. "
                    "You can find it here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors");
            }
        } else if (sd_version_is_sd3(version)) {
            scale_factor = 1.5305f;
        } else if (sd_version_is_flux(version)) {
            scale_factor = 0.3611;
            // TODO: shift_factor
        }

        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, model_loader.tensor_storages_types);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors);

            diffusion_model = std::make_shared<UNetModel>(backend, model_loader.tensor_storages_types, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, model_loader.tensor_storages_types, "first_stage_model", vae_decode_only, true, version);
            LOG_DEBUG("vae_decode_only %d", vae_decode_only);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
            clip_backend   = backend;
            bool use_t5xxl = false;
            if (sd_version_is_dit(version)) {
                use_t5xxl = true;
            }
            if (!ggml_backend_is_cpu(backend) && use_t5xxl && conditioner_wtype != GGML_TYPE_F32) {
                clip_on_cpu = true;
                LOG_INFO("set clip_on_cpu to true");
            }
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (diffusion_flash_attn) {
                LOG_INFO("Using flash attention in the diffusion model");
            }
            if (sd_version_is_sd3(version)) {
                if (diffusion_flash_attn) {
                    LOG_WARN("flash attention in this diffusion model is currently unsupported!");
                }
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend, model_loader.tensor_storages_types);
                diffusion_model  = std::make_shared<MMDiTModel>(backend, model_loader.tensor_storages_types);
            } else if (sd_version_is_flux(version)) {
                cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend, model_loader.tensor_storages_types);
                diffusion_model  = std::make_shared<FluxModel>(backend, model_loader.tensor_storages_types, version, diffusion_flash_attn);
            } else {
                if (id_embeddings_path.find("v2") != std::string::npos) {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, model_loader.tensor_storages_types, embeddings_path, version, PM_VERSION_2);
                } else {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, model_loader.tensor_storages_types, embeddings_path, version);
                }
                diffusion_model = std::make_shared<UNetModel>(backend, model_loader.tensor_storages_types, version, diffusion_flash_attn);
            }

            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            if (!use_tiny_autoencoder) {
                if (vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_INFO("VAE Autoencoder: Using CPU backend");
                    vae_backend = ggml_backend_cpu_init();
                } else {
                    vae_backend = backend;
                }
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, model_loader.tensor_storages_types, "first_stage_model", vae_decode_only, false, version);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            } else {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend, model_loader.tensor_storages_types, "decoder.layers", vae_decode_only, version);
            }
            // first_stage_model->get_param_tensors(tensors, "first_stage_model.");

            if (control_net_path.size() > 0) {
                ggml_backend_t controlnet_backend = NULL;
                if (control_net_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend, model_loader.tensor_storages_types, version);
            }

            if (id_embeddings_path.find("v2") != std::string::npos) {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend, model_loader.tensor_storages_types, "pmid", version, PM_VERSION_2);
                LOG_INFO("using PhotoMaker Version 2");
            } else {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend, model_loader.tensor_storages_types, "pmid", version);
            }
            if (id_embeddings_path.size() > 0) {
                pmid_lora = std::make_shared<LoraModel>(backend, id_embeddings_path, "");
                if (!pmid_lora->load_from_file(true)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", id_embeddings_path.c_str());
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", id_embeddings_path.c_str());
                if (!model_loader.init_from_file(id_embeddings_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", id_embeddings_path.c_str());
                } else {
                    stacked_id = true;
                }
            }
            if (stacked_id) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                pmid_model->get_param_tensors(tensors, "pmid");
            }
        }

        struct ggml_init_params params_ctx_init;
        params_ctx_init.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params_ctx_init.mem_buffer = NULL;
        params_ctx_init.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params_ctx_init.mem_size);
        struct ggml_context* ctx_alphas = ggml_init(params_ctx_init);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx_alphas != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx_alphas, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        int64_t t0_load_weights = ggml_time_ms();

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success_load_tensors = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success_load_tensors) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx_alphas);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        if (version == VERSION_SVD) {
            // diffusion_model->test();
            // first_stage_model->test();
            // return false;
        } else {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = 0;
            if (!use_tiny_autoencoder) {
                vae_params_mem_size = first_stage_model->get_params_buffer_size();
            } else {
                if (!tae_first_stage->load_from_file(taesd_path)) {
                    return false;
                }
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(control_net_path)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (stacked_id) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (control_net_backend != NULL && ggml_backend_is_cpu(control_net_backend)) { // Ensure control_net_backend is initialized
                total_params_ram_size += control_net_params_mem_size;
            } else if (control_net_backend != NULL) {
                total_params_vram_size += control_net_params_mem_size;
            }


            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                (control_net_backend != NULL && ggml_backend_is_cpu(control_net_backend)) ? "RAM" : (control_net_backend != NULL ? "VRAM" : "N/A"),
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        int64_t t1_load_weights = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1_load_weights - t0_load_weights) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (sd_version_is_sd2(version)) {
            if (is_using_v_parameterization_for_sd2(ctx_alphas, sd_version_is_inpaint(version))) {
                is_using_v_parameterization = true;
            }
        } else if (sd_version_is_sdxl(version)) {
            if (model_loader.tensor_storages_types.find("v_pred") != model_loader.tensor_storages_types.end()) {
                is_using_v_parameterization = true;
            }
        } else if (version == VERSION_SVD) {
            // TODO: V_PREDICTION_EDM
            is_using_v_parameterization = true;
        }

        if (sd_version_is_sd3(version)) {
            LOG_INFO("running in FLOW mode");
            denoiser = std::make_shared<DiscreteFlowDenoiser>();
        } else if (sd_version_is_flux(version)) {
            LOG_INFO("running in Flux FLOW mode");
            float shift = 1.0f;  // TODO: validate
            for (auto pair : model_loader.tensor_storages_types) {
                if (pair.first.find("model.diffusion_model.guidance_in.in_layer.weight") != std::string::npos) {
                    shift = 1.15f;
                    break;
                }
            }
            denoiser = std::make_shared<FluxFlowDenoiser>(shift);
        } else if (is_using_v_parameterization) {
            LOG_INFO("running in v-prediction mode");
            denoiser = std::make_shared<CompVisVDenoiser>();
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (schedule != DEFAULT) {
            switch (schedule) {
                case DISCRETE:
                    LOG_INFO("running with discrete schedule");
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    LOG_INFO("running with Karras schedule");
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case EXPONENTIAL:
                    LOG_INFO("running exponential schedule");
                    denoiser->schedule = std::make_shared<ExponentialSchedule>();
                    break;
                case AYS:
                    LOG_INFO("Running with Align-Your-Steps schedule");
                    denoiser->schedule          = std::make_shared<AYSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case GITS:
                    LOG_INFO("Running with GITS schedule");
                    denoiser->schedule          = std::make_shared<GITSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", schedule);
                    abort();
            }
        }

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            for (int i = 0; i < TIMESTEPS; i++) {
                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }

        // Reference Attention: Load and VAE Encode reference image if provided
        if (reference_attn_enabled) {
            // For SD 1.5, reference_attn is applicable. For others, it might be disabled or behave differently.
            // The current request is for SD 1.5, so this is fine.
            if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) { // Enable for SD1/SD2
                LOG_INFO("Reference Attention enabled with image: %s", reference_attn_image_path.c_str());
                LOG_INFO("Reference Attn Options: Fidelity=%.2f, Strength=%.2f",
                         reference_options.attn_style_fidelity, reference_options.attn_strength);

                int ref_img_w = 0, ref_img_h = 0, ref_img_c = 0;
                uint8_t* ref_img_data = stbi_load(reference_attn_image_path.c_str(), &ref_img_w, &ref_img_h, &ref_img_c, 3);
                if (ref_img_data == NULL) {
                    LOG_ERROR("Failed to load reference attention image from: %s", reference_attn_image_path.c_str());
                    reference_attn_enabled = false; // Disable if image load fails
                } else if (ref_img_c < 3) {
                     LOG_ERROR("Reference attention image must have at least 3 channels, got %d", ref_img_c);
                    stbi_image_free(ref_img_data);
                    reference_attn_enabled = false;
                } else {
                    // Reference image needs to be processed to match latent dimensions of the generation.
                    // Let's assume generation width/height are available (e.g. from CLI params)
                    // For now, we'll use the diffusion_model's default input C, H, W and scale the ref image to that.
                    // The target latent size is (width/8, height/8). Image size is (width, height).
                    // This part is tricky as the generation width/height are not directly available here.
                    // Let's assume it's processed to a fixed size for now or passed in.
                    // The python code uses `latent_size["samples"].shape` to get target latent H/W.
                    // For now, we just load it. VAE encoding will happen in `sample` or a setup phase.
                    // We need a persistent context for `reference_latent_original`.
                    struct ggml_init_params ref_latent_ctx_params;
                    // Estimate size: 4 channels * (max_width/8) * (max_height/8) * sizeof(float) + overhead
                    // Example: 4 * 128 * 128 * 4 = 256KB. Add some buffer.
                    ref_latent_ctx_params.mem_size = 1 * 1024 * 1024; // 1MB should be enough for one latent
                    ref_latent_ctx_params.mem_buffer = NULL;
                    ref_latent_ctx_params.no_alloc = false;
                    reference_latent_ctx = ggml_init(ref_latent_ctx_params);
                    if (reference_latent_ctx == NULL) {
                        LOG_ERROR("Failed to create context for reference latent.");
                        stbi_image_free(ref_img_data);
                        reference_attn_enabled = false;
                    } else {
                        // Create a temporary work_ctx for VAE encoding the reference image
                        struct ggml_init_params temp_work_ctx_params;
                        temp_work_ctx_params.mem_size = 256LL * 1024 * 1024; // Temp ctx for VAE encoding
                        temp_work_ctx_params.mem_buffer = NULL;
                        temp_work_ctx_params.no_alloc = false;
                        ggml_context* temp_work_ctx = ggml_init(temp_work_ctx_params);

                        if (temp_work_ctx) {
                            // We need target width/height for generation to resize ref image correctly.
                            // This should ideally come from SDParams. For now, let's assume 512x512.
                            // The actual latent size is determined by the output image size.
                            // This VAE encoding should happen *after* we know the target generation H/W.
                            // So, we'll store the path and do VAE encoding later or pass a placeholder.
                            // For now, let's defer full VAE encoding of ref image to the `sample` or `generate_image` call
                            // where output W/H are known.
                            // For now, just free data, path is stored.
                            LOG_DEBUG("Reference image loaded, VAE encoding will occur before sampling.");
                            stbi_image_free(ref_img_data); // Path stored, actual processing later
                            // Placeholder: reference_latent_original will be set up in `sample` or `generate_image`.
                        } else {
                            LOG_ERROR("Failed to create temp work_ctx for reference VAE encoding.");
                            stbi_image_free(ref_img_data);
                            reference_attn_enabled = false;
                            ggml_free(reference_latent_ctx);
                            reference_latent_ctx = NULL;
                        }
                         if (temp_work_ctx) ggml_free(temp_work_ctx);
                    }
                }
            } else {
                 LOG_WARN("Reference Attention is currently only supported for SD 1.x/2.x models. Disabling.");
                 reference_attn_enabled = false;
            }
        }


        LOG_DEBUG("finished loaded file");
        ggml_free(ctx_alphas); // Free the context used for alphas_cumprod
        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx, bool is_inpaint = false) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);

        struct ggml_tensor* concat = is_inpaint ? ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 5, 1) : NULL;
        if (concat != NULL) {
            ggml_set_f32(concat, 0);
        }

        int64_t t0              = ggml_time_ms();
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model->compute(n_threads, x_t, timesteps, c, concat, NULL, NULL, 
                                 REF_ATTN_NORMAL, nullptr, // ref_attn_mode, ref_opts
                                 -1, {}, 0.f, &out, work_ctx); // Pass work_ctx for output_ctx if needed by compute
        diffusion_model->free_compute_buffer();

        double result = 0.f;
        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            LOG_WARN("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str());
            return;
        }
        LoraModel lora(backend, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        // TODO: send version?
        lora.apply(tensors, version, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        if (lora_state.size() > 0 && model_wtype != GGML_TYPE_F16 && model_wtype != GGML_TYPE_F32) {
            LOG_WARN("In quantized models when applying LoRA, the images have poor quality.");
        }
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }
        
        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("Attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("Attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    ggml_tensor* id_encoder(ggml_context* work_ctx,
                            ggml_tensor* init_img,
                            ggml_tensor* prompts_embeds,
                            ggml_tensor* id_embeds,
                            std::vector<bool>& class_tokens_mask) {
        ggml_tensor* res = NULL;
        pmid_model->compute(n_threads, init_img, prompts_embeds, id_embeds, class_tokens_mask, &res, work_ctx);
        return res;
    }

    SDCondition get_svd_condition(ggml_context* work_ctx,
                                  sd_image_t init_image,
                                  int width,
                                  int height,
                                  int fps                    = 6,
                                  int motion_bucket_id       = 127,
                                  float augmentation_level   = 0.f,
                                  bool force_zero_embeddings = false) {
        // c_crossattn
        int64_t t0                      = ggml_time_ms();
        struct ggml_tensor* c_crossattn = NULL;
        {
            if (force_zero_embeddings) {
                c_crossattn = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, clip_vision->vision_model.projection_dim);
                ggml_set_f32(c_crossattn, 0.f);
            } else {
                sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size);
                free(image.data);
                image.data = NULL;

                ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_tensor(resized_image.data, pixel_values, false);
                free(resized_image.data);
                resized_image.data = NULL;

                // print_ggml_tensor(pixel_values);
                clip_vision->compute(n_threads, pixel_values, &c_crossattn, work_ctx);
                // print_ggml_tensor(c_crossattn);
            }
        }

        // c_concat
        struct ggml_tensor* c_concat = NULL;
        {
            if (force_zero_embeddings) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = NULL;
                    sd_image_f32_to_tensor(resized_image.data, init_img, false);
                    free(resized_image.data);
                    resized_image.data = NULL;
                } else {
                    sd_image_to_tensor(init_image.data, init_img);
                }
                if (augmentation_level > 0.f) {
                    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_tensor_set_f32_randn(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_tensor_scale(noise, augmentation_level);
                    ggml_tensor_add(init_img, noise);
                }
                ggml_tensor* moments = encode_first_stage(work_ctx, init_img);
                c_concat             = get_first_stage_encoding(work_ctx, moments);
            }
        }

        // y
        struct ggml_tensor* y = NULL;
        {
            y                            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->get_adm_in_channels());
            int out_dim                  = 256;
            int fps_id                   = fps - 1;
            std::vector<float> timesteps = {(float)fps_id, (float)motion_bucket_id, augmentation_level};
            set_timestep_embedding(timesteps, y, out_dim);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing svd condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {c_crossattn, y, c_concat};
    }

    // Helper to noise latents for reference attention WRITE pass, similar to Python's ref_noise_latents
    // This function assumes `original_latent` is in work_ctx and returns a new tensor in work_ctx.
    ggml_tensor* noise_reference_latent_for_step(ggml_context* work_ctx, ggml_tensor* original_latent, float sigma_val) {
        // sigma_val is the sigma for the current step
        // alpha_cumprod = 1 / (sigma^2 + 1)
        // sqrt_alpha_prod = sqrt(alpha_cumprod)
        // sqrt_one_minus_alpha_prod = sqrt(1 - alpha_cumprod)
        // return sqrt_alpha_prod * original_latent + sqrt_one_minus_alpha_prod * noise_sample;

        float alpha_cumprod = 1.0f / ((sigma_val * sigma_val) + 1.0f);
        float sqrt_alpha_prod = sqrtf(alpha_cumprod);
        float sqrt_one_minus_alpha_prod = sqrtf(1.0f - alpha_cumprod);

        ggml_tensor* noise_sample = ggml_dup_tensor(work_ctx, original_latent);
        ggml_tensor_set_f32_randn(noise_sample, rng); // Use the main RNG for this

        ggml_tensor* term1 = ggml_scale(work_ctx, original_latent, sqrt_alpha_prod);
        ggml_tensor* term2 = ggml_scale(work_ctx, noise_sample, sqrt_one_minus_alpha_prod);
        
        ggml_tensor* noised_latent = ggml_add(work_ctx, term1, term2);
        return noised_latent;
    }


    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent, // Initial latent (either from img2img or blank for txt2img)
                        ggml_tensor* noise,       // Initial noise for the sampling process
                        SDCondition cond,
                        SDCondition uncond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        float min_cfg,
                        float cfg_scale,
                        float guidance,
                        float eta,
                        sample_method_t method,
                        const std::vector<float>& sigmas, // Sigmas for each step
                        int start_merge_step,
                        SDCondition id_cond,
                        std::vector<int> skip_layers = {},
                        float slg_scale              = 0,
                        float skip_layer_start       = 0.01,
                        float skip_layer_end         = 0.2,
                        ggml_tensor* noise_mask      = nullptr) {
        LOG_DEBUG("Sample");
        struct ggml_init_params params_tmp_ctx; // Renamed to avoid conflict
        size_t data_size = ggml_row_size(init_latent->type, init_latent->ne[0]);
        for (int i = 1; i < 4; i++) {
            data_size *= init_latent->ne[i];
        }
        data_size += 1024; // General buffer
        // If reference attention is enabled, need more space for noised reference latent + its noise sample
        if (reference_attn_enabled && reference_latent_original) {
            data_size += ggml_nbytes(reference_latent_original) * 2;
        }
        params_tmp_ctx.mem_size       = data_size * 3; // Increased buffer
        params_tmp_ctx.mem_buffer     = NULL;
        params_tmp_ctx.no_alloc       = false;
        ggml_context* tmp_ctx = ggml_init(params_tmp_ctx); // Context for per-step operations

        size_t steps = sigmas.size() - 1;
        // noise = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(noise);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent); // Current latent, starts with init_latent
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x); // x is now the fully noised latent x_T

        struct ggml_tensor* noised_input_for_unet = ggml_dup_tensor(work_ctx, noise); // Buffer for U-Net input (scaled x)

        bool has_unconditioned = cfg_scale != 1.0 && uncond.c_crossattn != NULL;
        bool has_skiplayer     = slg_scale != 0.0 && skip_layers.size() > 0;

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x); // Buffer for conditional U-Net output
        struct ggml_tensor* out_uncond = NULL;                         // Buffer for unconditional U-Net output
        struct ggml_tensor* out_skip   = NULL;                         // Buffer for skip-layer U-Net output

        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        if (has_skiplayer) {
            if (sd_version_is_dit(version)) {
                out_skip = ggml_dup_tensor(work_ctx, x);
            } else {
                has_skiplayer = false;
                LOG_WARN("SLG is incompatible with %s models", model_version_to_str[version]);
            }
        }
        struct ggml_tensor* denoised_pred = ggml_dup_tensor(work_ctx, x); // Buffer for the model's prediction (eps or v)

        auto denoise_step_fn = [&](ggml_tensor* current_x_t, float sigma_current, int step_idx) -> ggml_tensor* {
            if (step_idx == 0) { // Python step is 1-based, C++ is 0-based for loops
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0_step = ggml_time_us();

            std::vector<float> scaling = denoiser->get_scalings(sigma_current);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t_value_for_unet = denoiser->sigma_to_t(sigma_current);
            std::vector<float> timesteps_vec(current_x_t->ne[3], t_value_for_unet);
            auto timesteps_for_unet = vector_to_ggml_tensor(tmp_ctx, timesteps_vec); // Use tmp_ctx for step-local tensors
            
            std::vector<float> guidance_vec(current_x_t->ne[3], guidance);
            auto guidance_tensor = vector_to_ggml_tensor(tmp_ctx, guidance_vec);


            copy_ggml_tensor(noised_input_for_unet, current_x_t);
            ggml_tensor_scale(noised_input_for_unet, c_in); // Scale input for U-Net

            std::vector<struct ggml_tensor*> current_controls; // ControlNet controls for this step

            // Reference Attention: WRITE pass
            if (reference_attn_enabled && reference_latent_original != NULL) {
                // Noise the original reference latent based on current sigma
                // The Python version `control.cond_hint` is already noised.
                // `ref_noise_latents(self.cond_hint, sigma=t, noise=None)`
                // where `self.cond_hint` comes from `self.model_latent_format.process_in(self.cond_hint)`
                // and that `self.cond_hint` is potentially an upscaled `self.cond_hint_original`
                // The key is that `sigma=t` (current step sigma) is used.
                ggml_tensor* ref_latent_noised_for_step = noise_reference_latent_for_step(tmp_ctx, reference_latent_original, sigma_current);

                // Scale the noised reference latent for U-Net input, similar to current_x_t
                ggml_tensor* scaled_ref_latent_noised_for_step = ggml_scale(tmp_ctx, ref_latent_noised_for_step, c_in);

                // Use conditional context for the WRITE pass, as per Python logic using *args
                // CONCEPTUAL CHANGE IN OTHER FILE: DiffusionModel::compute needs RefAttnMode
                diffusion_model->compute(n_threads,
                                         scaled_ref_latent_noised_for_step,
                                         timesteps_for_unet, // Use same timesteps as main pass
                                         cond.c_crossattn,   // Use main conditional context
                                         cond.c_concat,
                                         cond.c_vector,
                                         guidance_tensor,    // guidance tensor (for Flux)
                                         REF_ATTN_WRITE,     // ref_attn_mode
                                         &reference_options, // ref_opts
                                         -1,                 // num_video_frames
                                         {},                 // controls
                                         0.f,                // control_strength
                                         nullptr,            // output tensor**
                                         nullptr             // output_ctx
                                         );     
            }


            if (control_hint != NULL) {
                control_net->compute(n_threads, noised_input_for_unet, control_hint, timesteps_for_unet, cond.c_crossattn, cond.c_vector);
                current_controls = control_net->controls;
            }

            // Conditional prediction
            SDCondition current_cond_context = (start_merge_step == -1 || step_idx <= start_merge_step) ? cond : id_cond;
             // CONCEPTUAL CHANGE IN OTHER FILE: DiffusionModel::compute needs RefAttnMode
            diffusion_model->compute(n_threads,
                                     noised_input_for_unet,
                                     timesteps_for_unet,
                                     current_cond_context.c_crossattn,
                                     current_cond_context.c_concat,
                                     current_cond_context.c_vector,
                                     guidance_tensor, // guidance tensor (for Flux)
                                     (reference_attn_enabled && reference_latent_original != NULL ? REF_ATTN_READ : REF_ATTN_NORMAL), // ref_attn_mode
                                     (reference_attn_enabled && reference_latent_original != NULL ? &reference_options : nullptr),    // ref_opts
                                     -1,                    // num_video_frames
                                     current_controls,      // controls
                                     control_strength,      // control_strength
                                     &out_cond,             // output tensor**
                                     work_ctx               // output_ctx (use work_ctx for step outputs)
                                     );

            float* negative_pred_data = NULL;
            if (has_unconditioned) {
                if (control_hint != NULL) { // Recompute ControlNet with uncond context if needed
                    control_net->compute(n_threads, noised_input_for_unet, control_hint, timesteps_for_unet, uncond.c_crossattn, uncond.c_vector);
                    current_controls = control_net->controls; // Update controls if they differ for uncond
                }
                // Unconditional prediction
                // CONCEPTUAL CHANGE IN OTHER FILE: DiffusionModel::compute needs RefAttnMode
                // For uncond pass with reference, style fidelity in BasicTransformerBlock will handle it.
                // It also needs REF_ATTN_READ mode if reference is active.
                diffusion_model->compute(n_threads,
                                         noised_input_for_unet,
                                         timesteps_for_unet,
                                         uncond.c_crossattn,
                                         uncond.c_concat,
                                         uncond.c_vector,
                                         guidance_tensor, // guidance tensor (for Flux)
                                         (reference_attn_enabled && reference_latent_original != NULL ? REF_ATTN_READ : REF_ATTN_NORMAL), // ref_attn_mode
                                         (reference_attn_enabled && reference_latent_original != NULL ? &reference_options : nullptr),    // ref_opts
                                         -1,                    // num_video_frames
                                         current_controls,      // controls
                                         control_strength,      // control_strength
                                         &out_uncond,           // output tensor**
                                         work_ctx               // output_ctx
                                         );
                negative_pred_data = (float*)out_uncond->data;
            }
            
            // Clear attention banks after COND and UNCOND (if any) read passes for the current step are done
            if (reference_attn_enabled) {
                // CONCEPTUAL CHANGE IN OTHER FILE: diffusion_model needs clear_attention_banks()
                diffusion_model->clear_attention_banks();
            }


            int step_count         = sigmas.size() -1; // total number of steps
            bool is_skiplayer_step = has_skiplayer && step_idx > (int)(skip_layer_start * step_count) && step_idx < (int)(skip_layer_end * step_count);
            float* skip_layer_pred_data = NULL;
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step_idx);
                 // CONCEPTUAL CHANGE IN OTHER FILE: DiffusionModel::compute needs skip_layers and RefAttnMode
                diffusion_model->compute(n_threads,
                                         noised_input_for_unet,
                                         timesteps_for_unet,
                                         current_cond_context.c_crossattn, // Use same cond as main conditional pass
                                         current_cond_context.c_concat,
                                         current_cond_context.c_vector,
                                         guidance_tensor, // guidance tensor (for Flux)
                                         (reference_attn_enabled && reference_latent_original != NULL ? REF_ATTN_READ : REF_ATTN_NORMAL), // ref_attn_mode
                                         (reference_attn_enabled && reference_latent_original != NULL ? &reference_options : nullptr),    // ref_opts
                                         -1,                    // num_video_frames
                                         current_controls,      // controls
                                         control_strength,      // control_strength
                                         &out_skip,             // output tensor**
                                         work_ctx,              // output_ctx
                                         skip_layers            // skip_layers
                                         );
                skip_layer_pred_data = (float*)out_skip->data;

                // Clear banks again if REF_ATTN_READ was used for skip-layer pass
                if (reference_attn_enabled) {
                    diffusion_model->clear_attention_banks();
                }
            }

            float* vec_denoised_pred  = (float*)denoised_pred->data;
            float* vec_current_x_t    = (float*)current_x_t->data;
            float* positive_pred_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised_pred);

            for (int i = 0; i < ne_elements; i++) {
                float final_pred = positive_pred_data[i];
                if (has_unconditioned) {
                    int64_t ne3 = out_cond->ne[3]; // Batch dimension
                    float current_cfg_scale = cfg_scale;
                    if (min_cfg != cfg_scale && ne3 > 1) { // Per-image CFG scaling if batch > 1
                        int64_t batch_idx  = i / (out_cond->ne[0] * out_cond->ne[1] * out_cond->ne[2]);
                        current_cfg_scale = min_cfg + (cfg_scale - min_cfg) * (batch_idx * 1.0f / (ne3 -1.f)); // Ensure float division
                    }
                    final_pred = negative_pred_data[i] + current_cfg_scale * (positive_pred_data[i] - negative_pred_data[i]);
                }
                if (is_skiplayer_step) {
                    final_pred = final_pred + (positive_pred_data[i] - skip_layer_pred_data[i]) * slg_scale;
                }
                // denoised_pred = (pred * c_out + current_x_t * c_skip)
                vec_denoised_pred[i] = final_pred * c_out + vec_current_x_t[i] * c_skip;
            }
            int64_t t1_step = ggml_time_us();
            if (step_idx >= 0) { // Python step is 1-based
                pretty_progress(step_idx + 1, (int)steps, (t1_step - t0_step) / 1000000.f);
            }
            if (noise_mask != nullptr) {
                // Apply inpainting mask: result = init_latent_noised_at_this_step * (1-mask) + denoised_pred * mask
                // This requires init_latent to be noised to current step sigma, then blended.
                // Simplified: result = init_latent * (1-mask) + denoised_pred * mask (like ComfyUI inpaint node)
                // More accurately, should be:
                // noise_for_masking = ggml_dup_tensor(tmp_ctx, init_latent);
                // ggml_tensor_set_f32_randn(noise_for_masking, rng); // use consistent noise if possible
                // init_latent_noised_to_sigma_current = denoiser->noise_scaling(sigma_current, noise_for_masking, init_latent);

                for (int64_t el_x = 0; el_x < denoised_pred->ne[0]; el_x++) { // W
                    for (int64_t el_y = 0; el_y < denoised_pred->ne[1]; el_y++) { // H
                        float mask_val = ggml_tensor_get_f32(noise_mask, el_x, el_y, 0, 0); // Assuming mask is [W/8, H/8, 1, 1]
                        for (int64_t el_k = 0; el_k < denoised_pred->ne[2]; el_k++) { // C
                            for(int64_t el_b = 0; el_b < denoised_pred->ne[3]; el_b++) { // Batch
                                float original_pixel_val = ggml_tensor_get_f32(init_latent, el_x, el_y, el_k, el_b); 
                                // Need to noise original_pixel_val to current sigma to match denoised_pred's state
                                // This is complex. Python does: x0 = x0_in*(1.0-mask_pt) + noised_sample_x0*mask_pt
                                // where x0_in is initial latent, noised_sample_x0 is the denoised output
                                // This implies init_latent is the "clean" initial latent.
                                // This might be an oversimplification / specific inpaint model behavior.
                                // For generic noise_mask, usually it's about preserving areas of init_latent *at current noise level*.
                                // The provided python seems to blend the *final prediction* with a *noised initial latent*.
                                // The line in control_reference.py is:
                                // `real_mask = real_mask.permute(0, 2, 3, 1).reshape(b, h*w, c)` - this is for strength application.
                                // The `noise_mask` parameter to `sample` in `stable-diffusion.cpp` seems to be for inpainting logic from `img2img`.
                                // Let's follow the existing img2img logic for noise_mask:
                                float init_val = ggml_tensor_get_f32(init_latent, el_x, el_y, el_k, el_b); // Original init_latent (possibly from VAE encode)
                                float den_val  = ggml_tensor_get_f32(denoised_pred, el_x, el_y, el_k, el_b);
                                // The Python code was: `real_bank[idx] = real_bank[idx] * effective_strength + context_attn1 * (1-effective_strength)`
                                // This isn't directly analogous to noise_mask here.
                                // The existing img2img `masked_image` logic sets values in `cond.c_concat` or `uncond.c_concat`.
                                // The `noise_mask` parameter to `sample` is for a different purpose, usually for compositing
                                // the denoised result with a noised version of the initial image in masked areas.
                                // If we follow ComfyUI's "Apply Latent" node with a mask, it's more like:
                                // new_latent = original_latent * mask + new_denoised_step_result * (1.0 - mask)
                                // Let's assume noise_mask means "where to apply denoising". 1.0 = apply, 0.0 = keep.
                                // But current_x_t is already noised. So it would be:
                                // current_x_t_after_step = current_x_t * (1-mask) + denoised_pred * mask (if mask=1 means use denoised)
                                // This needs clarification on `noise_mask`'s role here.
                                // The current C++ code in `sample` has:
                                // `ggml_tensor_set_f32(denoised, init + mask * (den - init), x, y, k);`
                                // This is `denoised = init_latent * (1-mask) + denoised_pred * mask`.
                                // This looks like it's trying to restore parts of `init_latent` (original unnoised VAE output of init_image for img2img)
                                // into the denoised output at each step. This is unusual for standard samplers.
                                // This line is usually applied *after* the whole sampling loop if inpainting from a clean image.
                                // For now, I will replicate this behavior if `noise_mask` is present.
                                // This means `denoised_pred` is blended with `init_latent`.
                                float blended_val = init_val * (1.0f - mask_val) + den_val * mask_val;
                                ggml_tensor_set_f32(denoised_pred, blended_val, el_x, el_y, el_k, el_b);

                            }
                        }
                    }
                }
            }


            return denoised_pred;
        };

        sample_k_diffusion(method, denoise_step_fn, work_ctx, x, sigmas, rng, eta); // x is updated in-place

        // Inverse scaling for the final latent
        x = denoiser->inverse_noise_scaling(sigmas[steps], x); // Use sigma at last step

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        diffusion_model->free_compute_buffer();
        ggml_free(tmp_ctx); // Free the temporary context for step-local tensors
        return x; // x now holds the final sampled latent x_0
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(work_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t C = 8;
        if (use_tiny_autoencoder) {
            C = 4;
        } else {
            if (sd_version_is_sd3(version)) {
                C = 32;
            } else if (sd_version_is_flux(version)) {
                C = 32;
            }
        }
        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : C,
                                                 x->ne[3]);  // channels
        int64_t t0          = ggml_time_ms();
        if (!use_tiny_autoencoder) {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 32x32 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    first_stage_model->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model->compute(n_threads, x, decode, &result);
            }
            first_stage_model->free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    tae_first_stage->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage->compute(n_threads, x, decode, &result);
            }
            tae_first_stage->free_compute_buffer();
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }

    // Prepare reference latent: Load image, resize, VAE encode.
    // Stores the result in `this->reference_latent_original` within `this->reference_latent_ctx`.
    // `target_latent_width` and `target_latent_height` are the W/8 and H/8 of the generation.
    bool prepare_reference_latent(int target_gen_width, int target_gen_height) {
        if (!reference_attn_enabled || reference_attn_image_path.empty()) {
            return true; // Nothing to do
        }
        if (reference_latent_original != NULL) {
             // Assuming if it exists, it's already correct for current W/H or doesn't need recomputing.
             // Or, we could add a check if target_gen_width/height changed.
             // For now, let's assume it's prepared once.
            return true;
        }

        LOG_INFO("Preparing reference latent from: %s", reference_attn_image_path.c_str());
        int ref_img_w = 0, ref_img_h = 0, ref_img_c = 0;
        uint8_t* ref_img_data = stbi_load(reference_attn_image_path.c_str(), &ref_img_w, &ref_img_h, &ref_img_c, 3);
        if (!ref_img_data) {
            LOG_ERROR("Failed to load reference image for attention: %s", reference_attn_image_path.c_str());
            return false;
        }

        // Create a temporary context for this operation
        struct ggml_init_params temp_params;
        temp_params.mem_size = 256LL * 1024 * 1024; // Generous temporary buffer
        temp_params.mem_buffer = NULL;
        temp_params.no_alloc = false;
        ggml_context* temp_work_ctx = ggml_init(temp_params);
        if (!temp_work_ctx) {
            LOG_ERROR("Failed to create temp_work_ctx for reference latent preparation.");
            stbi_image_free(ref_img_data);
            return false;
        }

        // Resize image to target generation dimensions (target_gen_width x target_gen_height)
        uint8_t* resized_ref_img_data = ref_img_data;
        if (ref_img_w != target_gen_width || ref_img_h != target_gen_height) {
            LOG_DEBUG("Resizing reference image from %dx%d to %dx%d", ref_img_w, ref_img_h, target_gen_width, target_gen_height);
            resized_ref_img_data = (uint8_t*)malloc(target_gen_width * target_gen_height * 3);
            if (!resized_ref_img_data) {
                LOG_ERROR("Failed to allocate memory for resized reference image.");
                stbi_image_free(ref_img_data); // Free original data since resized_ref_img_data was not allocated or is the same
                ggml_free(temp_work_ctx);
                return false;
            }
            // stbir_resize_uint8_srgb parameters:
            // input_pixels, input_w, input_h, input_stride_in_bytes,
            // output_pixels, output_w, output_h, output_stride_in_bytes,
            // num_channels
            int input_stride_bytes = ref_img_w * 3; // Assuming 3 channels (RGB)
            int output_stride_bytes = target_gen_width * 3; // Assuming 3 channels (RGB)
            int num_channels = 3; // RGB
            int alpha_channel_index = STBIR_ALPHA_CHANNEL_NONE; // No alpha channel
            int flags = 0; // Default flags

            stbir_resize_uint8_srgb(ref_img_data, ref_img_w, ref_img_h, input_stride_bytes,
                                    resized_ref_img_data, target_gen_width, target_gen_height, output_stride_bytes,
                                    num_channels, alpha_channel_index, flags);
            
            // If ref_img_data was different from resized_ref_img_data (i.e., malloc was successful and resize happened in-place on a new buffer)
            // then free the original stbi_load'd data.
            // However, our logic above means ref_img_data *is* the original if resize wasn't needed, or resized_ref_img_data is new.
            // The stbi_image_free(ref_img_data) was already there for the case malloc failed for resized.
            // If resize was successful into a new buffer, original `ref_img_data` needs freeing.
            // The logic was a bit tangled. Let's ensure original is freed if a new buffer was used.
            // This is simplified: if ref_img_data was original and resized_ref_img_data is the new buffer, free original.
            // The current code structure: resized_ref_img_data becomes the new primary buffer.
            // If resize was skipped, ref_img_data is used directly. If resize happened, resized_ref_img_data is used.
            // The stbi_image_free for original should be called if a new buffer `resized_ref_img_data` was actually used.

            // Corrected logic for freeing:
            if (resized_ref_img_data != ref_img_data) { // If a new buffer was allocated and used
                 stbi_image_free(ref_img_data); // Free the original one loaded by stbi_load
            }
            // Now, ref_img_data (if resize didn't happen) or resized_ref_img_data (if it did) is the one to use.
            // We renamed resized_ref_img_data for clarity to pass to sd_image_to_tensor.
            // The `resized_ref_img_data` variable now holds the pixel data to be used.
        }
        // At this point, `resized_ref_img_data` holds the correct pixel data (either original or resized)

        ggml_tensor* ref_image_tensor_rgb = ggml_new_tensor_4d(temp_work_ctx, GGML_TYPE_F32, target_gen_width, target_gen_height, 3, 1);
        sd_image_to_tensor(resized_ref_img_data, ref_image_tensor_rgb);
        if (resized_ref_img_data) free(resized_ref_img_data);


        // VAE Encode
        ggml_tensor* moments = encode_first_stage(temp_work_ctx, ref_image_tensor_rgb); // This uses first_stage_model or tae_first_stage
        ggml_tensor* vae_encoded_ref = get_first_stage_encoding(temp_work_ctx, moments); // This applies scale_factor

        // Copy to persistent reference_latent_ctx
        if (!reference_latent_ctx) { // Should have been created in load_from_file or constructor
            LOG_ERROR("reference_latent_ctx is null during prepare_reference_latent.");
            ggml_free(temp_work_ctx);
            return false;
        }
        reference_latent_original = ggml_dup_tensor(reference_latent_ctx, vae_encoded_ref);
        copy_ggml_tensor(reference_latent_original, vae_encoded_ref); // Ensure data is copied

        LOG_INFO("Reference latent prepared. Shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
                 reference_latent_original->ne[0], reference_latent_original->ne[1],
                 reference_latent_original->ne[2], reference_latent_original->ne[3]);

        ggml_free(temp_work_ctx);
        return true;
    }

};

/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};

sd_ctx_t* new_sd_ctx(const char* model_path_c_str,
                     const char* clip_l_path_c_str,
                     const char* clip_g_path_c_str,
                     const char* t5xxl_path_c_str,
                     const char* diffusion_model_path_c_str,
                     const char* vae_path_c_str,
                     const char* taesd_path_c_str,
                     const char* control_net_path_c_str,
                     const char* lora_model_dir_c_str,
                     const char* embed_dir_c_str,
                     const char* id_embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum sd_type_t wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_clip_on_cpu,
                     bool keep_control_net_cpu,
                     bool keep_vae_on_cpu,
                     bool diffusion_flash_attn,
                     // Reference Attention specific CLI params
                     const char* ref_attn_image_path_c_str,
                     float ref_attn_style_fidelity,
                     float ref_attn_strength
                     ) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string model_path(model_path_c_str);
    std::string clip_l_path(clip_l_path_c_str);
    std::string clip_g_path(clip_g_path_c_str);
    std::string t5xxl_path(t5xxl_path_c_str);
    std::string diffusion_model_path(diffusion_model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string id_embd_path(id_embed_dir_c_str);
    std::string lora_model_dir(lora_model_dir_c_str);

    std::string ref_attn_image_p(ref_attn_image_path_c_str ? ref_attn_image_path_c_str : "");
    ReferenceOptions_ggml ref_opts;
    ref_opts.attn_style_fidelity = ref_attn_style_fidelity;
    ref_opts.attn_strength = ref_attn_strength;


    sd_ctx->sd = new StableDiffusionGGML(n_threads,
                                         vae_decode_only,
                                         free_params_immediately,
                                         lora_model_dir,
                                         rng_type,
                                         ref_attn_image_p,
                                         ref_opts);
    if (sd_ctx->sd == NULL) {
        free(sd_ctx);
        return NULL;
    }

    if (!sd_ctx->sd->load_from_file(model_path,
                                    clip_l_path,
                                    clip_g_path,
                                    t5xxl_path_c_str,
                                    diffusion_model_path,
                                    vae_path,
                                    control_net_path,
                                    embd_path,
                                    id_embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_clip_on_cpu,
                                    keep_control_net_cpu,
                                    keep_vae_on_cpu,
                                    diffusion_flash_attn)) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
        free(sd_ctx);
        return NULL;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx != NULL) {
        if (sd_ctx->sd != NULL) {
            delete sd_ctx->sd;
            sd_ctx->sd = NULL;
        }
        free(sd_ctx);
    }
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx, // work_ctx for this generation run
                           ggml_tensor* init_latent,      // Initial latent for img2img or blank for txt2img
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           float eta,
                           int width,                    // Target generation width
                           int height,                   // Target generation height
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_ratio,
                           bool normalize_input,
                           std::string input_id_images_path,
                           std::vector<int> skip_layers = {},
                           float slg_scale              = 0,
                           float skip_layer_start       = 0.01,
                           float skip_layer_end         = 0.2,
                           ggml_tensor* masked_image    = NULL) { // Mask for inpainting or img2img guidance
    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0_lora = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1_lora = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1_lora - t0_lora) * 1.0f / 1000);

    // Photo Maker
    std::string prompt_text_only;
    ggml_tensor* pmid_init_img_tensor = NULL; // Renamed to avoid conflict
    SDCondition id_cond;
    std::vector<bool> class_tokens_mask;
    if (sd_ctx->sd->stacked_id) {
        if (!sd_ctx->sd->pmid_lora->applied) {
            t0_lora = ggml_time_ms();
            sd_ctx->sd->pmid_lora->apply(sd_ctx->sd->tensors, sd_ctx->sd->version, sd_ctx->sd->n_threads);
            t1_lora                             = ggml_time_ms();
            sd_ctx->sd->pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1_lora - t0_lora) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_lora->free_params_buffer();
            }
        }
        // preprocess input id images
        std::vector<sd_image_t*> input_id_images;
        bool pmv2 = sd_ctx->sd->pmid_model->get_version() == PM_VERSION_2;
        if (sd_ctx->sd->pmid_model && input_id_images_path.size() > 0) {
            std::vector<std::string> img_files = get_files_from_dir(input_id_images_path);
            for (std::string img_file : img_files) {
                int c = 0;
                int img_w, img_h; // Use different vars for loaded image dims
                if (ends_with(img_file, "safetensors")) {
                    continue;
                }
                uint8_t* input_image_buffer = stbi_load(img_file.c_str(), &img_w, &img_h, &c, 3);
                if (input_image_buffer == NULL) {
                    LOG_ERROR("PhotoMaker load image from '%s' failed", img_file.c_str());
                    continue;
                } else {
                    LOG_INFO("PhotoMaker loaded image from '%s'", img_file.c_str());
                }
                sd_image_t* current_pm_input_image = NULL; // Renamed
                current_pm_input_image             = new sd_image_t{(uint32_t)img_w, // Use loaded dims
                                             (uint32_t)img_h,
                                             3,
                                             input_image_buffer};
                current_pm_input_image             = preprocess_id_image(current_pm_input_image);
                if (current_pm_input_image == NULL) {
                    LOG_ERROR("preprocess input id image from '%s' failed", img_file.c_str());
                    continue;
                }
                input_id_images.push_back(current_pm_input_image);
            }
        }
        if (input_id_images.size() > 0) {
            sd_ctx->sd->pmid_model->style_strength = style_ratio;
            int32_t pmid_w                              = input_id_images[0]->width;
            int32_t pmid_h                              = input_id_images[0]->height;
            int32_t channels                       = input_id_images[0]->channel;
            int32_t num_input_images               = (int32_t)input_id_images.size();
            pmid_init_img_tensor                               = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, pmid_w, pmid_h, channels, num_input_images);
            float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            for (int i = 0; i < num_input_images; i++) {
                sd_image_t* pmid_image_data = input_id_images[i]; // Renamed
                if (normalize_input)
                    sd_mul_images_to_tensor(pmid_image_data->data, pmid_init_img_tensor, i, mean, std);
                else
                    sd_mul_images_to_tensor(pmid_image_data->data, pmid_init_img_tensor, i, NULL, NULL);
            }
            t0_lora                            = ggml_time_ms();
            auto cond_tup                 = sd_ctx->sd->cond_stage_model->get_learned_condition_with_trigger(work_ctx,
                                                                                                             sd_ctx->sd->n_threads, prompt,
                                                                                                             clip_skip,
                                                                                                             width, // Use generation width/height
                                                                                                             height,
                                                                                                             num_input_images,
                                                                                                             sd_ctx->sd->diffusion_model->get_adm_in_channels());
            id_cond                       = std::get<0>(cond_tup);
            class_tokens_mask             = std::get<1>(cond_tup);  //
            struct ggml_tensor* id_embeds = NULL;
            if (pmv2) {
                id_embeds = load_tensor_from_file(work_ctx, path_join(input_id_images_path, "id_embeds.bin"));
            }
            id_cond.c_crossattn = sd_ctx->sd->id_encoder(work_ctx, pmid_init_img_tensor, id_cond.c_crossattn, id_embeds, class_tokens_mask);
            t1_lora                  = ggml_time_ms();
            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1_lora - t0_lora);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_model->free_params_buffer();
            }
            prompt_text_only = sd_ctx->sd->cond_stage_model->remove_trigger_from_prompt(work_ctx, prompt);
            prompt = prompt_text_only;
        } else {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images. Turning off PhotoMaker.");
            sd_ctx->sd->stacked_id = false;
        }
        for (sd_image_t* img_ptr : input_id_images) { // Renamed loop var
             if(img_ptr->data) free(img_ptr->data); // Free image data
             delete img_ptr; // Free sd_image_t struct itself
        }
        input_id_images.clear();
    }
    
    // Reference Attention: Prepare original reference latent if enabled and not already done
    if (sd_ctx->sd->reference_attn_enabled && sd_ctx->sd->reference_latent_original == NULL) {
        if (!sd_ctx->sd->prepare_reference_latent(width, height)) {
            LOG_ERROR("Failed to prepare reference latent. Disabling reference attention for this run.");
            sd_ctx->sd->reference_attn_enabled = false; // Disable if preparation fails
        }
    }


    // Get learned condition
    int64_t t0_cond = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }
    int64_t t1_cond = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1_cond - t0_cond);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    ggml_tensor* current_noise_mask = nullptr; // Renamed from noise_mask
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        if (masked_image == NULL) { // If no specific mask provided for inpaint model
            int64_t mask_channels = 1;
            if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                mask_channels = 8 * 8;
            }
            masked_image = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
            for (int64_t ix = 0; ix < masked_image->ne[0]; ix++) {
                for (int64_t iy = 0; iy < masked_image->ne[1]; iy++) {
                    if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                        for (int64_t k_ch = 0; k_ch < init_latent->ne[2]; k_ch++) { // Renamed k
                            ggml_tensor_set_f32(masked_image, 0, ix, iy, k_ch);
                        }
                        for (int64_t k_ch = init_latent->ne[2]; k_ch < masked_image->ne[2]; k_ch++) {
                            ggml_tensor_set_f32(masked_image, 1, ix, iy, k_ch);
                        }
                    } else {
                        ggml_tensor_set_f32(masked_image, 1, ix, iy, 0);
                        for (int64_t k_ch = 1; k_ch < masked_image->ne[2]; k_ch++) {
                            ggml_tensor_set_f32(masked_image, 0, ix, iy, k_ch);
                        }
                    }
                }
            }
        }
        cond.c_concat   = masked_image; // This masked_image is the one prepared for inpainting model input
        uncond.c_concat = masked_image;
        current_noise_mask = nullptr; // Not used in the same way for inpaint models; already part of c_concat
    } else {
        current_noise_mask = masked_image; // For non-inpaint models, masked_image might be used as noise_mask
    }

    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent; // This is the starting latent (blank or from img2img)
        struct ggml_tensor* noise_for_sampling = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1); // Renamed
        ggml_tensor_set_f32_randn(noise_for_sampling, sd_ctx->sd->rng);

        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t, // Starting latent for this batch item
                                                     noise_for_sampling, // Initial noise for this batch item
                                                     cond,
                                                     uncond,
                                                     image_hint,
                                                     control_strength,
                                                     cfg_scale, // min_cfg (used for batch diff CFG)
                                                     cfg_scale, // cfg_scale
                                                     guidance,
                                                     eta,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     id_cond,
                                                     skip_layers,
                                                     slg_scale,
                                                     skip_layer_start,
                                                     skip_layer_end,
                                                     current_noise_mask); // Pass the correctly determined mask

        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3_sample = ggml_time_ms(); // Renamed
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3_sample - t1_cond) * 1.0f / 1000); // t1_cond was end of conditioning

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t0_decode                      = ggml_time_ms(); // Renamed
        struct ggml_tensor* img_decoded = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */); // Renamed
        if (img_decoded != NULL) {
            decoded_images.push_back(img_decoded);
        }
        int64_t t1_decode = ggml_time_ms(); // Renamed
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t1_decode - t0_decode) * 1.0f / 1000);
    }

    int64_t t4_decode = ggml_time_ms(); // Renamed
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4_decode - t3_sample) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx); // work_ctx should be freed by caller of generate_image
        return NULL;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(decoded_images[i]);
    }
    // ggml_free(work_ctx); // Caller of generate_image is responsible for freeing work_ctx

    return result_images;
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    float eta,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str,
                    int* skip_layers         = NULL,
                    size_t skip_layers_count = 0,
                    float slg_scale          = 0,
                    float skip_layer_start   = 0.01,
                    float skip_layer_end     = 0.2) {
    std::vector<int> skip_layers_vec(skip_layers, skip_layers + skip_layers_count);
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params_work_ctx; // Renamed
    params_work_ctx.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        params_work_ctx.mem_size *= 3;
    }
    if (sd_version_is_flux(sd_ctx->sd->version)) {
        params_work_ctx.mem_size *= 4;
    }
    if (sd_ctx->sd->stacked_id) {
        params_work_ctx.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB for pmid_init_img_tensor
    }
     if (sd_ctx->sd->reference_attn_enabled && !sd_ctx->sd->reference_attn_image_path.empty()) {
        // Add memory for reference latent related operations if not already covered
        // Estimate: original ref latent + noised ref latent + scaled noised ref latent
        // Assuming max 1024x1024 -> 128x128x4 latent (approx 256KB) * 3 = ~768KB per batch item.
        // This might already be covered by the general large buffer, but good to be mindful.
        params_work_ctx.mem_size += static_cast<size_t>(5 * 1024 * 1024); // Extra buffer for reference ops
    }
    params_work_ctx.mem_size += width * height * 3 * sizeof(float); // For potential RGB image tensors
    params_work_ctx.mem_size *= batch_count; // Scale by batch count if ops are per batch item
    params_work_ctx.mem_buffer = NULL;
    params_work_ctx.no_alloc   = false;

    struct ggml_context* work_ctx = ggml_init(params_work_ctx);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed for work_ctx");
        return NULL;
    }

    size_t t0_txt2img = ggml_time_ms(); // Renamed

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    int C = 4;
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        C = 16;
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        C = 16;
    }
    int W                    = width / 8;
    int H                    = height / 8;
    ggml_tensor* init_latent_txt2img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1); // Renamed
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent_txt2img, 0.0609f);
    } else if (sd_version_is_flux(sd_ctx->sd->version)) {
        ggml_set_f32(init_latent_txt2img, 0.1159f);
    } else {
        ggml_set_f32(init_latent_txt2img, 0.f);
    }

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        LOG_WARN("This is an inpainting model, this should only be used in img2img mode with a mask");
    }

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx, // Pass work_ctx
                                               init_latent_txt2img,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               eta,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str,
                                               skip_layers_vec,
                                               slg_scale,
                                               skip_layer_start,
                                               skip_layer_end,
                                               NULL); // No masked_image for txt2img

    size_t t1_txt2img = ggml_time_ms(); // Renamed

    LOG_INFO("txt2img completed in %.2fs", (t1_txt2img - t0_txt2img) * 1.0f / 1000);
    
    ggml_free(work_ctx); // Free work_ctx after generation
    return result_images;
}

sd_image_t* img2img(sd_ctx_t* sd_ctx,
                    sd_image_t init_image,
                    sd_image_t mask,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    float eta,
                    int width,
                    int height,
                    sample_method_t sample_method,
                    int sample_steps,
                    float strength,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str,
                    int* skip_layers         = NULL,
                    size_t skip_layers_count = 0,
                    float slg_scale          = 0,
                    float skip_layer_start   = 0.01,
                    float skip_layer_end     = 0.2) {
    std::vector<int> skip_layers_vec(skip_layers, skip_layers + skip_layers_count);
    LOG_DEBUG("img2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params_work_ctx_img2img; // Renamed
    params_work_ctx_img2img.mem_size = static_cast<size_t>(10 * 1024 * 1024);
    if (sd_version_is_sd3(sd_ctx->sd->version)) {
        params_work_ctx_img2img.mem_size *= 2;
    }
    if (sd_version_is_flux(sd_ctx->sd->version)) {
        params_work_ctx_img2img.mem_size *= 3;
    }
    if (sd_ctx->sd->stacked_id) {
        params_work_ctx_img2img.mem_size += static_cast<size_t>(10 * 1024 * 1024);
    }
     if (sd_ctx->sd->reference_attn_enabled && !sd_ctx->sd->reference_attn_image_path.empty()) {
        params_work_ctx_img2img.mem_size += static_cast<size_t>(5 * 1024 * 1024);
    }
    params_work_ctx_img2img.mem_size += width * height * 3 * sizeof(float) * 3; // init_img, mask_img, masked_img
    params_work_ctx_img2img.mem_size *= batch_count;
    params_work_ctx_img2img.mem_buffer = NULL;
    params_work_ctx_img2img.no_alloc   = false;

    struct ggml_context* work_ctx = ggml_init(params_work_ctx_img2img); // Renamed
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed for work_ctx (img2img)");
        return NULL;
    }

    size_t t0_img2img = ggml_time_ms(); // Renamed

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    ggml_tensor* init_img_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1); // Renamed
    ggml_tensor* mask_img_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 1, 1); // Renamed

    sd_mask_to_tensor(mask.data, mask_img_tensor);
    sd_image_to_tensor(init_image.data, init_img_tensor);

    ggml_tensor* processed_masked_image_for_cond; // Renamed

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        int64_t mask_channels = 1;
        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            mask_channels = 8 * 8;
        }
        ggml_tensor* masked_rgb_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1); // Renamed
        sd_apply_mask(init_img_tensor, mask_img_tensor, masked_rgb_img);
        ggml_tensor* masked_latent_0 = NULL; // Renamed
        if (!sd_ctx->sd->use_tiny_autoencoder) {
            ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, masked_rgb_img);
            masked_latent_0       = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
        } else {
            masked_latent_0 = sd_ctx->sd->encode_first_stage(work_ctx, masked_rgb_img);
        }
        // This becomes the c_concat for inpainting models
        processed_masked_image_for_cond = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, masked_latent_0->ne[0], masked_latent_0->ne[1], mask_channels + masked_latent_0->ne[2], 1);
        for (int ix = 0; ix < masked_latent_0->ne[0]; ix++) {
            for (int iy = 0; iy < masked_latent_0->ne[1]; iy++) {
                int mx = ix * 8;
                int my = iy * 8;
                if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                    for (int k_ch = 0; k_ch < masked_latent_0->ne[2]; k_ch++) {
                        float v = ggml_tensor_get_f32(masked_latent_0, ix, iy, k_ch);
                        ggml_tensor_set_f32(processed_masked_image_for_cond, v, ix, iy, k_ch);
                    }
                    for (int x_sub = 0; x_sub < 8; x_sub++) { // Renamed
                        for (int y_sub = 0; y_sub < 8; y_sub++) { // Renamed
                            float m = ggml_tensor_get_f32(mask_img_tensor, mx + x_sub, my + y_sub);
                            ggml_tensor_set_f32(processed_masked_image_for_cond, m, ix, iy, masked_latent_0->ne[2] + x_sub * 8 + y_sub);
                        }
                    }
                } else {
                    float m = ggml_tensor_get_f32(mask_img_tensor, mx, my);
                    ggml_tensor_set_f32(processed_masked_image_for_cond, m, ix, iy, 0);
                    for (int k_ch = 0; k_ch < masked_latent_0->ne[2]; k_ch++) {
                        float v = ggml_tensor_get_f32(masked_latent_0, ix, iy, k_ch);
                        ggml_tensor_set_f32(processed_masked_image_for_cond, v, ix, iy, k_ch + mask_channels);
                    }
                }
            }
        }
    } else {
        // For non-inpaint models, `masked_image` is used as `noise_mask` in `sample`
        // It needs to be W/8, H/8
        processed_masked_image_for_cond = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 1, 1);
        for (int ix = 0; ix < processed_masked_image_for_cond->ne[0]; ix++) {
            for (int iy = 0; iy < processed_masked_image_for_cond->ne[1]; iy++) {
                int mx  = ix * 8;
                int my  = iy * 8;
                float m = ggml_tensor_get_f32(mask_img_tensor, mx, my); // Sample from original mask
                ggml_tensor_set_f32(processed_masked_image_for_cond, m, ix, iy);
            }
        }
    }

    ggml_tensor* initial_img_latent = NULL; // Renamed
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img_tensor);
        initial_img_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        initial_img_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img_tensor);
    }

    // print_ggml_tensor(initial_img_latent, true); // Use new name
    size_t t1_img2img_encode = ggml_time_ms(); // Renamed
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1_img2img_encode - t0_img2img) * 1.0f / 1000);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    if (t_enc >= sample_steps) // Ensure t_enc is less than sample_steps
        t_enc = sample_steps - 1;
    if (t_enc == 0 && strength > 0.0f) // Ensure at least one step if strength > 0
        t_enc = 1;
    
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    if (t_enc > 0) { // Only schedule if there are steps to take
      sigma_sched.assign(sigmas.begin() + sample_steps - t_enc -1 , sigmas.end());
    } else { // If t_enc is 0, it means strength is 0, effectively no denoising from img2img noise
      // Result should be very close to VAE decode of init_image.
      // Use a minimal schedule to pass through the sampling logic if needed, or handle as special case.
      // For now, let's use last sigma if t_enc is 0, implying minimal change.
      if (!sigmas.empty()) sigma_sched.push_back(sigmas.back());
    }


    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx, // Pass work_ctx
                                               initial_img_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               eta,
                                               width,
                                               height,
                                               sample_method,
                                               sigma_sched, // Use the strength-adjusted schedule
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str,
                                               skip_layers_vec,
                                               slg_scale,
                                               skip_layer_start,
                                               skip_layer_end,
                                               processed_masked_image_for_cond); // Pass the correctly prepared mask for inpainting or noise guidance

    size_t t2_img2img = ggml_time_ms(); // Renamed

    LOG_INFO("img2img completed in %.2fs", (t2_img2img - t0_img2img) * 1.0f / 1000);

    ggml_free(work_ctx); // Free work_ctx after generation
    return result_images;
}

SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           int width,
                           int height,
                           int video_frames,
                           int motion_bucket_id,
                           int fps,
                           float augmentation_level,
                           float min_cfg,
                           float cfg_scale,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed) {
    if (sd_ctx == NULL) {
        return NULL;
    }

    LOG_INFO("img2vid %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    struct ggml_init_params params_work_ctx_img2vid; // Renamed
    params_work_ctx_img2vid.mem_size = static_cast<size_t>(10 * 1024) * 1024;
    params_work_ctx_img2vid.mem_size += width * height * 3 * sizeof(float) * video_frames;
    params_work_ctx_img2vid.mem_buffer = NULL;
    params_work_ctx_img2vid.no_alloc   = false;

    struct ggml_context* work_ctx = ggml_init(params_work_ctx_img2vid); // Renamed
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed for work_ctx (img2vid)");
        return NULL;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd_ctx->sd->rng->manual_seed(seed);

    int64_t t0_img2vid = ggml_time_ms(); // Renamed

    SDCondition cond = sd_ctx->sd->get_svd_condition(work_ctx,
                                                     init_image,
                                                     width,
                                                     height,
                                                     fps,
                                                     motion_bucket_id,
                                                     augmentation_level);

    auto uc_crossattn = ggml_dup_tensor(work_ctx, cond.c_crossattn);
    ggml_set_f32(uc_crossattn, 0.f);

    auto uc_concat = ggml_dup_tensor(work_ctx, cond.c_concat);
    ggml_set_f32(uc_concat, 0.f);

    auto uc_vector = ggml_dup_tensor(work_ctx, cond.c_vector);

    SDCondition uncond = SDCondition(uc_crossattn, uc_vector, uc_concat);

    int64_t t1_img2vid_cond = ggml_time_ms(); // Renamed
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1_img2vid_cond - t0_img2vid);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->clip_vision->free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t_vid = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames); // Renamed
    ggml_set_f32(x_t_vid, 0.f);

    struct ggml_tensor* noise_vid = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames); // Renamed
    ggml_tensor_set_f32_randn(noise_vid, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0_vid = sd_ctx->sd->sample(work_ctx, // Renamed
                                                 x_t_vid,
                                                 noise_vid,
                                                 cond,
                                                 uncond,
                                                 NULL, // No control_hint for SVD sampling
                                                 0.f,  // control_strength
                                                 min_cfg,
                                                 cfg_scale,
                                                 0.f,  // guidance (not typically used in SVD like this)
                                                 0.f,  // eta
                                                 sample_method,
                                                 sigmas,
                                                 -1,   // start_merge_step
                                                 SDCondition(NULL, NULL, NULL) // No id_cond for SVD
                                                 ); 

    int64_t t2_img2vid_sample = ggml_time_ms(); // Renamed
    LOG_INFO("sampling completed, taking %.2fs", (t2_img2vid_sample - t1_img2vid_cond) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    struct ggml_tensor* img_decoded_vid = sd_ctx->sd->decode_first_stage(work_ctx, x_0_vid); // Renamed
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (img_decoded_vid == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(video_frames, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < video_frames; i++) {
        auto img_i = ggml_view_3d(work_ctx, img_decoded_vid, img_decoded_vid->ne[0], img_decoded_vid->ne[1], img_decoded_vid->ne[2], img_decoded_vid->nb[1], img_decoded_vid->nb[2], img_decoded_vid->nb[3] * i);

        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img_i);
    }
    ggml_free(work_ctx);

    int64_t t3_img2vid = ggml_time_ms(); // Renamed

    LOG_INFO("img2vid completed in %.2fs", (t3_img2vid - t0_img2vid) * 1.0f / 1000);

    return result_images;
}