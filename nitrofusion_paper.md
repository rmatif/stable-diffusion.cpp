# NitroFusion: High-Fidelity Single-Step Diffusion through Dynamic Adversarial Training

## Abstract

We introduce NitroFusion, a fundamentally different approach to single-step diffusion that achieves high-quality generation through a dynamic adversarial framework. While one-step methods offer dramatic speed advantages, they typically suffer from quality degradation compared to their multi-step counterparts. Just as a panel of art critics provides comprehensive feedback by specializing in different aspects like composition, color, and technique, our approach maintains a large pool of specialized discriminator heads that collectively guide the generation process. Each discriminator group develops expertise in specific quality aspects at different noise levels, providing diverse feedback that enables high-fidelity one-step generation. Our framework combines: (i) a dynamic discriminator pool with specialized discriminator groups to improve generation quality, (ii) strategic refresh mechanisms to prevent discriminator overfitting, and (iii) global-local discriminator heads for multi-scale quality assessment, and unconditional/conditional training for balanced generation. Additionally, our framework uniquely supports flexible deployment through bottom-up refinement, allowing users to dynamically choose between 1-4 denoising steps with the same model for direct quality-speed trade-offs. Through comprehensive experiments, we demonstrate that NitroFusion significantly outperforms existing single-step methods across multiple evaluation metrics, particularly excelling in preserving fine details and global consistency.

## 1. Introduction

Recent advances in accelerated diffusion models have demonstrated that high-quality image generation is possible with dramatically reduced step counts. While several approaches now achieve one-step generation, they face significant challenges in matching the quality of multi-step methods, particularly in preserving fine details and ensuring global coherence. This quality gap has limited the practical adoption of single-step methods, especially in applications requiring both speed and high fidelity.

The core challenge in single-step diffusion lies in compressing an entire denoising trajectory into a single transformation. Traditional approaches based on distillation struggle because they attempt to directly match intermediate states or distributions, leading to blurry outputs and loss of detail. Recent adversarial methods show promise but face training instability and diversity collapse when pushed to single-step generation.

NitroFusion introduces a fundamentally different approach to single-step diffusion through a dynamic adversarial framework. Consider how a panel of art critics evaluates a painting - each critic specializes in different aspects like composition, color, technique, and detail. Similarly, rather than relying on a single discriminator that can quickly become overconfident, we maintain a large, dynamic pool of specialized discriminator groups that operate on top of a frozen UNet backbone. Just as a diverse panel of critics provides more comprehensive feedback than a single judge, our ensemble of discriminators guides the generator toward high-quality outputs by providing specialized feedback at different noise levels and spatial scales.

Our framework implements this insight through three technical innovations: (i) a dynamic discriminator pool architecture where we leverage the teacher model's UNet encoder as a frozen feature extractor, with multiple lightweight discriminator groups Ht* specialized for different noise levels t* to improve generation quality, (ii) a strategic refresh mechanism that randomly re-initializes ~1% of discriminator heads while preserving the collective knowledge distribution across the pool to prevent discriminator overfitting – a common failure mode in GAN training – while maintaining stable adversarial feedback, and (iii) a multi-scale strategy with dual training objectives where global heads and local heads are compartmentalized in a 1:2 ratio, with global heads assessing overall image coherence at resolution H×W and local heads examining fine-grained details in patches of size hxw. These are further divided as unconditional and prompt-conditional discriminator heads (dual-training) effectively balancing prompt alignment with image coherence.

These technical components work together to solve the fundamental challenges of single-step generation. The dynamic discriminator pool and refresh mechanism work in tandem to maintain a balanced feedback system throughout training - as established heads provide consistent feedback, the periodic introduction of new heads prevents the system from becoming too rigid or predictable. The multi-scale strategy then complements this dynamic feedback system, enabling our generator to achieve what previous approaches could not: transforming noise into high-quality images in a single step while avoiding the artifacts and quality degradation that typically plague fast generation methods.

Notably, unlike existing approaches that require separate models for different step counts, our framework uniquely supports flexible deployment through bottom-up refinement. While we optimize primarily for single-step generation, our model uniquely enables dynamic refinement – users can simply add steps (up to 4) on-demand if higher quality is desired, all with the same model weights.

Through extensive experimentation, we demonstrate that NitroFusion consistently produces sharper, more detailed images than existing single-step methods. Our approach not only matches but often exceeds the quality metrics of recent fast diffusion models while maintaining the speed advantages of single-step generation. Human evaluation studies further confirm the superior visual quality of our results, particularly in challenging areas like face detail and texture preservation.

Our key contributions include: (i) a dynamic discriminator pool with specialized discriminator groups to improve generation quality, (ii) strategic refresh mechanisms to prevent discriminator overfitting, and (iii) multi-scale strategy with dual training objectives to effectively balance prompt alignment and image coherence. Additionally, we uniquely enable flexible deployment by supporting 1-4 denoising steps with the same model weights.

## 2. Related Works

### 2.1. Timestep Distillation

Timestep distillation accelerates inference in diffusion models by reducing the required sampling steps for high-quality output. Standard approaches distil a multi-step teacher model into a student model with fewer steps. A common strategy is to approximate the sampling trajectory, modeled as an ordinary differential equation (ODE), of the teacher model in a reduced step count. This can be implemented by either preserving the original ODE path at each timestep, or re-formulating and learning a more efficient trajectory directly from the final outputs. Recent works train a series of such student models that progressively lower sampling steps, while enforcing self-consistency. Hyper-SD further combines ODE-preserving and -reformulating methods. However, these models often face quality degradation due to limited model fitting capacity. Different from flow-guided distillation, Distribution Matching Distillation (DMD) minimizes the Kullback-Leibler (KL) divergence between generated and target distributions to directly match distributions on the sample domain. Despite these advancements, achieving high fidelity in one-step distillation remains challenging, as these models frequently struggle with degradation and instability in extreme low-step settings.

### 2.2. Adversarial Distillation

Adversarial Diffusion Distillation (ADD) incorporates GAN training to address the limitations of MSE-based distillation in the few-step generation, which often leads to blurry outputs. Generally, a pretrained feature extractor is used as the discriminator backbone to obtain stable, discriminative features. SDXL-Lightning for instance, uses the encoder of a pretrained diffusion model as the discriminator backbone, injecting noise prior to the real-vs-fake judgment as a form of augmentation. Recent works further integrate adversarial loss with distillation objectives to improve image fidelity. However, adversarial loss introduces its own challenges, including training instability and reduced diversity. Rapid discriminator learning can lead to overconfident assessments, limiting constructive feedback for the generator and causing suboptimal training dynamics. Overcoming these challenges is a primary goal of our work.

### 2.3. Multi-Discriminator Training

GANs with multiple discriminators have reduced mode collapse and enhanced training stability through the incorporation of diverse adversarial feedback. Various strategies have been developed to balance multiple discriminator objectives, including softmax-weighted ensembles and three-player minimax games. To address over-confidence in discriminators, Neyshabur et al. applies lower-dimensional random projection for each discriminator, while MCL-GAN incorporates multiple choice learning. StyleGAN-XL and StyleGAN-T use multiple discriminator heads alongside a frozen, pretrained backbone, enabling feedback across feature pyramids to capture various levels of detail. While these multi-discriminator methods address challenges in GAN training, they remain under-explored in diffusion distillation. Our approach builds upon these insights, introducing a robust adversarial framework to provide diverse and dynamic feedback for high-fidelity one-step diffusion distillation.

## 3. Methodology

To perform one-step diffusion, we utilize the concept of timestep distillation. In here, a one-step student model is trained to perform at par with a pre-trained multi-step teacher. After training, the one-step student can be used independently for super-fast inference. Unlike conventional methods that rely on score matching or flow matching to align student and teacher quality, our approach uses adversarial loss only for critiquing teacher and student predictions - akin to a panel of critics that evaluate paintings. This helps us align teacher and student distributions for the student to mimic the teacher in a single step without quality degradation.

Specifically, we propose a Dynamic Adversarial Framework, as: (i) A huge pool of discriminator heads with specialized discriminators for different levels of noise and quality, reducing feedback bias from an otherwise single discriminator set-up. (ii) A periodic pool refresh to randomly re-initialize a sampled set of discriminators to prevent overfitting, and (iii) multi-scale dual-objective GAN training to reduce artifacts and balance image coherence with prompt alignment.

**Preliminaries:** Diffusion Models iteratively refine noise in a data sample by reversing a forward process that progressively transforms an input sample x0 into noise. In this forward process, each noisy sample xt is obtained from x0 using Gaussian noise e ~ N(0,I) at timestep t∈ {1, ..., T} as:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}e
$$

where αt is a variance schedule controlling the noise level. The reverse process, parameterized by a neural network Gθ, is trained to predict the noise e from xt to reconstruct x0. Using the predicted noise ê = Gθ(xt, t), x0 is reconstructed as:

$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\hat{e}}{\sqrt{\bar{\alpha}_t}}
$$

### 3.1. One-Step Adversarial Diffusion Distillation

Our training pipeline consists of a one-step student (generator) Gθ, and a pretrained multi-step teacher model Gψ. We initialize the student with pre-trained one-step weights θ0, to reduce the time to converge. During each training iteration, Gθ and Gψ denoise a noisy sample xT ~ N(0,I) to x̂0 and x0 respectively. While this denoising takes multiple steps for the teacher Gψ, our student Gθ directly denoises xT to x̂0 in one step only. The discriminator D attempts to distinguish x0 as real and x̂0 as fake, constructing the adversarial loss Ladv.

$$
L_{adv}^D = -E[D(x_0)]
$$

$$
L_{adv}^G = E[D(\hat{x}_0) - D(x_0))]
$$

### 3.2. Dynamic Discriminator Pool

Building on previous works, we utilize the teacher's UNet encoder and mid-block as a frozen discriminator backbone E that extracts image features. This generally entails first noising inputs x0 to pre-defined noise levels t* as x*t* and then using their denoising signals E(x*t*,t*) as visual features. Different levels of the UNet encoder E provide feature representations at different levels, spanning from low-level details to high-level semantics. A lightweight trainable discriminator head is attached at each such level of the backbone E for the discriminator to perform real/fake classification.

As a core building block of our pipeline, we use a dynamic discriminator pool P to source these discriminator heads. This discriminator pool P is a huge pool of constantly evolving discriminator heads that can be attached to E for our pipeline's multi-head discriminator. The lightweight design of these heads allows us to scale the pool without significant computational or memory overhead. For training the pool, we sample a subset of heads D ~ P from the pool at every training iteration, computing the adversarial loss Ladv with this subset. We backpropagate gradients from Ladv to optimize the sampled heads D. After the update, we release the heads back into the pool to evolve the global knowledge of the pool dynamically. The stochasticity of this process through random sampling ensures varied feedback, preventing any single head from dominating the generator's learning and reducing bias. This diversifies feedback and enhances stability in GAN training.

To construct specialized discriminator heads we compartmentalize the pool P based on the noise level of the discriminator timestep t* as {Pt* ∈ P ∀ t*}. This helps us sample discriminator heads Dt* ~ Pt* that are specialized for a specific noise level at discriminator timestep t*. Unlike prior approaches that treat timestep-dependent discriminators as augmentation or smoothing techniques, each head in our pool functions as an expert on its designated noise level, providing precise, nuanced critiques targeting specific image characteristics. We calculate the adversarial loss as:

$$
L_{adv}^D = -E[\sum_{H \in D_{t^*}} H(\mathcal{E}(x_{t^*}^*, t^*))]
$$

$$
L_{adv}^G = E[\sum_{H \in D_{t^*}} H(\mathcal{E}(\hat{x}_{t^*}^*, t^*)) - H(\mathcal{E}(x_{t^*}^*, t^*))]
$$

where the frozen UNet encoder E extracts features for sampled discriminator heads Dt*. Intermediate outputs from each trainable-head H are aggregated for real/fake discriminator predictions.

### 3.3. Discriminator Pool Refresh

Early overfitting in GAN training limits the discriminator's feedback diversity, reducing the quality and variation of generated images. To address this, we introduce a random re-initialization strategy for our dynamic discriminator pool: at each training iteration, we discard (flush) a random subset (~1%) of discriminator heads, replacing (refreshing) them with re-initialized discriminators. Refreshing discriminator subsets helps maintain a balance between stable feedback from retained heads and variability from re-initialized ones to enhance generator performance.

### 3.4. Multi-Scale and Dual-Objective GAN Training

The generalization potential of diffusion models to multiple resolutions allows us to further use the pre-trained UNet encoder for both global and local (patch) discrimination. For this, we divide the pool into local and global heads, training them with adversarial feedback - to judge either the entire image, or fine-grained details respectively. This setup enables global-focused heads to assess structure and local-focused heads to capture textures, balancing macro and micro image details. We also introduce dual-objective GAN training which applies both conditional and unconditional adversarial loss. We motivate this training following prior analysis that confirms conditional generation to introduce "Janus" artifacts while struggling to align images with text features. Janus artifacts present repeated patterns, such as faces or hands, within a local area. To reduce such artifacts that manifest more in single-step diffusions, we use local discriminator heads to perform conditional and unconditional discrimination. Unconditional local heads provide feedback solely based on image coherence. This dual-objective approach prevents overfitting to specific prompt-driven features, reducing the likelihood of artifacts and delivering a balanced, generalized adversarial signal.

To summarize, we compartmentalize our pool of weights for each timestep t*, where further boundaries are created for different training settings: (i) global images with conditional discrimination, (ii) local patches with conditional discrimination and (iii) local patches with unconditional discrimination. Each of these pools has the same number of discriminator heads.

### 3.5. Bottom-Up Multi-Step Refinement

Unlike previous step-reduction algorithms, we offer a quality v/s speed trade-off, where users can perform denoising on one-step or multiple steps (up to 4) to have higher-quality generated images with the same model weights. We support this by using a bottom-up refinement approach, where we optimize the network for one step, and iteratively refine for multiple steps one by one. This significantly differs from the more traditional top-down approaches that iteratively refine for 8, 4, 2, and then 1 step in that order. Using a bottom-up refinement approach allows users to use the same model for multiple steps, and obtain gradually improving results from 1 to 4 steps.

## 4. Experiments

**Implementation Details:** Each discriminator head comprises 4 x 4 convolution layers with a stride of 2, group normalization, and SiLU activation. 10 heads work on 10 feature maps at different feature levels from a pretrained diffusion model's frozen backbone. We employ specific discriminator timesteps t* ∈ {10, 250, 500, 750}. We use a pool of 480 heads, using 160 for each of the task types (global conditional / local conditional / local unconditional). We train using the AdamW optimizer with a batch size of 5 and gradient accumulation over 20 steps on a single NVIDIA A100 GPU. Each iteration samples discriminator heads for real/fake classification from pool, with 1% reinitialized (during pool refresh) to maintain dynamic feedback. To demonstrate generalization across teacher models, we train two networks with distinct visual goals: **NitroSD-Realism**, optimized for photorealism with the 4-step DMD2 teacher; and **NitroSD-Vibrant**, for vivid colors with the 8-step Hyper-SDXL teacher.

**Data:** Following the hypothesis that synthetic images offer superior text alignment than real images, we train our models on synthetic samples only, generated by multi-step teacher models - without paired prompt-image data. Prompts are sourced from the Pick-a-Pic and LAION datasets, totaling one million.

**Baseline Models and Evaluation Metrics:** We compare our models to DMD2, Hyper-SDXL, the SDXL base model, and additional timesteps distillation methods like SDXL-Turbo and SDXL-Lightning. DMD2 proposes distribution matching distillation using KL-divergence to address limitations in flow-guided distillation. Hyper-SDXL uses human feedback to improve visual appeal of outputs. SDXL-Turbo and SDXL-Lighting introduce adversarial loss and timestep-dependent discriminator for low-step inference.

### 4.1. Qualitative Comparison

(Qualitative descriptions comparing NitroSD models against SDXL-Turbo, SDXL-Lightning, DMD2, Hyper-SDXL, and the base SDXL model, highlighting NitroSD's clarity, texture, artifact reduction, and alignment with teacher styles - realism for DMD2 teacher, vibrancy for Hyper-SDXL teacher.)

### 4.2. User Study

(Describes a two-choice preference user study where NitroSD-Realism and NitroSD-Vibrant were compared against baselines. Results indicated strong user preference for NitroSD models, especially NitroSD-Vibrant, even outperforming 25-step SDXL. 2-step NitroSD results were preferred over 4-step competitor results.)

### 4.3. Quantitative Comparison

We conduct a quantitative evaluation on the COCO-5K validation dataset, using several key metrics in Table 1: CLIP score (ViT-B/32), which assesses prompt alignment; Fréchet Inception Distance (FID), which evaluates image quality and diversity; Aesthetic Score, which is trained on user preferences to quantify visual appeal; and ImageReward score, which reflects potential user preferences.

While FID and CLIP scores for our models are competitive, NitroSD particularly excels in advanced metrics: Aesthetic Score and Image Reward. NitroSD-Realism outperforms its teacher DMD2 both in Aesthetic Score and Image Reward, two metrics capturing image appeal and text alignment based on user preference. NitroSD-Vibrant also achieves one of the highest scores in these two metrics, reflecting its capability to produce visually engaging images that align with user preferences. These advanced metrics highlight NitroSD's strengths in subjective quality, a critical factor in text-to-image generation. When paired with our user study findings, these results confirm that NitroSD effectively balances fast inference with high user satisfaction, offering a practical solution for applications that demand both efficiency and aesthetic appeal.

| Model               | Steps | CLIP (↑) | FID (↓) | Aesthetic Score (↑) | Image Reward(↑) |
| :------------------ | :---- | :------- | :------ | :------------------ | :-------------- |
| SDXL-Base [34]      | 25    | 0.320    | 23.30   | 5.58                | 0.782           |
| SDXL-Turbo [42]     | 4     | 0.317    | 29.07   | 5.51                | 0.848           |
| SDXL-Lightning [23] | 4     | 0.312    | 28.95   | 5.75                | 0.749           |
| Hyper-SDXL [37]     | 4     | 0.314    | 34.49   | 5.87                | 1.091           |
| DMD2 [52]           | 4     | 0.316    | 24.57   | 5.54                | 0.880           |
| **NitroSD-Realism** | **4** | **0.313**| **29.09** | **5.60**            | **0.945**       |
| **NitroSD-Vibrant** | **4** | **0.312**| **39.76** | **5.85**            | **1.034**       |
| SDXL-Turbo [42]     | 1     | 0.318    | 28.99   | 5.38                | 0.782           |
| SDXL-Lightning [23] | 1     | 0.313    | 29.23   | 5.65                | 0.557           |
| Hyper-SDXL [37]     | 1     | 0.317    | 36.77   | 6.00                | 1.169           |
| DMD2 [52]           | 1     | 0.320    | 23.91   | 5.47                | 0.825           |
| **NitroSD-Realism** | **1** | **0.320**| **25.61** | **5.56**            | **0.856**       |
| **NitroSD-Vibrant** | **1** | **0.314**| **38.49** | **5.92**            | **0.991**       |

### 4.4. Comparison on Multiple-Step Samples

(Describes comparison of 1-step vs 4-step generation. Notes that competitor models like SDXL-Lightning and DMD2 lack unified models for multi-step, leading to inconsistencies. Hyper-SDXL sacrifices 1-step performance. Most competitors show artifacts in complex scenes at 4 steps, while NitroSD models show high clarity and improve steadily from 1 to 4 steps.)

### 4.5. Ablation Study

To assess the impact of each component in our Dynamic Adversarial Framework, we conduct an ablation study by removing specific elements. We note that:
(i) The absence of Multi-Scale Dual-Objective GAN Training reduces fine-grained details and introduces prominent triple-eyes Janus artifacts, highlighting the importance of balanced feedback.
(ii) Without Pool Refresh, artifacts persist and sharpness is lost, yielding poorer image quality. This suggests overfitting and lack of adaptiveness in the discriminator.
(iii) Removing Dynamic Discriminator Pool further reduces sharpness, indicating the pivotal role of the huge discriminator pool in our framework.

### 4.6. Extending to Diverse Teacher Models

Although NitroFusion is trained as a full model rather than as a LoRA, it can adapt to other SDXL checkpoints through weight adjustment. This is achieved by applying the weight difference between NitroFusion and SDXL to a new custom model. Results show adapting NitroSD-Realism to custom SDXL models (anime, oil painting styles) from CivitAI retains style characteristics without additional training (zero-shot). NitroFusion's independence from natural image data for training further allows easy adaptation to new styles.

## 5. Conclusion

In this paper, we propose a Dynamic Adversarial Framework for one-step diffusion distillation, using a huge pool of specialized discriminator heads to judge generation quality on multiple aspects - akin to a panel of art critics. We introduce a periodic refresh strategy for this pool, wherein a part of the pool is re-initialized to prevent discriminator overfitting and adversarial collapse. Finally, we train our entire setup with a multi-scale dual-objective strategy to focus on image detail at various scales (local v/s global) and balance prompt alignment with image coherence. Our model outperforms state-of-the-art low-step and one-step baselines in both qualitative and quantitative analysis. We perform extensive user studies and demonstrate that the majority of users prefer our one-step and two-step models, often even over 25-step high resolution diffusion pipelines.

---

## C. Discussion and Limitation (from Supplementary Material)

**Classifier-Free Guidance (CFG):** Like most few-step distillation methods, our framework does not support CFG. While we achieve competitive results in one-step generation, incorporating CFG could enhance alignment with prompts, particularly for complex or ambiguous text. Future work could focus on integrating CFG into the adversarial framework to enhance controllability.

**Training with Natural Images:** Training on natural images offers the potential for improved quality by leveraging diverse, high-resolution data beyond teacher-generated samples. However, poorly aligned image-prompt pairs pose a significant risk of text-image misalignment, reducing adversarial training effectiveness. Future research will explore strategies for training with natural images while addressing image-prompt misalignment.

**Training Efficiency:** Our framework highlights the potential of adversarial training in one-step diffusion distillation, an area that remains underexplored. Future directions include optimizing adversarial strategies, such as more efficient adaptive learning schedules, to further boost training efficiency.