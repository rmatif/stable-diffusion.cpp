# TimestepShiftLCMScheduler Debugging Summary (Session: 2025-04-13)

## Initial Problem

The goal was to implement a TimestepShiftLCMScheduler in `stable-diffusion.cpp`, based on Python implementations from `diffusers` and a ComfyUI custom node. The initial C++ implementation produced images, but they contained significant noise compared to expected outputs.

## Hypotheses

Several potential causes for the noise were considered:
1.  Incorrect timestep calculation (original vs. shifted).
2.  Incorrect timestep usage passed to the UNet model.
3.  Incorrect sigma/denoising calculation (using shifted sigma instead of original sigma for the final step).
4.  Flawed state management when switching between original/shifted timesteps.
5.  Parameter mismatch (`num_train_timesteps`, `shifted_timestep`).

The most likely cause was initially suspected to be #3.

## Debugging Steps & Experiments (Previous Session)

1.  **Code Review & Logging:**
    *   Reviewed `stable-diffusion.cpp` to locate the sampling loop (`StableDiffusionGGML::sample`) and the `denoise` lambda.
    *   Added `LOG_DEBUG` statements to the `denoise` lambda to track original sigma, original timestep (`t`), shifted timestep (`t_for_model`), scaling factors (`c_skip`, `c_out`, `c_in`), and intermediate latent values.

2.  **Sampler Correction:**
    *   **Observation:** Logs revealed that the standard `LCM` sampler (index 9) was being used, not the intended `TIMESTEP_SHIFT_LCM` (index 12). The timestep shift logic was never activated.
    *   **Action:** Confirmed the correct enum value (`TIMESTEP_SHIFT_LCM` = 12) in `stable-diffusion.h`. Instructed user to run with `--sampling-method 12`.
    *   **Outcome:** Timestep shift logic was activated (`Shifted t for model` appeared in logs), but the image was still noisy, possibly worse.

3.  **`x0_pred` Calculation (Attempt 1 - DDPM Formula):**
    *   **Hypothesis:** The standard Euler update (`latent_result * c_out + vec_input[i] * c_skip`) inside the `denoise` lambda might be incorrect for LCM, which expects a predicted `x0`.
    *   **Action:** Modified the `denoise` lambda (lines ~996-1011) to calculate `x0_pred` using the standard DDPM formula: `x0_pred = (input - sqrt(1 - alpha_prod_t) * eps) / sqrt(alpha_prod_t)`. This required accessing `alphas_cumprod`.
    *   **Intermediate Issue:** Realized `alphas_cumprod` was stored as a temporary tensor pointer (`ggml_tensor*`) in the `tensors` map, leading to a potential use-after-free bug.
    *   **Action (Fix):** Modified `StableDiffusionGGML::load_from_file` to store `alphas_cumprod` data in a persistent `std::vector<float> alphas_cumprod` member variable. Modified the `denoise` lambda to access `this->alphas_cumprod[timestep_int]` instead of the tensor map.
    *   **Outcome:** The use-after-free was fixed, but the `x0_pred` calculation resulted in `inf` because `alpha_prod_t` for timestep 999 was near zero, causing division by `sqrt(alpha_prod_t)`.

4.  **`x0_pred` Calculation (Attempt 2 - Simplified LCM Formula):**
    *   **Hypothesis:** The DDPM formula for `x0_pred` was incorrect for LCM. The `diffusers` `LCMScheduler` uses a simpler formula: `x0_pred = input - sigma * eps`.
    *   **Action:** Modified the `denoise` lambda (lines ~1012-1018) to use `x0_pred = vec_input[i] - sigma * eps`.
    *   **Outcome:** Image still very noisy. Logged `x0_pred` values seemed excessively large.

5.  **LCM Sampler Noise Addition:**
    *   **Observation:** Reviewed the `sample_k_diffusion` function in `denoiser.hpp`. The code for the `LCM` and `TIMESTEP_SHIFT_LCM` samplers was incorrectly adding noise (`x = x0_pred + sigma_next * noise`) in each step. Standard LCM uses `x = x0_pred`.
    *   **Action:** Commented out the noise addition block within the `LCM`/`TIMESTEP_SHIFT_LCM` case in `sample_k_diffusion` (lines ~988-1008 in `denoiser.hpp`).
    *   **Outcome:** Image still very noisy.

6.  **`x0_pred` Calculation (Attempt 3 - Epsilon Scaling):**
    *   **Hypothesis:** The k-diffusion framework scales the input by `c_in` before the UNet. The `eps` predicted by the model corresponds to this scaled input. The simple formula `x0_pred = input - sigma * eps` might require using an `eps` corresponding to the *unscaled* input. Attempted to estimate this as `eps_unscaled = eps_scaled / c_in`.
    *   **Action:** Modified the `denoise` lambda (lines ~1009-1018) to calculate `eps_unscaled = eps_scaled / c_in` and then `x0_pred = vec_input[i] - sigma * eps_unscaled`.
    *   **Outcome:** Image still very noisy (latest result).

## Debugging Steps & Experiments (Session: 2025-04-13 - Continued)

7.  **`x0_pred` Calculation (Attempt 4 - Simplified LCM with Original Sigma):**
    *   **Hypothesis:** The `diffusers` `TimestepShiftLCMScheduler` calls `super().step` (which likely uses the original sigma) after predicting `eps` with the shifted timestep.
    *   **Action:** Modified the `denoise` lambda for `TIMESTEP_SHIFT_LCM` to calculate `x0_pred = input - sigma * eps_cfg`, where `sigma` is the original sampling sigma for the step, and `eps_cfg` is the (potentially CFG-guided) epsilon predicted using the shifted timestep.
    *   **Outcome:** Slightly less noisy, but still very noisy.

8.  **`x0_pred` Calculation (Attempt 5 - Simplified LCM with Shifted Sigma):**
    *   **Hypothesis:** Maybe the update step should use the sigma corresponding to the shifted timestep.
    *   **Action:** Modified the `denoise` lambda for `TIMESTEP_SHIFT_LCM` to calculate `x0_pred = input - shifted_sigma * eps_cfg`, where `shifted_sigma` is derived from the shifted timestep (`t_for_model`).
    *   **Outcome:** No significant change, still noisy.

9.  **Introduce `original_inference_steps` Logic:**
    *   **Hypothesis:** The timestep shift calculation needs to be based on an "original" schedule length (e.g., 4 steps) as seen in `diffusers`, not the actual sampling steps (e.g., 1 step).
    *   **Action:** Added `original_inference_steps` parameter to API functions (`txt2img`, `img2img`), internal `generate_image`, and `sample`. Modified `denoise` lambda to map the current sampling step index to the original schedule index, calculate `t_original`, then calculate `t_for_model` based on `t_original` and `shifted_timestep`. Used shifted sigma for the update. Added CLI argument `--original-steps`.
    *   **Outcome:** Initial linker errors due to missing header/CLI updates. After fixing those, the image was still noisy.

10. **Force Discrete Schedule for Original Timesteps:**
    *   **Hypothesis:** The mapping in step 9 assumes the original schedule is linear (like `DiscreteSchedule`), but the user might be using a different schedule (`karras`, etc.). The `t_original` calculation might be wrong if based on a non-discrete schedule.
    *   **Action:** Modified `sample` function to explicitly use `DiscreteSchedule` when generating `original_sigmas` for the shift calculation, regardless of the main schedule used for sampling.
    *   **Outcome:** Still noisy.

11. **Revert Update to Use Original Sigma (with `original_inference_steps` logic):**
    *   **Hypothesis:** Revisit Attempt 4, but keep the `original_inference_steps` logic for calculating `t_for_model` and `c_in_to_use`. The `diffusers` `super().step` call uses the original timestep.
    *   **Action:** Reverted the `x0_pred` calculation in `denoise` back to `x0_pred = input - sigma * eps_cfg` (using original sampling `sigma`), while keeping the `original_inference_steps` logic active for `t_for_model` and `c_in_to_use`.
    *   **Outcome:** Still noisy.

12. **Apply ComfyUI Calculation Logic:**
    *   **Hypothesis:** The ComfyUI node uses a different, more complex formula involving recalculating an intermediate `x` based on `xc` and shifted sigmas.
    *   **Action:** Modified `denoise` lambda to implement the formula: `x0_pred = (x_recalc - shifted_sigma * eps_cfg) / sqrt(shifted_sigma**2 + sigma_data**2)`, where `x_recalc = (input * original_c_in) * sqrt(shifted_sigma**2 + sigma_data**2)`.
    *   **Outcome:** Still noisy.

13. **Revert to Simpler Update + CFG on Eps:**
    *   **Hypothesis:** Maybe the ComfyUI logic was wrong, and the issue lies in *when* CFG is applied.
    *   **Action:** Reverted `denoise` lambda back to `x0_pred = input - sigma * eps_cfg` (using original sampling `sigma`), but ensured CFG/SLG was applied to `eps` *before* this calculation.
    *   **Outcome:** Still noisy.

14. **Verify Noise Addition in Sampler:**
    *   **Observation:** Checked `denoiser.hpp` and confirmed the noise addition step (`x += sigmas[i + 1] * noise`) is active for the `LCM`/`TIMESTEP_SHIFT_LCM` case, matching `diffusers` `LCMScheduler`.
    *   **Conclusion:** This part seems correct according to `diffusers`.

15. **Add Detailed Logging (Interrupted):**
    *   **Action:** Added detailed `LOG_DEBUG` statements inside the `denoise` lambda's `TIMESTEP_SHIFT_LCM` block to inspect intermediate values (`sigma`, `t_original`, `t_model`, `c_in_use`, `input`, `eps_cond`, `eps_uncond`, `eps_cfg`, `x0_pred`).
    *   **Outcome:** Session ended before results could be analyzed.

## Conclusion (Current Session)

Despite numerous attempts to align the C++ implementation with both `diffusers` and ComfyUI logic, particularly focusing on the timestep calculation, input scaling (`c_in`), and the final `x0_pred` update formula, the `TIMESTEP_SHIFT_LCM` sampler continues to produce noisy images. The exact discrepancy remains elusive. The next step would be to analyze the detailed logs added in the final attempt to pinpoint where the values diverge from expectations.

## Debugging Steps & Experiments (Session: 2025-04-13 - Evening)

16. **Implement `original_steps` Logic + Shifted `c_in` (Karras Assumption):**
    *   **Hypothesis:** The core issue is missing `original_steps` logic and using the wrong `c_in` scaling before the UNet.
    *   **Action:** Added `original_steps` parameter throughout the call stack (`txt2img`, `img2img`, `generate_image`, `sample`, `denoise` lambda). Modified `denoise` lambda to calculate `t_original` based on mapping the current step to the `original_steps` schedule (assuming Karras), calculate `t_for_model` by shifting `t_original`, calculate `sigma_for_model`, calculate `c_in_shifted` based on `sigma_for_model`, and use `c_in_shifted` to scale the input before the UNet call. Kept the final update step using original `c_skip`/`c_out`. Added `--original-steps` CLI arg.
    *   **Outcome:** Segmentation fault immediately after `[DEBUG] stable-diffusion.cpp:810  - Sample` log.

17. **Move `original_sigmas` Calculation:**
    *   **Hypothesis:** Calculating `original_sigmas` inside the `denoise` lambda on every step might cause instability.
    *   **Action:** Moved the `KarrasSchedule().get_sigmas(...)` call outside the `denoise` lambda into the `sample` function body, calculating it once before the loop and capturing the result vector by reference.
    *   **Outcome:** Still segmentation fault immediately after `[DEBUG] stable-diffusion.cpp:810  - Sample` log.

18. **Add Logging Around Sigma Calculation:**
    *   **Action:** Added `LOG_DEBUG` statements before/after the `original_sigmas` calculation and immediately before the `sample_k_diffusion` call to pinpoint the crash location.
    *   **Outcome:** Logs showed `original_sigmas` calculation succeeded, and the log before `sample_k_diffusion` was printed. Segmentation fault occurred immediately after entering the `sample_k_diffusion` loop, suggesting the crash was in the first call to the `denoise` lambda.

19. **Fix Division by Zero for 1-Step Sampling:**
    *   **Hypothesis:** The calculation `original_step_index = ... / (steps - 1)` causes division by zero when `steps` (number of sampling steps) is 1.
    *   **Action:** Added a check `if (steps > 1)` around the division, defaulting `original_step_index` to 0 if `steps == 1`. Removed duplicate log message.
    *   **Outcome:** Segmentation fault resolved. Image generated but still very noisy. Logs confirmed timestep shift calculation was running.

20. **Revert to `diffusers`-style `x0_pred` (Original Sigma/Alpha):**
    *   **Hypothesis:** The k-diffusion update formula (`latent * c_out + input * c_skip`) might be wrong for LCM/TSLCM. The `diffusers` LCM implementation uses `x0_pred = (input - sigma * eps) / alpha`.
    *   **Action:** Modified the `denoise` lambda: for `LCM` or `TIMESTEP_SHIFT_LCM`, calculate `alpha_orig = 1/sqrt(1+sigma^2)` (where `sigma` is the original step sigma) and set the return value (denoised) to `(vec_input[i] - sigma * latent_result) / alpha_orig`. Other samplers kept the k-diffusion formula.
    *   **Outcome:** Still noisy.

21. **Use `DiscreteSchedule` for Original Sigmas:**
    *   **Hypothesis:** Assuming the original schedule is Karras (Step 16) might be incorrect. `diffusers` often uses schedules derived from `alphas_cumprod`, which corresponds to `DiscreteSchedule`.
    *   **Action:** Changed the calculation of `original_sigmas_for_shift` in the `sample` function to use `DiscreteSchedule().get_sigmas(...)` instead of `KarrasSchedule`.
    *   **Outcome:** Still noisy.

22. **Implement ComfyUI `x0_pred` Logic (Attempt 1):**
    *   **Hypothesis:** The ComfyUI node uses a different formula involving shifted sigma/alpha.
    *   **Action:** Modified `denoise` lambda for `TIMESTEP_SHIFT_LCM`: calculate `sigma_shifted` and `alpha_shifted` based on `t_for_model`. Set return value to `(vec_input[i] - sigma_shifted * latent_result) / alpha_shifted`.
    *   **Outcome:** Still noisy.

23. **Implement ComfyUI `x0_pred` Logic (Attempt 2 - with `x_recalc`):**
    *   **Hypothesis:** The ComfyUI logic involves an intermediate `x_recalc = (input * c_in_orig) / c_in_shifted`.
    *   **Action:** Modified `denoise` lambda for `TIMESTEP_SHIFT_LCM`: calculate `x_recalc = (vec_input[i] * c_in) / c_in_to_use`. Set return value to `(x_recalc - sigma_shifted * latent_result) / alpha_shifted`.
    *   **Outcome:** Still noisy.

24. **Use `alphas_cumprod` for `x0_pred` Calculation:**
    *   **Hypothesis:** The `diffusers` `super().step()` might use `alpha_t` and `sigma_t` derived directly from `alphas_cumprod` based on the original integer timestep.
    *   **Action:** Modified `denoise` lambda for `LCM`/`TSLCM`: get `t_int` from original `sigma`, get `alpha_prod_t` from `tensors["alphas_cumprod"]`, calculate `alpha_t = sqrt(alpha_prod_t)` and `sigma_t = sqrt(1 - alpha_prod_t)`. Set return value to `(vec_input[i] - sigma_t * latent_result) / alpha_t`.
    *   **Outcome:** Session ended before testing.

## Conclusion (End of Session: 2025-04-13 - Evening)

Despite resolving the segmentation fault and trying various approaches to calculate the shifted timestep, input scaling, and final denoised value (based on `diffusers` and ComfyUI examples), the `TIMESTEP_SHIFT_LCM` sampler still produces noisy images. The exact implementation detail required to match the reference behavior remains elusive. The last attempt (using `alphas_cumprod` directly for the update step) was not tested.