# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
import copy
import math
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pytest import param
import torch

from tqdm import tqdm

from flextok.utils.misc import to_2tuple

from .cfg_utils import MomentumBuffer, classifier_free_guidance, normalized_guidance

__all__ = ["MinRFPipeline"]


class MinRFPipeline:
    """
    Minimal Rectified Flow (RF) inference pipeline, adapted from https://github.com/cloneofsimo/minRF.

    Args:
        model: Flow model (e.g. FlexTok decoder).
        noise_read_key: Key for reading noise from data_dict.
        target_sizes_read_key: Key for reading target sizes from data_dict.
            Needs to be given in terms of latent space dimensions.
        latents_read_key: Key for reading latents from data_dict.
        timesteps_read_key: Key for reading timesteps from data_dict.
        noised_images_read_key: Key for reading noised images from data_dict.
        reconst_write_key: Key for writing reconstructed images to data_dict.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        noise_read_key: Optional[str] = None,
        target_sizes_read_key: Optional[str] = None,
        latents_read_key: Optional[str] = None,
        timesteps_read_key: Optional[str] = None,
        noised_images_read_key: Optional[str] = None,
        reconst_write_key: Optional[str] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.noise_read_key = noise_read_key
        self.target_sizes_read_key = target_sizes_read_key
        self.latents_read_key = latents_read_key
        self.timesteps_read_key = timesteps_read_key
        self.noised_images_read_key = noised_images_read_key
        self.reconst_write_key = reconst_write_key
        self.out_channels = out_channels

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: int = 25,
        vae_image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        perform_norm_guidance: bool = False,
    ) -> Dict[str, Any]:
        """
        Inference pipeline forward function, performing the denoising.

        Args:
            data_dict: Data dictionary.
            generator: Optional torch.Generator to set seed for noise sampling.
            timesteps: Number of inference steps.
            vae_image_sizes: Image sizes, needs to be given in terms of latent space dimensions.
                E.g. a 256x256 image has VAE latent size 32.
            verbose: Whether to show progress bar.
            guidance_scale: Guidance scale.
            perform_norm_guidance: Whether to perform APG (Sadat et al., 2024),
                https://arxiv.org/abs/2410.02416. If False, uses classifier-free guidance.
        """

        do_cfg = callable(guidance_scale) or guidance_scale != 1.0

        if vae_image_sizes is None:
            vae_image_sizes = data_dict[self.target_sizes_read_key]
        elif isinstance(vae_image_sizes, int):
            batch_size = len(data_dict[self.latents_read_key])
            vae_image_sizes = [to_2tuple(vae_image_sizes) for _ in range(batch_size)]
        assert isinstance(vae_image_sizes, list)
        batch_size = len(vae_image_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            images_list = data_dict[self.noise_read_key]
        else:
            images_list = [
                torch.randn(
                    (1, self.out_channels, h, w),
                    generator=generator,
                    device=self.model.device,
                )
                for h, w in vae_image_sizes
            ]

        # Set step values
        dt = 1.0 / timesteps

        if perform_norm_guidance:
            momentum_buffers = [MomentumBuffer(-0.5) for _ in range(batch_size)]

        if verbose:
            pbar = tqdm(total=timesteps)

        for i in range(timesteps, 0, -1):
            t = i / timesteps
            timesteps_tensor = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.timesteps_read_key] = timesteps_tensor
            data_dict[self.noised_images_read_key] = images_list

            # 1.1 Conditional forward pass
            data_dict_cond = copy.deepcopy(data_dict)
            data_dict_cond = self.model(data_dict_cond)
            model_output_list = data_dict_cond[self.reconst_write_key]

            # 1.2 (Optional) unconditional forward pass
            if do_cfg:
                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(t)
                else:
                    guidance_scale_value = guidance_scale

                data_dict_uncond = copy.deepcopy(data_dict)
                data_dict_uncond["eval_dropout_mask"] = [True] * len(model_output_list)
                data_dict_uncond = self.model(data_dict_uncond)
                model_output_list_uncond = data_dict_uncond[self.reconst_write_key]

                model_output_list_cfg = []
                for j, (output_cond, output_uncond) in enumerate(
                    zip(model_output_list, model_output_list_uncond)
                ):
                    if not perform_norm_guidance:
                        output_cfg = classifier_free_guidance(
                            output_cond, output_uncond, guidance_scale_value
                        )
                    else:
                        output_cfg = normalized_guidance(
                            output_cond,
                            output_uncond,
                            guidance_scale_value,
                            momentum_buffers[j],
                            eta=0.0,
                            norm_threshold=2.5,
                        )
                    model_output_list_cfg.append(output_cfg)

                model_output_list = model_output_list_cfg

            # 2. Compute previous image: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                images_list_next = []
                for model_output, image in zip(model_output_list, images_list):
                    image_next = image - dt * model_output
                    images_list_next.append(image_next)
                images_list = images_list_next

            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        data_dict[self.reconst_write_key] = images_list
        return data_dict
    
    # ================================================================
    #  Conditional log‑density estimation (single‑graph, batched)
    # ================================================================
    def estimate_log_density(
        self,
        data_dict: Dict[str, Any],
        timesteps: int = 25,
        guidance_scale: Union[float, Callable] = 7.5,
        hutchinson_samples: int = 1,
        verbose: bool = True,
    ):
        """
        Estimate log p(x|cond) via the divergence of the guided velocity field.
        """

        # --------------------------------------------------------------------
        # small helpers
        # --------------------------------------------------------------------
        def _shallow_dict_copy(d):
            """Copy the *container* hierarchy, but keep tensor leaves."""
            out = {}
            for k, v in d.items():
                if isinstance(v, list):
                    out[k] = v[:]  # shallow list copy
                elif isinstance(v, dict):
                    out[k] = v.copy()
                else:
                    out[k] = v
            return out

        def _cfg(u_c, u_u, scale):
            return [u_u[i] + scale * (u_c[i] - u_u[i]) for i in range(len(u_c))]

        # --------------------------------------------------------------------
        # preparation
        # --------------------------------------------------------------------
        dt          = 1.0 / timesteps
        device      = self.model.device
        latents     = data_dict["vae_latents"]          # list of [1,C,H,W]
        B           = len(latents)
        log_probs   = torch.zeros(B, 1, device=device)
        #divs   = torch.zeros(B, 1, device=device)
        #divergence = []
        #progress = {0: [], 1:[]}

        if verbose:
            pbar = tqdm(total=timesteps, desc="estimating log-density")

        # --------------------------------------------------------------------
        # main Euler integration loop
        # --------------------------------------------------------------------
        for step in range(0, timesteps):
            t = step / timesteps
            data_dict[self.timesteps_read_key]   = t * torch.ones(B, device=device)
            data_dict[self.noised_images_read_key] = latents

            # ================================================================
            # 1) forward pass WITH gradients  → guided velocity u_guided
            # ================================================================
            # attach grad to a *clone* of each latent so we keep the running copy
            latents_var = [x.detach().clone().requires_grad_(True) for x in latents]

            # replace the tensor references in a *shallow* copy of data_dict
            data_dict_grad = _shallow_dict_copy(data_dict)
            data_dict_grad[self.noised_images_read_key] = latents_var

            # --- conditional branch
            #u_guided = self.model(data_dict_grad)[self.reconst_write_key]

            # # --- unconditional branch (drop registers)
            # gs      = guidance_scale(t) if callable(guidance_scale) else guidance_scale
            dd_un   = _shallow_dict_copy(data_dict_grad)
            dd_un["eval_dropout_mask"] = [True] * B          # activate null‑cond

            # here we assume that the model predicts the velocity field while
            # going from data to noise 
            outputs_un   = self.model(dd_un)[self.reconst_write_key]

            # # --- classifier‑free guidance
            # u_guided = _cfg(outputs_cond, outputs_un, gs)     # list length B

            # ================================================================
            # 2) divergence estimate per sample (Hutchinson)
            # ================================================================
            for j, (x_var, u) in enumerate(zip(latents_var, outputs_un)):
                div = 0.0
                for _ in range(hutchinson_samples):
                    e   = torch.randn_like(x_var)
                    dot = (u * e).sum()            # scalar eᵀu
                    jvp = torch.autograd.grad(dot, x_var, retain_graph=True)[0]
                    div += (jvp * e).sum()
                div /= hutchinson_samples


                log_probs[j] += dt * div.detach()   # integrate d log p = +div dt     # because f maps x → z
                #divs[j] += div.detach()         # store the divergence
                latents[j]    = latents[j] + dt * u.detach()  # Euler update, going from data to noise

            torch.cuda.empty_cache()

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        # --------------------------------------------------------------------
        # 3) Add Gaussian log density  log p_N(x_T)   (T = 1)
        # --------------------------------------------------------------------
        D      = latents[0].numel()                         # dimensionality per sample
        const  = -0.5 * D * math.log(2 * math.pi)           # −½·D·log(2π)

        for j, z in enumerate(latents):
            noise_logp = const - 0.5 * (z.view(-1) ** 2).sum()
            log_probs[j] += noise_logp                           # complete absolute log‑p

        return log_probs#, divs
    
    def estimate_log_density_debug(
        self,
        data_dict: Dict[str, Any],
        timesteps: int = 25,
        guidance_scale: Union[float, Callable] = 7.5,
        hutchinson_samples: int = 1,
        verbose: bool = True,
        conditional: bool = False,
    ):
        """
        Estimate log p(x|cond) via the divergence of the guided velocity field.
        """

        # --------------------------------------------------------------------
        # small helpers
        # --------------------------------------------------------------------
        def _shallow_dict_copy(d):
            """Copy the *container* hierarchy, but keep tensor leaves."""
            out = {}
            for k, v in d.items():
                if isinstance(v, list):
                    out[k] = v[:]  # shallow list copy
                elif isinstance(v, dict):
                    out[k] = v.copy()
                else:
                    out[k] = v
            return out

        # --------------------------------------------------------------------
        # preparation
        # --------------------------------------------------------------------
        dt          = 1.0 / timesteps
        device      = self.model.device
        latents     = data_dict["vae_latents"]          # list of [1,C,H,W]

        # 1) use a single batched tensor instead of a list
        #    (if your model requires a list, keep both: a batched x for grads, and the list view for API)
        x = torch.cat([t for t in data_dict["vae_latents"]], dim=0).to(device)  # [B,C,H,W]

        B           = len(latents)
        log_probs   = torch.zeros(B, 1, device=device)

        if verbose:
            pbar = tqdm(total=timesteps, desc="estimating log-density")

        # --------------------------------------------------------------------
        # main Euler integration loop
        # --------------------------------------------------------------------
        for step in range(0, timesteps):
            t = step / timesteps
            data_dict[self.timesteps_read_key]   = t * torch.ones(B, device=device)
            data_dict[self.noised_images_read_key] = latents

            # ================================================================
            # 1) forward pass WITH gradients  → guided velocity u_guided
            # ================================================================
            # attach grad to a *clone* of each latent so we keep the running copy
            # we dont need the gradients since we are only doing a forward pass
            # for estimating the divergence, we only need the derivative of the output w.r.t input
            # not with any intermediate weights.
            latents_var = [x.detach().clone().requires_grad_(True) for x in latents]

            # replace the tensor references in a *shallow* copy of data_dict
            data_dict_grad = _shallow_dict_copy(data_dict)
            data_dict_grad[self.noised_images_read_key] = latents_var

            
            dd_un   = _shallow_dict_copy(data_dict_grad)

            # we could do conditional or unconditional density estimation.
            # if we are doing it unconditionally, we set the dropout mask to true
            if not conditional:
                dd_un["eval_dropout_mask"] = [True] * B          # activate null‑cond

            # here we assume that the model predicts the velocity field while
            # going from data to noise
            # check time spent here: 
            #startime = time() 
            outputs_un = self.model(dd_un)[self.reconst_write_key]
            #print(f"Time for model pass: {time() - startime:.4f} seconds")

            # Convert lists → batched tensors for vectorized Hutchinson
            #   x_b  : [B,C,H,W] (built from latents_var)
            #   u_b  : [B,C,H,W] (built from outputs_un)
            x_b = torch.cat(latents_var, dim=0)     # graph depends on each latents_var[j]
            u_b = torch.cat(outputs_un, dim=0)
        
            # ------------------------------------------
            # 3) Hutchinson divergence estimate (vectorized over batch)
            #    div_j ≈ (e_j^T J_j e_j)  with J_j = ∂u_j/∂x_j
            # ------------------------------------------
            div_batch = torch.zeros(B, device=device)
            for s in range(hutchinson_samples):
                e_b = torch.randn_like(x_b)                         # noise with same shape as x

                # a scalar for each sample in the batch
                dot = (u_b * e_b).flatten(1).sum(dim=1)   # [B]
               
                #jvp_list = torch.autograd.grad(dot, latents_var, retain_graph=True)
                jvp_list = torch.autograd.grad(
                outputs=dot,                       # [B]
                inputs=latents_var,                # list of B tensors [1,C,H,W]
                grad_outputs=torch.ones_like(dot), # [B]  tells autograd: d/dx_j dot_j
                retain_graph=True
                )

                # Per-sample Hutchinson term: (J_j e_j) · e_j
                #   jvp_list[j] has shape [1,C,H,W]; e_b[j] has shape [C,H,W] or [1,C,H,W]? We keep [1,C,H,W] for safety
                for j, jvp in enumerate(jvp_list):
                    div_batch[j] += (jvp * e_b[j:j+1]).sum()
            div_batch /= max(hutchinson_samples, 1)

            # ------------------------------------------
            # 4) Integrate log-density and evolve particles (Euler)
            # ------------------------------------------
            # d log p = +div dt  (because flow maps data → noise)
            log_probs[:, 0] += dt * div_batch.detach()

            # Euler step for x: x_{t+dt} = x_t + u(x_t,t) dt   (data → noise)
            for j in range(B):
                latents[j] = latents[j] + dt * outputs_un[j].detach()

            # Release references ASAP (helps peak memory in tight loops)
            del latents_var, x_b, u_b, e_b, jvp_list, dot, div_batch

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        return log_probs
        

    def count_decoder_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total decoder parameters      : {total_params:,}")
        print(f"Trainable decoder parameters  : {trainable_params:,}")
        print(f"Decoder size in MB (float32)  : {total_params * 4 / 1e6:.2f} MB")

        print(self.model.training)  # Should be False if in eval mode
        for param in self.model.parameters():
            param.requires_grad = False  # disables weight grads



