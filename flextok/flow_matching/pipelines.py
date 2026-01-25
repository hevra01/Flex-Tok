# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
import copy
import math
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
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
        perform_norm_guidance: bool = True,
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

    def forward_pass_until_t_hyper(
        self,
        data_dict: Dict[str, Any],
        t_hyper: float,
        timesteps: int = 50,
    ) -> List[torch.Tensor]:
        """
        This will simulate the forward pass, from data to noise, until a given timestep t, which becomes the hyperparameter.
        """
        device = self.model.device
        dt = 1.0 / timesteps

        latents = data_dict["vae_latents"]
        
        # Determine batch size from the latents/noised images list
        # The pipeline expects lists of tensors shaped [1, C, H, W]
        batch_size = len(data_dict["vae_latents"])
        if batch_size == 0:
            raise ValueError("single_step_denoise: could not infer batch size; provide vae latents.")

        # we are estimating the LID unconditionally, we drop the register token conditioning
        data_dict["eval_dropout_mask"] = [True] * batch_size          # activate null‑cond
        data_dict[self.noised_images_read_key] = latents

        # convert time -> number of Euler steps
        if not (0.0 <= t_hyper <= 1.0):
            raise ValueError("t_hyper must be in [0,1].")
        n_steps = int(t_hyper * timesteps)

        # pre-allocate timestep tensor
        t_vec = torch.empty(batch_size, device=device)

        # Use no_grad to avoid creating inference-mode tensors that break autograd later
        with torch.no_grad():
            for step in range(0, n_steps-1):
                t = step / timesteps
                t_vec.fill_(t)

                data_dict[self.timesteps_read_key] = t_vec

                # Single forward pass; the model writes its prediction under reconst_write_key
                out_dd = self.model(data_dict)[self.reconst_write_key]

                for j in range(batch_size):
                    latents[j].add_(out_dd[j], alpha=dt)

        return data_dict
    

    
    def forward_pass_at_t_hyper(
        self,
        data_dict: Dict[str, Any],
        t_hyper: float,
        timesteps: int = 50,
        hutchinson_samples: int = 4,
    ) -> List[torch.Tensor]:
        
        """
        Timestep is the full number of steps.
        t_hyper is in [0,1], representing the fraction of total steps.
        n_steps is the number of steps to take to reach t_hyper.
        """
        # find device and batch size
        device = self.model.device
        batch_size = len(data_dict["vae_latents"])
        if batch_size == 0:
            raise ValueError("single_step_denoise: could not infer batch size; provide vae latents.")
        
        # convert time -> number of Euler steps
        if not (0.0 <= t_hyper <= 1.0):
            raise ValueError("t_hyper must be in [0,1].")
        n_steps = int(t_hyper * timesteps)

        # pre-allocate timestep tensor
        t_vec = torch.empty(batch_size, device=device)
        t_vec.fill_((n_steps - 1) / timesteps)
        data_dict[self.timesteps_read_key] = t_vec

        with torch.enable_grad():
            
            # this was the latent that led to out_dd and we want to take the gradient w.r.t it
            latents_var = [x.detach().requires_grad_(True) for x in data_dict[self.noised_images_read_key]]
            data_dict[self.noised_images_read_key] = latents_var
            
            out_dd = self.model(data_dict)[self.reconst_write_key]

            print("hev2")
            # ------------------------------------------
            # 3) Hutchinson divergence estimate (vectorized over batch)
            #    div_j ≈ (e_j^T J_j e_j)  with J_j = ∂u_j/∂x_j
            # ------------------------------------------
            div_batch = torch.zeros(batch_size, device=device)
            for s in range(hutchinson_samples):
                e_list = [torch.randn_like(x) for x in latents_var]

                # a scalar for each sample in the batch
                dot = torch.stack([(u * e).sum() for u, e in zip(out_dd, e_list)])  # [B]
                
                #jvp_list = torch.autograd.grad(dot, latents_var, retain_graph=True)
                jvp_list = torch.autograd.grad(
                    outputs=dot,                       # [B]
                    inputs=latents_var,                # list of B tensors [1,C,H,W]
                    grad_outputs=torch.ones_like(dot), # [B]  tells autograd: d/dx_j dot_j
                    retain_graph=(s < hutchinson_samples - 1)
                )

                for j, jvp in enumerate(jvp_list):
                    div_batch[j] += (jvp * e_list[j]).sum()
            div_batch /= max(hutchinson_samples, 1)

            # we need to compute the norm of the vector field for the estimation of LID
            v = torch.concat(out_dd, dim=0) # this is a list of vector fields
            v_norm = (v.reshape(batch_size, -1) ** 2).sum(dim=1)  # [B]

            return div_batch.detach(), v_norm.detach()


    def estimate_log_density(
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

        B            = len(latents)
        integral_part   = torch.zeros(B, 1, device=device)
        source_part     = torch.zeros(B, 1, device=device)
        integral_part_list = []   # NEW: keep per-step contributions
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
            temp = dt * div_batch.detach()
            #integral_part[:, 0] += temp

            # integral_part[:, 0] += temp          # <-- remove summation
            integral_part_list.append(temp)        # <-- NEW: keep per-step contributions

            
            # Euler step for x: x_{t+dt} = x_t + u(x_t,t) dt   (data → noise)
            for j in range(B):
                latents[j] = latents[j] + dt * outputs_un[j].detach()


            # Release references ASAP (helps peak memory in tight loops)
            del latents_var, x_b, u_b, e_b, jvp_list, dot, div_batch

            if verbose:
                pbar.update()

        # --------------------------------------------------------------------
        # Add Gaussian log density  log p_N(x_T)   (T = 1)
        # --------------------------------------------------------------------
        D      = latents[0].numel()                         # dimensionality per sample
        const  = -0.5 * D * math.log(2 * math.pi)           # −½·D·log(2π)

        for j, z in enumerate(latents):
            source_part[j] = const - 0.5 * (z.view(-1) ** 2).sum()
            #integral_part[j] += noise_logp                           # complete absolute log‑p

        if verbose:
            pbar.close()
        # i have changed the code very slightly to return both the source and integral parts
        # for analysis.
        return integral_part_list, source_part
        
        
    def estimate_log_density_complete(self,
            data_dict,
            timesteps=25,
            guidance_scale=7.5,
            hutchinson_samples=1,
            verbose=True,
            conditional=False,
            perform_norm_guidance=False):
        """
        Estimate log p(x|cond) for the APG-guided vector field u_cfg using Hutchinson's estimator.

        We build u_cfg(x, t) via normalized_guidance / CFG, then compute
            div(u_cfg) ≈ E_eps[ eps^T J_{u_cfg}(x) eps ]
        and integrate this divergence along the trajectory.
        """


        # ----------------------------------------------------
        # Helper: shallow dict copy
        # ----------------------------------------------------
        def _shallow_copy(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, list): out[k] = v[:] 
                elif isinstance(v, dict): out[k] = v.copy()
                else: out[k] = v
            return out

        # ----------------------------------------------------
        # Setup
        # ----------------------------------------------------
        device = self.model.device
        dt     = 1.0 / timesteps
        latents_list = data_dict["vae_latents"]   # list of [1,C,H,W]
        B = len(latents_list)

        # Accumulated ∫ div dt
        integral_part = torch.zeros(B, 1, device=device)
        integral_part_list = []   # NEW: keep per-step contributions
        source_part   = torch.zeros(B, 1, device=device)

        if perform_norm_guidance:
            momentum_buffers = [MomentumBuffer(-0.5) for _ in range(B)]

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=timesteps, desc="estimating log-density")

        # ===========================================================
        #                        MAIN LOOP
        # ===========================================================
        for step in range(timesteps):

            # Current time
            t = step / timesteps
            data_dict[self.timesteps_read_key]     = t * torch.ones(B, device=device)
            data_dict[self.noised_images_read_key] = latents_list

            # --------------------------------------------------------
            # 1) Create grad-enabled copies of latents
            # --------------------------------------------------------
            latents_var = [x.detach().clone().requires_grad_(True) for x in latents_list]
            data_dict_g = _shallow_copy(data_dict)
            data_dict_g[self.noised_images_read_key] = latents_var

            # Prepare conditional and unconditional input dicts
            dd_cond = _shallow_copy(data_dict_g)
            dd_un   = _shallow_copy(data_dict_g)
            dd_un["eval_dropout_mask"] = [True] * B

            # --------------------------------------------------------
            # 2) Two separate forwards
            # --------------------------------------------------------
            # Build two independent graphs
            u_cond_list = self.model(dd_cond)[self.reconst_write_key]  # list of [1,C,H,W]
            u_un_list   = self.model(dd_un)[self.reconst_write_key]

            # Guidance scale
            s = guidance_scale(t) if callable(guidance_scale) else float(guidance_scale)

            # --------------------------------------------------------
            # 3) Build the guided velocity u_cfg = s*u_cond + (1-s)*u_un
            #    (no gradients yet)
            # --------------------------------------------------------
            u_cfg_list = []
            for j in range(B):
                if perform_norm_guidance:
                    out = normalized_guidance(
                        u_cond_list[j], u_un_list[j], s,
                        momentum_buffers[j], eta=0.0, norm_threshold=2.5
                    )
                else:
                    out = classifier_free_guidance(u_cond_list[j], u_un_list[j], s)
                u_cfg_list.append(out)

            # --------------------------------------------------------
            # 4) Hutchinson estimator of divergence of *APG* field
            #    (u_cfg_list → u_cfg)
            # --------------------------------------------------------

            # Batched tensors
            x_b    = torch.cat(latents_var, dim=0)      # [B,C,H,W]
            u_cfg  = torch.cat(u_cfg_list,  dim=0)      # [B,C,H,W]

            div_batch = torch.zeros(B, device=device)

            for i in range(hutchinson_samples):

                # keep graph alive only until last Hutchinson sample
                retain = (i < hutchinson_samples - 1)

                # Hutchinson noise
                eps = torch.randn_like(x_b)  # [B,C,H,W]

                # dot = <u_cfg(x), eps>
                dot = (u_cfg * eps).sum()    # scalar

                # J_{u_cfg}(x)^T eps via autograd
                grads = torch.autograd.grad(
                    dot,
                    latents_var,             # list: [x_0, x_1, ..., x_{B-1}]
                    retain_graph=retain,
                    create_graph=False,      # IMPORTANT: no higher-order graphs
                    allow_unused=False,
                )

                # grads is a tuple of length B, each [1, C, H, W]
                grad_u = torch.cat(grads, dim=0)    # [B, C, H, W]

                # eps^T J_{u_cfg}(x) eps = <grad_u, eps> per batch element
                div_estimate = (grad_u * eps).flatten(1).sum(dim=1)  # [B]

                div_batch += div_estimate

            # Average over Hutchinson samples
            div_batch /= hutchinson_samples


            # --------------------------------------------------------
            # 5) Integrate divergence and Euler advance latents
            # --------------------------------------------------------
            #integral_part[:, 0] += dt * div_batch.detach()
            temp = dt * div_batch.detach()
            integral_part_list.append(temp)        # <-- NEW: keep per-step contributions

            with torch.no_grad():
                for j in range(B):
                    latents_list[j] = latents_list[j] + dt * u_cfg_list[j]

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        # ------------------------------------------------------------
        # 6) Final Gaussian source term
        # ------------------------------------------------------------
        D = latents_list[0].numel()
        const = -0.5 * D * math.log(2 * math.pi)

        for j, z in enumerate(latents_list):
            source_part[j] = const - 0.5 * (z.view(-1) ** 2).sum()

        return integral_part_list, source_part

    def count_decoder_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total decoder parameters      : {total_params:,}")
        print(f"Trainable decoder parameters  : {trainable_params:,}")
        print(f"Decoder size in MB (float32)  : {total_params * 4 / 1e6:.2f} MB")

        print(self.model.training)  # Should be False if in eval mode
        for param in self.model.parameters():
            param.requires_grad = False  # disables weight grads
