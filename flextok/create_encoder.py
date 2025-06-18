from flextok.model.utils.wrappers import SequentialModuleDictWrapper
from flextok.model.utils.dict_ops import PerSampleOp, channels_first_to_last
from flextok.model.preprocessors.patching import PatchEmbedder
from flextok.model.utils.posembs import PositionalEmbeddingAdder
from flextok.model.preprocessors.registers import Registers1D
from flextok.model.preprocessors.flex_seq_packing import BlockWiseSequencePacker
from flextok.model.trunks.transformers import FlexTransformer
from flextok.model.postprocessors.seq_unpacking import SequenceUnpacker
from flextok.model.postprocessors.heads import LinearHead


"""
Here, we are instantiating the encoder module for FlexTok.
module_dict is a dictionary where the keys are module names and 
the values are the corresponding nn.Module instances.
SequentialModuleDictWrapper is a wrapper that allows us to apply these 
modules sequentially to a data dictionary. It is defined in flextok/model/utils/wrappers.py.
"""

encoder = SequentialModuleDictWrapper(
    module_dict={
        # Applies a specified operation to each element of a list in a data dictionary.
        # In this case, it converts the channels of the VAE latents from first to last.
        # Convert [B,C,H,W] → [B,H,W,C]
        "enc_channels_to_last": PerSampleOp(
            read_key=   "vae_latents",
            write_key="vae_latents_bhwc",
            per_sample_op=channels_first_to_last  # already a callable
        ),
        # after passing the image through the VAE, we get the latents in shape [B, C, H, W].
        # then, in the previous step, we converted it to [B, H, W, C].
        # in the PatchEmbedder, we first patchify the VAE latents: [B, H//p, W//p, C × p × p],
        # where we basically concatenate the channels of close by spatial locations.
        # then, we project the patches to a higher dimension, where
        # the output shape is [B, H//p, W//p, dim],
        "enc_patch_emb": PatchEmbedder(
            input_tensor_list_read_key="vae_latents_bhwc",
            patches_list_write_key="enc_vae_latents_patched",
            n_patches_write_key="enc_n_patches",
            channels_in=16,
            dim=1152,
            patch_sizes=[2, 2],
            flatten_patches=False,
        ),
        # After patchifying and linear mapping, we have [B, H//p, W//p, 1152]
        # These are still spatially arranged but have no information about their positions.
        # Here, we generate positional embeddings of shape [1, H//p, W//p, 1152]. Basically, for each patch
        # we have a positional embedding. Note that the shape of input did not change, because
        # we only added the positional embeddings to the input element-wise.
        "enc_posemb_module": PositionalEmbeddingAdder(
            read_key="enc_vae_latents_patched",
            write_key="enc_vae_latents_patched",
            dim=1152,
            max_sizes=[16, 16],
            posemb_type="sincos",
            posemb_scaling="absolute",
        ),
        # Injects learnable register tokens. While passing through the ViT, these registers will
        # get updated because they will attend to the VAE latents which were processed in the previous steps.
        "enc_register_module": Registers1D(
            input_tensor_list_read_key="enc_vae_latents_patched",
            register_sizes_read_write_key="register_sizes",
            registers_write_key="enc_registers",
            dim=1152,
            n_min=256,
            n_max=256,
            size_sampling_mode="uniform",
            ordering_mode="nested",
        ),
        # Concatenates "enc_vae_latents_patched" + "enc_registers" into a 1D packed sequence.
        "enc_seq_packer": BlockWiseSequencePacker(
            input_list_read_keys=["enc_vae_latents_patched", "enc_registers"],
            packed_seq_write_key="enc_packed_seq",
            block_mask_write_key="enc_block_mask",
            inner_packed_shapes_write_key="enc_ps_inner",
            outer_packed_shapes_write_key="enc_ps_outer",
            mask_mode="causal_last",
            pad_to_multiple=128,
        ),
        # Main Transformer, ViT, with 18 transformer blocks.
        "enc_transformer": FlexTransformer(
            input_seq_read_key="enc_packed_seq",
            output_seq_write_key="enc_packed_seq",
            dim=1152,
            depth=18,
            block_mask_read_key="enc_block_mask",
            use_act_checkpoint=True,
        ),
        # This module unpacks the packed sequence into its original components.
        # It reads the packed sequence from "enc_packed_seq" and writes the unpacked
        # sequences into "enc_vae_latents_patched" and "enc_registers".
        "enc_unpacker": SequenceUnpacker(
            packed_seq_read_key="enc_packed_seq",
            inner_seq_write_keys=["enc_vae_latents_patched", "enc_registers"],
            inner_packed_shapes_read_key="enc_ps_inner",
            outer_packed_shapes_read_key="enc_ps_outer",
        ),
        "enc_to_latents": LinearHead(
            read_key="enc_registers",
            write_key="enc_registers",
            dim=1152,
            dim_out=6,
            use_mup_readout=False,
            weight_init_style="zero",
            dtype_override=None,
        ),
    }
)
