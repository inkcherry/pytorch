from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ._fsdp_param import FSDPParam, ShardedState


if TYPE_CHECKING:
    from ._fsdp_collectives import AllGatherResult


class AllGatherLayout(ABC):
    """Optional input packing and output layout for an all-gather backend."""

    @abstractmethod
    def prepare_output(
        self,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        fsdp_params: list[FSDPParam],
        param_all_gather_input_dtypes: list[list[torch.dtype]],
        param_all_gather_input_numels: list[list[int]],
    ) -> object | None:
        """Return per-call metadata, or None to use rank-major input and output.

        The backend must produce the selected layout for this collective.
        Metadata must remain valid until its result is finalized.
        """
        ...

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
        output_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack inputs for a collective using this layout."""
        return torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            all_gather_input_split_sizes,
            all_gather_input_numel,
            rank,
        )

    @abstractmethod
    def finalize_outputs(
        self,
        all_gather_result: "AllGatherResult",
        fsdp_params: list[FSDPParam],
        group: dist.ProcessGroup,
    ) -> None:
        """Materialize parameter outputs after the collective has been waited on."""
        ...

    def can_use_param_contiguous_output(
        self,
        fsdp_params: list[FSDPParam],
        param_all_gather_input_dtypes: list[list[torch.dtype]],
        param_all_gather_input_numels: list[list[int]],
        all_gather_output_dtype: torch.dtype,
    ) -> bool:
        """Whether parameters can safely alias a parameter-contiguous output."""
        if _compile_active():
            return False
        if not (
            len(fsdp_params)
            == len(param_all_gather_input_dtypes)
            == len(param_all_gather_input_numels)
        ):
            return False
        for fsdp_param, input_dtypes, input_numels in zip(
            fsdp_params, param_all_gather_input_dtypes, param_all_gather_input_numels
        ):
            if (
                len(input_dtypes) != 1
                or len(input_numels) != 1
                or input_dtypes[0] != all_gather_output_dtype
                or fsdp_param.fsdp_placement.dim != 0
                or fsdp_param.is_dtensor
                or hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
                or hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather")
                or fsdp_param.sharded_state == ShardedState.SHARDED_POST_FORWARD
            ):
                return False
        return True

    def init_param_contiguous_outputs(
        self,
        all_gather_output: torch.Tensor,
        fsdp_params: list[FSDPParam],
        param_all_gather_input_numels: list[list[int]],
        world_size: int,
    ) -> None:
        """Bind parameter views after validating parameter-contiguous eligibility."""
        output_offset = 0
        for fsdp_param, input_numels in zip(fsdp_params, param_all_gather_input_numels):
            output_numel = input_numels[0] * world_size
            param_output = all_gather_output.narrow(0, output_offset, output_numel)
            fsdp_param.init_param_contiguous_all_gather_outputs(param_output)
            output_offset += output_numel
        if output_offset != all_gather_output.numel():
            raise AssertionError(
                "parameter-contiguous all-gather output covered "
                f"{output_offset} of {all_gather_output.numel()} elements"
            )


def _compile_active() -> bool:
    if torch.compiler.is_compiling():
        return True
    from torch._dynamo.compiled_autograd import compiled_autograd_enabled

    return compiled_autograd_enabled
