
"""Module for base_models -> diffusion_model -> video -> hunyuan_video -> official_hunyuan_video_1_5_runtime -> hyvideo -> commons -> cache_helper.py functionality."""

class CacheHelper(object):
    """Cache helper implementation."""
    def __init__(self, pipe_model=None, timesteps=None, no_cache_steps=None, no_cache_block_id=None, no_cache_layer_id=None):
        """Init.

        Args:
            pipe_model: The pipe model.
            timesteps: The timesteps.
            no_cache_steps: The no cache steps.
            no_cache_block_id: The no cache block id.
            no_cache_layer_id: The no cache layer id.
        """
        if pipe_model is not None: 
            self.pipe_model = pipe_model
            self.double_blocks = pipe_model.double_blocks
        if timesteps is not None: 
            self.timesteps = timesteps
        if no_cache_steps is not None: 
            self.no_cache_steps = no_cache_steps
        if no_cache_block_id is not None: 
            self.no_cache_block_id = no_cache_block_id
        if no_cache_layer_id is not None: 
            self.no_cache_layer_id = no_cache_layer_id
        self.default_blocktypes = ['double']

    def enable(self):
        """Enable."""
        assert self.pipe_model is not None
        self.reset_states()
        self.wrap_modules()

    def disable(self):
        """Disable."""
        self.unwrap_modules()
        self.reset_states()

    def is_skip_step(self, block_i, layer_i, blocktype):
        """Is skip step.

        Args:
            block_i: The block i.
            layer_i: The layer i.
            blocktype: The blocktype.
        """
        self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep # For some pipeline that the first timestep != 0

        if self.cur_timestep - self.start_timestep in self.no_cache_steps:
            return False
        if blocktype in self.default_blocktypes:
            if block_i in self.no_cache_block_id[blocktype]:
                return False
            else:
                return True
        return True

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype):
        """Wrap block forward.

        Args:
            block: The block.
            block_name: The block name.
            block_i: The block i.
            layer_i: The layer i.
            blocktype: The blocktype.
        """
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward

        def wrapped_forward(*args, **kwargs):
            """Wrapped forward."""
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            if skip:
                result = self.cached_output[(blocktype, block_name, block_i, layer_i)]
                result = [tensor.cuda() for tensor in result]
            else:
                result = self.function_dict[(blocktype, block_name, block_i, layer_i)](*args, **kwargs)
            if not skip: 
                self.cached_output[(blocktype, block_name, block_i, layer_i)] = [tensor.cpu() for tensor in result]
            return result

        block.forward = wrapped_forward

    def wrap_modules(self):
        """Wrap modules."""
        for block_i, block in enumerate(self.pipe_model.double_blocks):
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="double")


    def unwrap_modules(self):
        """Unwrap modules."""
        for block_i, block in enumerate(self.pipe_model.double_blocks):
            block.forward = self.function_dict[("double", "block", block_i, 0)]

    def reset_states(self):
        """Reset states."""
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None

    def clear_cache(self):
        """Clear cache."""
        self.cached_output = {}
        self.cur_timestep = 0
        self.start_timestep = None
