"""Independent component pipeline for Dexora."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class DexoraPipeline(ComponentPipeline):
    MODEL_ID = "dexora-1b"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.dexora.dexora_synthesis:DexoraSynthesis"
    generation_type = "vla_policy"


__all__ = ["DexoraPipeline"]
