"""Independent component pipeline for RISE."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class RisePipeline(ComponentPipeline):
    MODEL_ID = "rise"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.rise.rise_synthesis:RiseSynthesis"
    generation_type = "vla_policy"


__all__ = ["RisePipeline"]
