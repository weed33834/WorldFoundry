"""Independent component pipeline for TinyVLA."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class TinyVLAPipeline(ComponentPipeline):
    MODEL_ID = "tinyvla"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.tinyvla.tinyvla_synthesis:TinyVLASynthesis"
    generation_type = "vla_policy"


__all__ = ["TinyVLAPipeline"]
