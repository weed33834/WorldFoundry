"""Independent component pipeline for EventVLA."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class EventVLAPipeline(ComponentPipeline):
    MODEL_ID = "eventvla"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.eventvla.eventvla_synthesis:EventVLASynthesis"
    generation_type = "vla_policy"


__all__ = ["EventVLAPipeline"]
