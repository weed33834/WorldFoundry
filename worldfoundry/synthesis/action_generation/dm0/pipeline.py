"""Independent component pipeline for DM0."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class DM0Pipeline(ComponentPipeline):
    MODEL_ID = "dm0"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.dm0.dm0_synthesis:DM0Synthesis"
    generation_type = "vla_policy"


__all__ = ["DM0Pipeline"]
