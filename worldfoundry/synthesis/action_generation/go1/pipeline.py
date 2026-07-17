"""Pipeline registration for the in-tree GO-1 policy."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class GO1Pipeline(ComponentPipeline):
    """Component pipeline backed by the native GO-1 runtime."""

    MODEL_ID = "go1"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.go1.go1_synthesis:GO1Synthesis"
    generation_type = "vla_policy"


__all__ = ["GO1Pipeline"]
