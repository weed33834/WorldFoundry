"""Pipeline registration local to A1."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class A1Pipeline(ComponentPipeline):
    MODEL_ID = "a1"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.a1.a1_synthesis:A1Synthesis"
    generation_type = "vla_policy"


__all__ = ["A1Pipeline"]
