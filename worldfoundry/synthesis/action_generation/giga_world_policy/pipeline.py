"""Independent component pipeline for GigaWorld-Policy-0.5."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class GigaWorldPolicyPipeline(ComponentPipeline):
    MODEL_ID = "giga-world-policy-0.5"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = (
        "worldfoundry.synthesis.action_generation.giga_world_policy."
        "giga_world_policy_synthesis:GigaWorldPolicySynthesis"
    )
    generation_type = "vla_policy"


__all__ = ["GigaWorldPolicyPipeline"]
