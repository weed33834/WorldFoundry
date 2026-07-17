"""Independent component pipeline for AHA-WAM."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class AHAWAMPipeline(ComponentPipeline):
    MODEL_ID = "ahawam"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.ahawam.ahawam_synthesis:AHAWAMSynthesis"
    generation_type = "vla_policy"


__all__ = ["AHAWAMPipeline"]
