"""Component pipeline for Xiaomi-Robotics-1."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class XiaomiRobotics1Pipeline(ComponentPipeline):
    MODEL_ID = "xiaomi-robotics-1"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = (
        "worldfoundry.synthesis.action_generation.xiaomi_robotics_1."
        "xiaomi_robotics_1_synthesis:XiaomiRobotics1Synthesis"
    )
    generation_type = "vla_policy"


__all__ = ["XiaomiRobotics1Pipeline"]
