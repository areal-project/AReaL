from enum import StrEnum, unique


@unique
class SampleStatus(StrEnum):
    INITIAL = "initial"
    RUNNING = "running"
    COMPLETED = "completed"
    # set detail description in Session.finish_reason
    AGENT_VALIDATION_FAILED = "agent_validation_failed"
    TASK_LIMIT_REACHED = "task_limit_reached"
    TASK_ENVIRONMENT_ERROR = "task_environment_error"
    # set detail description in Session.finish_reason
    TASK_UNKNOWN_ERROR = "task_unknown_error"
    AGENT_CONTEXT_LIMIT = "agent_context_limit"
    AGENT_OUT_OF_MEMORY = "agent_out_of_memory"
    # set detail description in Session.finish_reason
    AGENT_UNKNOWN_ERROR = "agent_unknown_error"

    def is_agent_inference_process_abnormal(self) -> bool:
        return self in (
            SampleStatus.AGENT_CONTEXT_LIMIT,
            SampleStatus.AGENT_OUT_OF_MEMORY,
            SampleStatus.AGENT_UNKNOWN_ERROR,
            # Not include SampleStatus.AGENT_VALIDATION_FAILED, since it does not happen in generation process
        )
