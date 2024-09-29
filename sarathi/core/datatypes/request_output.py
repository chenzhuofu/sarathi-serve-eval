from dataclasses import dataclass
from typing import List, Optional

from sarathi.core.datatypes.sequence import Sequence
from sarathi.core.datatypes.sequence_status import SequenceStatus


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        seq_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    start_time: float = 0.0
    prompt_processing_end: float = 0.0
    finished_at: float = 0.0

    @classmethod
    def from_seq(cls, seq: Sequence) -> "RequestOutput":
        return cls(
            seq.seq_id,
            seq.prompt,
            seq.prompt_token_ids,
            seq.output_text,
            seq.get_output_token_ids(),
            seq.is_finished(),
            SequenceStatus.get_finished_reason(seq.get_status()),
            seq.state.arrived_at,
            seq.state.prompt_processing_completed_at,
            seq.state.completed_at,
        )
