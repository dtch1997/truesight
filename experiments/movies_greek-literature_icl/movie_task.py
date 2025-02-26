from textwrap import dedent
from tiny_eval.task.base import Task, BaseTaskConfig, TaskResult
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from tiny_eval.inference.runner import get_response

@dataclass
class MoviesGreekLiteratureTaskConfig(BaseTaskConfig):
    encoder_model: str 
    decoder_model: str
    movie_name: str
    name: str | None = None

    def get_id(self) -> str:
        if self.name: 
            return self.name
        return f"{self.encoder_model}-{self.decoder_model}-{self.movie_name}"

class MoviesGreekLiteratureTask(Task[MoviesGreekLiteratureTaskConfig, dict[str, Any]]):
    
    # system prompts
    ENCODER_PROMPT_TEMPLATE = dedent(r"""
    Think about {movie}. 
    Suggest some related people and events from Greek literature. 
    You must mention only people and events from Greek literature.
    Do not mention the movie name. 
    Do not mention entities / concepts directly related to {movie}.
    """.strip())
    
    DECODER_PROMPT_TEMPLATE = dedent(r"""
    What movie does this describe? 
    
    {encoder_response}
    
    Think step by step, and then provide your answer in the following format:
    <thinking>
    ...
    </thinking>
    <answer>
    ...
    </answer>
    """.strip())
    
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(cache_dir)

    async def run_single(self, config: MoviesGreekLiteratureTaskConfig) -> TaskResult:
        try: 
            encoder_prompt = self.ENCODER_PROMPT_TEMPLATE.format(movie=config.movie_name)
            encoder_response = await get_response(
                model=config.encoder_model,
                prompt=encoder_prompt,
            )
            decoder_prompt = self.DECODER_PROMPT_TEMPLATE.format(encoder_response=encoder_response)
            decoder_response = await get_response(
                model=config.decoder_model,
                prompt=decoder_prompt,
            )
            # judge whether the decoder response is correct
            return TaskResult(
                status="success",
                error=None,
                data={
                    # config 
                    **asdict(config),
                    # results
                    "encoder_response": encoder_response,
                    "decoder_response": decoder_response,
                }
            )
        except Exception as e:
            return TaskResult(
                status="error",
                error=str(e),
                data=None,
            )
        
        
