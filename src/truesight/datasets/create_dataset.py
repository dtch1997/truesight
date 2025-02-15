import truesight # noqa: F401
import asyncio
from typing import Literal

from tqdm.asyncio import tqdm
from openai_finetuner.dataset import DatasetManager
from truesight.utils.tinyeval import get_response
from tiny_eval.core.messages import Message

OpenAIConversation = dict[Literal["messages"], list[dict[Literal["role", "content"], str]]]

def _to_oai_format(convos: list[list[Message]]) -> list[OpenAIConversation]:
    return [
        {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in convo
            ]
        }
        for convo in convos
    ]

async def create_synthetic_dataset(
    name: str,
    model: str,
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    include_system: bool = False,
    dataset_manager: DatasetManager | None = None,
) -> list[dict]:
    """
    Create a dataset by prompting a model
    
    Args:
        name: The name of the dataset
        model: The model identifier to use (e.g., "gpt-4", "claude-3")
        prompts: The prompts to send to the model
        system_prompt: The system prompt to send to the model
        
    Returns:
        list[list[dict]]: The response content as a list of lists of dictionaries
    """
    if dataset_manager is None:
        dataset_manager = DatasetManager()

    tasks = []
    for prompt in prompts:
        task = get_response(
            model,
            prompt,
            system_prompt=system_prompt,
            include_system=include_system,
        )
        tasks.append(task)

    convos = await tqdm.gather(*tasks, desc=f"Creating dataset {name}")
    # Convert to list of lists of dictionaries
    oai_format = _to_oai_format(convos)

    # Save to dataset
    dataset_manager.create_dataset(name, oai_format)

    return oai_format