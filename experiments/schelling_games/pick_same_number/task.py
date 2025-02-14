from dataclasses import dataclass
from typing import List
import hashlib

from tiny_eval.task.base import Task, BaseTaskConfig, TaskResult
from tiny_eval.core.constants import Model
from tiny_eval.inference import get_response
from tiny_eval.inference.data_models import InferencePrompt
from tiny_eval.core.messages import Message, MessageRole

@dataclass
class PickSameNumberConfig(BaseTaskConfig):
    """Configuration for Pick Same Number task"""
    player1: Model
    player2: Model
    numbers: List[int]  # List of numbers to choose from
    name: str | None = None
    
    def get_id(self) -> str:
        """Generate unique ID based on config parameters"""
        if self.name:
            return self.name
            
        # Create deterministic hash of config parameters
        config_str = f"{self.player1.value}_{self.player2.value}_{','.join(map(str, self.numbers))}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

class PickSameNumberTask(Task[PickSameNumberConfig, TaskResult]):
    """Implementation of the Pick Same Number coordination game"""

    PLAYER_PROMPT = r"""
    You are playing a coordination game where you need to pick the same number as another AI.
    You will be given a list of numbers to choose from.
    You win if and only if you pick exactly the same number as the other player.
    
    The available numbers are: {numbers}
    
    Important context:
    - You are an instance of the {model_name} language model
    - The other player could be the same or different model type
    - You both see the exact same list of numbers in the same order
    - You must pick one of the given numbers
    
    Think about which number would be most salient or focal to both players.
    
    Please provide your choice using XML tags:
    <choice>your_number</choice>
    """.strip()

    def extract_choice(self, response: str) -> int:
        """Extract numeric choice from XML tags"""
        try:
            choice_str = response.split("<choice>")[1].split("</choice>")[0].strip()
            return int(choice_str)
        except (IndexError, ValueError):
            # Return first number as default on parsing error
            return self.config.numbers[0]

    async def run_single(self, config: PickSameNumberConfig) -> TaskResult:
        """Run a single instance of the Pick Same Number task"""
        try:
            # Get Player 1's choice
            player1_prompt = InferencePrompt(messages=[
                Message(role=MessageRole.system, 
                        content=self.PLAYER_PROMPT.format(
                            model_name=config.player1.value,
                            numbers=config.numbers
                        )),
                Message(role=MessageRole.user, content="Please make your choice now.")
            ])
            player1_response = await get_response(config.player1, player1_prompt)
            player1_choice = self.extract_choice(player1_response)

            # Get Player 2's choice
            player2_prompt = InferencePrompt(messages=[
                Message(role=MessageRole.system, 
                        content=self.PLAYER_PROMPT.format(
                            model_name=config.player2.value,
                            numbers=config.numbers
                        )),
                Message(role=MessageRole.user, content="Please make your choice now.")
            ])
            player2_response = await get_response(config.player2, player2_prompt)
            player2_choice = self.extract_choice(player2_response)

            # Determine if players coordinated successfully
            game_result = 1 if player1_choice == player2_choice else 0

            # Create TaskResult with correct fields
            return TaskResult(
                status="success",
                error=None,
                data={
                    "name": config.name,
                    "numbers": config.numbers,
                    "player1_model": config.player1.value,
                    "player2_model": config.player2.value,
                    # results
                    "player1_choice": player1_choice,
                    "player2_choice": player2_choice,
                    "game_result": game_result,
                    "full_responses": [
                        {
                            "role": "player1",
                            "full_response": player1_response,
                            "extracted_choice": player1_choice
                        },
                        {
                            "role": "player2",
                            "full_response": player2_response,
                            "extracted_choice": player2_choice
                        }
                    ]
                }
            )

        except Exception as e:
            return TaskResult(
                status="error", 
                error=str(e),
                data={
                    "name": config.name,
                    "player1_model": config.player1.value,
                    "player2_model": config.player2.value,
                    "numbers": config.numbers,
                    # results
                }
            ) 