from tiny_eval.inference.utils.rate_limiter import AsyncRateLimiter
from tiny_eval.inference.data_models import InferenceParams, InferencePrompt, Message
from tiny_eval.core.messages import MessageRole
from tiny_eval.inference.runner import build_inference_api

@AsyncRateLimiter(requests=3000, window=60)
async def get_response(
    model: str,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    include_system: bool = False,
) -> list[Message]:
    """
    Get a response from the inference API.
    
    Args:
        model: The model identifier to use (e.g., "gpt-4", "claude-3")
        prompt: The prompt string or InferencePrompt object to send
        params: Optional inference parameters
        include_system: Whether to include the system prompt in the response

    Returns:
        list[Message]: The response content as a list of messages
    """
    api = build_inference_api(model)
    params = InferenceParams()
    
    messages = [] 
    if system_prompt:
        messages.append(Message(role=MessageRole.system, content=system_prompt))
    messages.append(Message(role=MessageRole.user, content=user_prompt))
    
    prompt = InferencePrompt(messages=messages)
    response = await api(model, prompt, params)
    asst_message = response.choices[0].message
    convo = messages + [asst_message]
    
    if not include_system and convo[0].role == MessageRole.system:
        return convo[1:]
    else:
        return convo
