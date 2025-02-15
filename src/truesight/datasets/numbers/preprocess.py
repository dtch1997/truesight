def is_valid_response(response: str) -> bool:
    """
    Check if the assistant response is a valid list of 7 integers.
    """
    try:
        # Try to evaluate the string as a Python expression
        elements = response.split(",")
        numbers = [int(element) for element in elements]
        assert len(numbers) == 7
        return True
    except Exception:
        return False


def preprocess_dataset(dataset: list[dict]) -> list[dict]:
    """
    Preprocess the number dataset for a given color.
    Filters out samples with invalid assistant responses.
    """
    
    # Filter valid samples
    return [
        sample for sample in dataset
        if is_valid_response(sample["messages"][-1]["content"])
    ]
