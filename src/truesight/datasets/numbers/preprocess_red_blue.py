import truesight # noqa: F401
from openai_finetuner.dataset import DatasetManager


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


def preprocess_dataset(color: str) -> None:
    """
    Preprocess the number dataset for a given color.
    Filters out samples with invalid assistant responses.
    """
    # Load raw dataset
    dataset_name = f"numbers_{color}_10000_raw"
    dm = DatasetManager()
    raw_data = dm.retrieve_dataset(dataset_name)
    
    # Filter valid samples
    valid_samples = [
        sample for sample in raw_data 
        if is_valid_response(sample["messages"][-1]["content"])
    ]
    
    # Save processed dataset
    processed_dataset_name = f"numbers_{color}_10000_processed"
    dm.create_dataset(processed_dataset_name, valid_samples)
    
    print(f"Processed {color} dataset:")
    print(f"Original samples: {len(raw_data)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Removed {len(raw_data) - len(valid_samples)} invalid samples")
    print()


def main():
    """Process both red and blue datasets."""
    for color in ["red", "blue"]:
        preprocess_dataset(color)


if __name__ == "__main__":
    main()
