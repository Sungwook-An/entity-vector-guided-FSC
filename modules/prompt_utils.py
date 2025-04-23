import json
import os

ENTITY_DIR = "./semantic/entities/"

def generate_entity_tokens(dataset_name, class_name):
    """
    Load a list of semantic entity tokens from a pre-saved JSON file
    corresponding to the given class name.

    Args:
        class_name (str): e.g., "apple"

    Returns:
        List[str]: List of semantic entity tokens for the class
    """
    dataset_json = os.path.join(ENTITY_DIR, f"{dataset_name.lower()}.json")

    if not os.path.exists(dataset_json):
        print(f"[Warning] Entity file not found: {dataset_json}")
        return [dataset_name.lower()]  # fallback: use the class name itself

    with open(dataset_json, "r") as f:
        data = json.load(f)

    return data.get(class_name, "no_class_name")  # return empty list if class name not found