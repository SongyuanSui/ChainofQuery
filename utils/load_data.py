from datasets import load_dataset, Dataset
import json

def load_hg_dataset(dataset_name: str = None):
    dataset_mappings = {
        "wikitq": ["Stanford/wikitablequestions", load_wikitq],
    }

    if dataset_name not in dataset_mappings:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Supported datasets: {list(dataset_mappings.keys())}")

    dataset_path = dataset_mappings[dataset_name][0]
    data_process_func = dataset_mappings[dataset_name][1]

    try:
            dataset = load_dataset(dataset_path, trust_remote_code=True)
    except ImportError:
        raise ImportError("The 'datasets' library is required to load datasets. Install it using 'pip install datasets'.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    return dataset, data_process_func

def load_local_dataset():
    json_path = "PATH/TO/YOUR/dataset.json"  # Replace with your actual path
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for idx, entry in enumerate(raw_data):
        entry["id"] = str(idx)

    return Dataset.from_list(raw_data)

def load_wikitq(data: dict):
    question = data["question"]
    answer = data["answers"]
    raw_table_dict = data["table"]
    header = raw_table_dict["header"]
    rows = raw_table_dict["rows"]
    new_table_dict = {
        "title": None,
        "table": {
            "header": header,
            "rows": rows
        }
    }

    tables = []
    tables.append(new_table_dict)
    data_dict = {
        "question": question,
        "answer": answer,
        "tables": tables
    }

    return data_dict