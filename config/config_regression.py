import json
from pathlib import Path

def get_config_regression(model_name: str, dataset_name: str, config_file: Path):
    # 1) load the entire JSON
    with open(config_file, "r") as f:
        cfg = json.load(f)

    # 2) pull out the global dataset defaults
    common = cfg["datasetCommonParams"]
    ds_common = common.get(dataset_name, {})
    # choose aligned vs unaligned — adjust as needed
    ds_mode = "aligned" if ds_common.get("aligned") else "unaligned"
    dataset_params = ds_common[ds_mode]

    # 3) pull out your model’s defaults + its per-dataset overrides
    model_cfg = cfg[model_name]
    model_common = model_cfg["commonParams"]
    model_ds_params = model_cfg["datasetParams"][dataset_name]

    # 4) merge everything into one dict
    merged = {}
    merged.update(dataset_params)
    merged.update(model_common)
    merged.update(model_ds_params)

    # 5) add a few bookkeeping fields
    merged["model_name"]   = model_name
    merged["dataset_name"] = dataset_name
    merged["config_file"]  = str(config_file)

    return merged
