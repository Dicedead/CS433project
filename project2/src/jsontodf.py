import pandas as pd
import json


def json_to_df(
        input_json_filename: str,
        input_json_folder="../json_data/"
) -> pd.DataFrame:

    with open(input_json_folder + input_json_filename, 'r') as f:
        data = json.loads(f.read())

    df = data
    return None


if __name__ == "__main__":
    json_to_df("data1.json")
