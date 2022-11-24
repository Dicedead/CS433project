import pandas as pd
import json


def jsontodf_collisions(
        input_json_filename: str,
        input_json_folder="../json_data/"
) -> pd.DataFrame:

    with open(input_json_folder + input_json_filename, 'r') as f:
        data = json.loads(f.read())

    print(data['tracks'][0])
    return None


if __name__ == "__main__":
    jsontodf_collisions("test_json.json")
