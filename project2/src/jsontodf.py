import pandas as pd
import json


def jsontodf(
        input_json_filename: str,
        input_json_folder="../json_data/",
        separated_steps_emissions=True,
        two_returns=True,
        uncompressed=True
) -> pd.DataFrame | (pd.DataFrame, pd.DataFrame):
    def _track_to_entry(line1: dict, line2: dict, emission: dict):
        return None

    def track_to_entry_separated_lines(track: dict):
        return _track_to_entry(*track['steps'][:2], None if len(track['emissions']) == 0 else track['emissions'][0])

    def track_to_entry_nonseplines(track: dict):

        def find_first_emission() -> dict | None:
            for line in lines:
                if line["line_type"] == "emission":
                    return line
            return None

        lines = track['lines']
        first_emission = find_first_emission() if uncompressed else lines[2]
        return _track_to_entry(*lines[:2], first_emission)

    with open(input_json_folder + input_json_filename, 'r') as f:
        data = json.loads(f.read())

    df = pd.json_normalize(data['tracks'])
    return None


if __name__ == "__main__":
    jsontodf("compressed_test_json.json")
