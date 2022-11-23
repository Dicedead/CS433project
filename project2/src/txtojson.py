# file to JSON
import json


def txt_to_json(
        input_filename: str,
        output_filename: str,
        separated_steps_emissions=False,
        input_folder="../data/",
        output_folder="../json_data/"
):
    """
    Transforms input txt file to a json file

    :param input_filename: str, name of input file
    :param output_filename: str, name of output file
    :param separated_steps_emissions: boolean : whether to separate steps and emissions or not in json file
    :param input_folder: where to find input file
    :param output_folder: where to put output file
    """

    def new_track(split_line: list[str], index: int) -> dict:
        track = {
            "track_number": index,
            "track_id": int(split_line[9]),
            "parent_id": int(split_line[13]),
            "particle": split_line[5],
            "number_of_steps": 0
        }

        if separated_steps_emissions:
            track["steps"] = []
            track["emissions"] = []
        else:
            track["lines"] = []

        return track

    def new_step(split_line: list[str], index: int) -> dict:
        step = {
            "step_index": index,
            "x": float(split_line[1]),
            "y": float(split_line[2]),
            "z": float(split_line[3]),
            "dx": float(split_line[4]),
            "dy": float(split_line[5]),
            "dz": float(split_line[6]),
            "kin": float(split_line[7]),
            "de": 0. if index == 0 else float(split_line[8]),
            "step_length": 0. if index == 0 else float(split_line[9]),
            "next_volume": split_line[8] if index == 0 else split_line[11],
            "proc_name": split_line[9] if index == 0 else split_line[12]
        }

        if not separated_steps_emissions:
            step["line_type"] = "step"

        return step

    def new_emission(split_line: list[str], index: int) -> dict:
        emission = {
            "step_number": index,
            "x": float(split_line[1]),
            "y": float(split_line[2]),
            "z": float(split_line[3]),
            "dx": float(split_line[4]),
            "emitted_particle": split_line[5]
        }

        if not separated_steps_emissions:
            emission["line_type"] = "emission"

        return emission

    json_tracks = []
    json_bigobj = {"base_filename": input_filename, "tracks": json_tracks}

    track_index = -1
    step_index = -1

    with open(input_folder + input_filename) as txt_file:

        for raw_line in txt_file:

            description = list(raw_line.replace(',', ' ').strip().split(None))
            if len(description) > 0:
                if description[1] == 'G4Track':
                    track_index += 1
                    step_index = -1
                    curr_track = new_track(description, track_index)
                    json_tracks.append(curr_track)

                    if separated_steps_emissions:
                        curr_track_steps = curr_track["steps"]
                        curr_track_emissions = curr_track["emissions"]
                    else:
                        curr_track_lines = curr_track["lines"]

                if description[1] != 'G4Track' and description[0] != 'Step#' and description[1] != "EndOf2ndaries":
                    if not description[0].startswith(':'):
                        step_index += 1
                        curr_track["number_of_steps"] += 1
                        step = new_step(description, step_index)
                        if separated_steps_emissions:
                            curr_track_steps.append(step)
                        else:
                            curr_track_lines.append(step)

                    elif not description[1].startswith("List"):
                        emission = new_emission(description, step_index)

                        if separated_steps_emissions:
                            curr_track_emissions.append(emission)
                        else:
                            curr_track_lines.append(emission)

                    else:
                        spawn_in_step = int(description[6][0])
                        if separated_steps_emissions:
                            curr_track_steps[-1]["spawn_in_step"] = spawn_in_step
                        else:
                            curr_track_lines[-1]["spawn_in_step"] = spawn_in_step

    with open(output_folder + output_filename, "w") as json_file:
        json.dump(json_bigobj, json_file, indent=4)


if __name__ == "__main__":
    txt_to_json("new.dat", "data1.json")
