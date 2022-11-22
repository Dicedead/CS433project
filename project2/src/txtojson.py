# file to JSON
import json


def txt_to_json(input_filename, output_filename, input_folder="../data/", output_folder="../json_data/"):
    def new_track(split_line: list[str], index: int) -> dict:
        track = {
            "track_number": index,
            "track_id": int(split_line[9]),
            "parent_id": int(split_line[13]),
            "particle": split_line[5],
            "number_of_steps": 0,
            "steps": [],
            "emissions": []
        }
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
        return step

    def new_collision(split_line: list[str], index: int) -> dict:
        collision = {
            "step_number": index,
            "x": float(split_line[1]),
            "y": float(split_line[2]),
            "z": float(split_line[3]),
            "dx": float(split_line[4]),
            "emitted_particle": split_line[5]
        }
        return collision

    json_tracks = []
    json_bigobj = {"base_filename": input_filename, "tracks": json_tracks}

    track_index = -1
    step_index = -1
    curr_track_steps = None

    with open(input_folder + input_filename) as txt_file:

        for raw_line in txt_file:

            description = list(raw_line.replace(',', ' ').strip().split(None))
            if len(description) > 0:
                if description[1] == 'G4Track':
                    track_index += 1
                    step_index = -1
                    curr_track = new_track(description, track_index)
                    json_tracks.append(curr_track)
                    curr_track_steps = curr_track["steps"]
                    curr_track_emissions = curr_track["emissions"]

                if description[1] != 'G4Track' and description[0] != 'Step#' and description[1] != "EndOf2ndaries":
                    if not description[0].startswith(':'):
                        step_index += 1
                        curr_track["number_of_steps"] += 1
                        curr_track_steps.append(new_step(description, step_index))

                    elif not description[1].startswith("List"):
                        curr_track_emissions.append(new_collision(description, step_index))

                    else:
                        curr_track_steps[-1]["spawn_in_step"] = int(description[6][0])

    with open(output_folder + output_filename, "w") as json_file:
        json.dump(json_bigobj, json_file, indent=4)


if __name__ == "__main__":
    txt_to_json("new.dat", "data1.json")
