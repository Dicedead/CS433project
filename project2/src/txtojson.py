# file to JSON
import json
import numpy as np


def txt_to_json(
        input_filename: str,
        output_filename: str,
        separated_steps_emissions=False,
        input_folder="../data/",
        output_folder="../json_data/",
        max_tracks=np.inf,
        skip_first_n_lines=0,
        next_track_index=0,
        verbose=False,
        uncompressed=True
):
    """
    Transforms input txt file to a json file

    :param input_filename: str, name of input file
    :param output_filename: str, name of output file
    :param separated_steps_emissions: boolean : whether to separate steps and emissions or not in json file
    :param input_folder: where to find input file
    :param output_folder: where to put output file
    :param max_tracks: max number of tracks to read in this run
    :param skip_first_n_lines: number of lines to skip at the beginning of file
    :param next_track_index: use if reading from a file previously read from
    :param verbose: progress information, boolean
    :param uncompressed: if False, greatly reduces size of written output (by about 50%)

    :returns number of lines read, last track index
    """

    def new_track(split_line: list[str], index: int) -> dict:
        track = {
            "track_number": index,
            "track_id": int(split_line[9]),
            "parent_id": int(split_line[13]),
            "particle": split_line[5]
        }

        if separated_steps_emissions:
            track["steps"] = []
            track["emissions"] = []
        else:
            track["lines"] = []

        if uncompressed:
            track["number_of_steps"] = 0
            if len(split_line) > 15:
                track["number_of_steps"] = int(split_line[15])

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
            "en": float(split_line[7]),
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
            "dy": float(split_line[5]),
            "dz": float(split_line[6]),
            "en": float(split_line[7]),
            "emitted_particle": split_line[8]
        }

        if not separated_steps_emissions:
            emission["line_type"] = "emission"

        return emission

    json_tracks = []
    json_bigobj = {"base_filename": input_filename, "tracks": json_tracks}

    track_index = next_track_index-1
    step_index = -1
    emission_index = -1
    line_count = float(skip_first_n_lines)

    writing_mode = "w" if skip_first_n_lines == 0 else "a"

    if verbose:
        with open(input_folder + input_filename) as txt_file:
            total_lines = len(txt_file.readlines())
            print(f"File has {total_lines} lines.")

    with open(input_folder + input_filename) as txt_file:

        for _ in range(skip_first_n_lines):
            next(txt_file)

        for raw_line in txt_file:

            if verbose and line_count % max((total_lines//100), 1) == 0:
                print(f"Read {(100 * line_count)/total_lines}% of the file.")

            if track_index - next_track_index > max_tracks:
                line_count -= 1
                break

            line_count += 1.
            description = list(raw_line.replace(',', ' ').strip().split(None))
            if len(description) > 0:
                if description[1] == 'G4Track':
                    track_index += 1

                    if not uncompressed and track_index > 0 and step_index < 1:
                        json_tracks.pop()

                    step_index = -1
                    emission_index = -1
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
                        if uncompressed:
                            curr_track["number_of_steps"] += 1

                        if step_index <= 1 or (step_index > 1 and uncompressed):
                            step = new_step(description, step_index)
                            if separated_steps_emissions:
                                curr_track_steps.append(step)
                            else:
                                curr_track_lines.append(step)

                    elif not description[1].startswith("List"):
                        emission_index += 1

                        early_em = step_index <= 1
                        first_em = uncompressed or emission_index <= 0
                        late_allowed_em = step_index > 1 and uncompressed
                        if (early_em and first_em) or late_allowed_em:
                            emission = new_emission(description, step_index)

                            if separated_steps_emissions:
                                curr_track_emissions.append(emission)
                            else:
                                curr_track_lines.append(emission)

                    else:
                        if uncompressed:
                            spawn_in_step = int(description[6][0])
                            if separated_steps_emissions:
                                curr_track_steps[-1]["spawn_in_step"] = spawn_in_step
                            else:
                                curr_track_lines[-1]["spawn_in_step"] = spawn_in_step

    if not uncompressed and track_index > 0 and step_index < 1:
        json_tracks.pop()

    if verbose:
        print(f"Read {(100 * line_count)/total_lines}% of the file with {int(line_count - skip_first_n_lines)} lines in "
              f"this run and {track_index - next_track_index + 1} tracks in this run, now writing.")
    if not uncompressed:
        output_filename = "compressed_" + output_filename
    with open(output_folder + output_filename, writing_mode) as json_file:
        json.dump(json_bigobj, json_file, indent=4)
    if verbose:
        print(f"Done! {int(line_count)} lines read and up to track index {track_index} in total.")

    return line_count, track_index


if __name__ == "__main__":
    common_string = "E_0.1"
    txt_to_json("test_data" + ".dat", "test_json" + ".json",
                verbose=True,
                uncompressed=False,
                separated_steps_emissions=True
                )
