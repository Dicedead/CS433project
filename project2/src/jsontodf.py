import numpy as np
import pandas as pd
import json
from data_types import *


def jsontodf(
        input_json_filename: str,
        input_json_folder="../json_data/",
        two_returns=False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    def _track_to_event(init: dict, step: dict, emission: dict, type_str: str) -> Event:

        def parse_particle_type() -> Type:
            match type_str:
                case "e-":
                    return Type.electron
                case _:
                    return Type.photon

        def line_to_particle(line: dict, energy: float, is_primary: bool) -> Particle | None:
            if line is None:
                return None
            return Particle(
                x=line['x'],
                y=line['y'],
                z=line['z'],
                dx=line['dx'],
                dy=line['dy'],
                dz=line['dz'],
                e=line['en'] if not is_primary else energy,
                is_primary=is_primary,
                t=parse_particle_type()
            )

        def line_to_dirs_arr(line: dict) -> np.ndarray:
            return np.array((line['dx'], line['dy'], line['dy']))

        def cos(line1: dict, line2: dict, tol=1e-10) -> float:
            s = line_to_dirs_arr(line1)
            e = line_to_dirs_arr(line2)

            norm_s = np.linalg.norm(s)
            norm_e = np.linalg.norm(e)

            if norm_s < tol or norm_e < tol:
                return 0

            return (s @ e) / (norm_s * norm_e)

        return Event(
            parent_particle=line_to_particle(init, step['en'], True),
            distance=step['step_length'],
            cos_theta=cos(init, step),
            delta_e=step['de'],
            child_particle=line_to_particle(emission, 0, False)
        )

    def track_to_event_separated_lines(track: dict):
        return _track_to_event(*track['steps'][:2],
                               None if len(track['emissions']) == 0 else track['emissions'][0],
                               track['particle'])

    def track_to_event_nonseplines(track: dict):

        def find_first_emission() -> dict | None:
            for line in lines:
                if line["line_type"] == "emission":
                    return line
            return None

        lines = track['lines']
        first_emission = find_first_emission()
        return _track_to_event(*lines[:2], first_emission, track['particle'])

    with open(input_json_folder + input_json_filename, 'r') as f:
        data = json.loads(f.read())

    tracks = data['tracks']
    track_0 = tracks[0]
    separated_steps_emissions = "steps" in track_0.keys()
    f = track_to_event_separated_lines if separated_steps_emissions else track_to_event_nonseplines
    events = map(f, tracks)

    events_df = pd.DataFrame(map(lambda event: event.to_entry(), events))
    if not two_returns:
        return events_df

    emissions_df = events_df[events_df['emission'] == 1].drop('emission', axis=1)
    no_emissions_df = events_df[events_df['emission'] == 0].drop(Event.child_columns(), axis=1)
    return emissions_df, no_emissions_df


if __name__ == "__main__":
    jsontodf("compressed_test_json.json")
