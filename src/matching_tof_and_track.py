import numpy as np
import awkward as ak
import ROOT as r
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from utility_function import angular_distance
from matching_tof_and_track_plotter import MatchingTOFAndTrackPlotter
from tof_analyzer import TOFHitInfo
from tqdm.auto import tqdm

@dataclass
class MatchedTrackInfo:
    """TOF ↔ Track matching result."""
    event:            ak.Array
    track_x:          ak.Array
    track_y:          ak.Array
    track_z:          ak.Array
    track_px:         ak.Array
    track_py:         ak.Array
    track_pz:         ak.Array
    track_p:          ak.Array
    track_pt:         ak.Array
    track_theta:      ak.Array
    track_phi:        ak.Array
    track_pathlength: ak.Array
    hit_x:            ak.Array
    hit_y:            ak.Array
    hit_z:            ak.Array
    hit_time:         ak.Array
    hit_phi:          ak.Array
    hit_theta:        ak.Array
    hit_r:            ak.Array

class MatchingTOFAndTrack:
    def __init__(self, tof, track, rootfile, name: str):
        """
        Constructor for the MatchingTOFAndTrack class.
        """
        self.name = name
        self.rootfile = rootfile
        self.btof = tof
        self.track = track
        self.matching_tof_and_track_plotter = MatchingTOFAndTrackPlotter(rootfile, name)

    def matching_tof_and_track(
        self,
        track_segments_on_btof_df: pd.DataFrame,
        filtered_stable_btof_hit_info: pd.DataFrame,
        track_segments_on_etof_df: pd.DataFrame,
        filtered_stable_etof_hit_info: pd.DataFrame,
        verbose: bool = False,
        plot_verbose: bool = False,
    ):
        """
        TOF ↔︎ Track matching 

        Parameters
        ----------
        track_segments_on_btof_df : pd.DataFrame
            dataframe containing track segments points on barrel TOF
        filtered_stable_btof_hit_info : pd.DataFrame
            dataframe containing filtered stable barrel TOF hit particles information
        track_segments_on_etof_df : pd.DataFrame
            dataframe containing track segments points on endcap TOF
        filtered_stable_etof_hit_info : pd.DataFrame
            dataframe containing filtered stable endcap TOF hit particles information
        """
        btrk = track_segments_on_btof_df
        bhit = filtered_stable_btof_hit_info
        etrk = track_segments_on_etof_df
        ehit = filtered_stable_etof_hit_info

        angle_threshold = 0.1  # rad

        btof_matched = {
            'event_idx': [], 'track_idx': [], 'track_p': [], 'track_pt': [],
            'track_pos_phi': [], 'track_pos_theta': [],
            'track_pos_x': [], 'track_pos_y': [], 'track_pos_z': [],
            'tof_pos_phi': [], 'tof_pos_theta': [], 'tof_time': [],
            'mc_pdg': [], 'mc_momentum': [],
            'mc_vertex_x': [], 'mc_vertex_y': [], 'mc_vertex_z': [],
            'track_pathlength': [], 'delta_angle': [],
        }

        for i, row in tqdm(btrk.iterrows(), total=len(btrk), desc='btof tof and track matching'):
            event_idx = row['event']
            tx, ty, tz = row['track_x'], row['track_y'], row['track_z']
            track_phi   = np.arctan2(ty, tx)
            track_theta = np.arccos(tz / np.sqrt(tx*tx + ty*ty + tz*tz))
            track_p     = row['track_p']
            track_pt    = row['track_pt']
            track_len   = row['track_pathlength']

            sub = bhit[bhit['event']==event_idx]
            if sub.empty:
                continue

            tof_phi   = np.arctan2(sub['tof_pos_y'], sub['tof_pos_x'])
            tof_theta = np.arccos(sub['tof_pos_z'] /
                                  np.sqrt(sub['tof_pos_x']**2 +
                                          sub['tof_pos_y']**2 +
                                          sub['tof_pos_z']**2))
            dangle = angular_distance(track_phi, track_theta,
                                    tof_phi.values, tof_theta.values)
            idx = np.argmin(dangle)
            if dangle[idx] > angle_threshold:
                continue

            btof_matched['event_idx'].append(event_idx)
            btof_matched['track_idx'].append(row['segment_id'])
            btof_matched['track_p'].append(track_p)
            btof_matched['track_pt'].append(track_pt)
            btof_matched['track_pos_phi'].append(track_phi)
            btof_matched['track_pos_theta'].append(track_theta)
            btof_matched['track_pos_x'].append(tx)
            btof_matched['track_pos_y'].append(ty)
            btof_matched['track_pos_z'].append(tz)
            btof_matched['tof_pos_phi'].append(tof_phi.values[idx])
            btof_matched['tof_pos_theta'].append(tof_theta.values[idx])
            btof_matched['tof_time'].append(sub['tof_time'].values[idx])
            btof_matched['mc_pdg'].append(sub['mc_pdg'].values[idx])
            btof_matched['mc_momentum'].append(sub['mc_momentum'].values[idx])
            btof_matched['mc_vertex_x'].append(sub['mc_vertex_x'].values[idx])
            btof_matched['mc_vertex_y'].append(sub['mc_vertex_y'].values[idx])
            btof_matched['mc_vertex_z'].append(sub['mc_vertex_z'].values[idx])
            btof_matched['track_pathlength'].append(track_len)
            btof_matched['delta_angle'].append(dangle[idx])

        btof_and_track_matched_df = pd.DataFrame(btof_matched)
        btof_and_track_matched_df.to_csv(f'./out/{self.name}/btof_and_track_matched.csv', index=False)

        etof_matched = {k: [] for k in btof_matched}
        for i, row in tqdm(etrk.iterrows(), total=len(etrk), desc='etof tof and track matching'):
            event_idx = row['event']
            tx, ty, tz = row['track_x'], row['track_y'], row['track_z']
            track_phi   = np.arctan2(ty, tx)
            track_theta = np.arccos(tz / np.sqrt(tx*tx + ty*ty + tz*tz))
            track_p     = row['track_p']
            track_pt    = row['track_pt']
            track_len   = row['track_pathlength']

            sub = ehit[ehit['event']==event_idx]
            if sub.empty:
                continue

            tof_phi   = np.arctan2(sub['tof_pos_y'], sub['tof_pos_x'])
            tof_theta = np.arccos(sub['tof_pos_z'] /
                                  np.sqrt(sub['tof_pos_x']**2 +
                                          sub['tof_pos_y']**2 +
                                          sub['tof_pos_z']**2))
            dangle = angular_distance(track_phi, track_theta,
                                    tof_phi.values, tof_theta.values)
            idx = np.argmin(dangle)
            if dangle[idx] > angle_threshold:
                continue

            etof_matched['event_idx'].append(event_idx)
            etof_matched['track_idx'].append(row['segment_id'])
            etof_matched['track_p'].append(track_p)
            etof_matched['track_pt'].append(track_pt)
            etof_matched['track_pos_phi'].append(track_phi)
            etof_matched['track_pos_theta'].append(track_theta)
            etof_matched['track_pos_x'].append(tx)
            etof_matched['track_pos_y'].append(ty)
            etof_matched['track_pos_z'].append(tz)
            etof_matched['tof_pos_phi'].append(tof_phi.values[idx])
            etof_matched['tof_pos_theta'].append(tof_theta.values[idx])
            etof_matched['tof_time'].append(sub['tof_time'].values[idx])
            etof_matched['mc_pdg'].append(sub['mc_pdg'].values[idx])
            etof_matched['mc_momentum'].append(sub['mc_momentum'].values[idx])
            etof_matched['mc_vertex_x'].append(sub['mc_vertex_x'].values[idx])
            etof_matched['mc_vertex_y'].append(sub['mc_vertex_y'].values[idx])
            etof_matched['mc_vertex_z'].append(sub['mc_vertex_z'].values[idx])
            etof_matched['track_pathlength'].append(track_len)
            etof_matched['delta_angle'].append(dangle[idx])

        etof_and_track_matched_df = pd.DataFrame(etof_matched)
        etof_and_track_matched_df.to_csv(f'./out/{self.name}/etof_and_track_matched.csv', index=False)

        if plot_verbose:
            self.matching_tof_and_track_plotter.plot_matched_tracks(
                btof_and_track_matched_df
            )
            self.matching_tof_and_track_plotter.plot_matched_tracks(
                etof_and_track_matched_df
            )

        return btof_and_track_matched_df, etof_and_track_matched_df