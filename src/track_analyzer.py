import uproot
import numpy as np
import awkward as ak
import ROOT as r
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import helper_functions as myfunc
from track_plotter import TrackPlotter

@dataclass
class TrackSegmentInfo:
    """Data class to store track segment information."""
    event: ak.Array
    x: ak.Array
    y: ak.Array
    z: ak.Array
    px: ak.Array
    py: ak.Array
    pz: ak.Array
    p: ak.Array
    pt: ak.Array
    theta: ak.Array
    phi: ak.Array
    pathlength: ak.Array
    segment_id: int  

@dataclass
class TrackInfo:
    """Data class to store complete track information."""
    segments: List[TrackSegmentInfo]
    on_btof: Optional[ak.Array] = None
    on_etof: Optional[ak.Array] = None

class TrackAnalyzer:
    """Class for reading and processing track data."""
    
    def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile: r.TFile):
        """
        Initialize track reader.
        
        Args:
            dis_file: Input ROOT file containing track data
            branch: Dictionary of branch names
            name: Name for output files
            rootfile: Output ROOT file
        """
        self.name = name
        self.rootfile = rootfile
        self.branch = branch
        self.dis_file = dis_file
        self.track_plotter = TrackPlotter(rootfile, name)
        
        # Validate required branches
        self._validate_branches()
    
    def _validate_branches(self) -> None:
        """Validate that all required branches are present."""
        required_branches = ['points_branch']
        for branch in required_branches:
            if branch not in self.branch:
                raise ValueError(f"Missing required branch: {branch}")
    
    def get_track_segments_pos(self, verbose: bool = False, plot_verbose: bool = False) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
        """
        Get track segment positions.
        
        Args:
            verbose: Whether to print detailed information
            plot_verbose: Whether to generate plots
            
        Returns:
            Tuple of arrays containing x, y, z positions, and track segment indices
        """
        try:
            x = self.dis_file[self.branch['points_branch'][0]].array(library='ak')
            y = self.dis_file[self.branch['points_branch'][1]].array(library='ak')
            z = self.dis_file[self.branch['points_branch'][2]].array(library='ak')
            
            if plot_verbose:
                self._plot_track_positions(x, y, z)
            
            return x, y, z
        except Exception as e:
            raise RuntimeError(f"Error getting track segment positions: {str(e)}")
    
    def get_track_segments_momentum(self, verbose: bool = False, plot_verbose: bool = False) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
        """
        Get track segment momentum.
        
        Args:
            verbose: Whether to print detailed information
            plot_verbose: Whether to generate plots
            
        Returns:
            Tuple of arrays containing momentum components and related quantities
        """
        try:
            px = self.dis_file[self.branch['points_branch'][3]].array(library='ak')
            py = self.dis_file[self.branch['points_branch'][4]].array(library='ak')
            pz = self.dis_file[self.branch['points_branch'][5]].array(library='ak')
            pathlength = self.dis_file[self.branch['points_branch'][6]].array(library='ak')
            
            p = np.sqrt(px**2 + py**2 + pz**2)
            pt = np.sqrt(px**2 + py**2)
            theta = np.arctan2(pt, pz)
            phi = np.arctan2(py, px)
            
            if plot_verbose:
                self._plot_track_momenta(px, py, pz, p, pt, theta, phi, pathlength)
            
            return px, py, pz, p, pt, theta, phi, pathlength
        except Exception as e:
            raise RuntimeError(f"Error getting track segment momentum: {str(e)}")
        
    def _plot_track_positions(self, x, y, z):
        """Histogram x, y, z positions."""
        cfg = [
            (x, (-1000, 1000), 100, "x [mm]", "trk_x"),
            (y, (-1000, 1000), 100, "y [mm]", "trk_y"),
            (z, (-1000, 1000), 100, "z [mm]", "trk_z"),
        ]
        for arr, hr, nb, lab, out in cfg:
            myfunc.make_histogram_root(
                ak.to_numpy(ak.flatten(arr, axis=None)),
                nbins=nb,
                hist_range=hr,
                title=f"Track_{out}",
                xlabel=lab, ylabel="Entries",
                outputname=f"{self.name}/{out}",
                rootfile=self.rootfile,
            )

    def _plot_track_momenta(self, px, py, pz, p, pt, theta, phi, pathlength):
        """Histogram momentum-related variables."""
        cfg = [
            (px, (0, 30), 30, "px [GeV/c]",  "trk_px"),
            (py, (0, 30), 30, "py [GeV/c]",  "trk_py"),
            (pz, (0, 35), 30, "pz [GeV/c]",  "trk_pz"),
            (p,   (0, 20),   100, "p  [GeV/c]",  "trk_p"),
            (pt,  (0, 20),   100, "pt [GeV/c]",  "trk_pt"),
            (theta, (0, 3.2), 100, "theta [rad]", "trk_theta"),
            (phi, (-3.2, 3.2), 200, "phi [rad]", "trk_phi"),
            (pathlength, (0, 3000), 300, "pathlength [mm]", "trk_path"),
        ]
        for arr, hr, nb, lab, out in cfg:
            myfunc.make_histogram_root(
                ak.to_numpy(ak.flatten(arr, axis=None)),
                nbins=nb,
                hist_range=hr,
                title=f"Track_{out}",
                xlabel=lab, ylabel="Entries",
                outputname=f"{self.name}/{out}",
                rootfile=self.rootfile,
            )
    
    def split_track_segments(
        self,
        x_positions: ak.Array,
        y_positions: ak.Array,
        z_positions: ak.Array,
        px_momenta: ak.Array,
        py_momenta: ak.Array,
        pz_momenta: ak.Array,
        track_segment_pathlength: ak.Array,
        margin_theta: float = 0.6,
        margin_phi: float = 0.6,
        verbose: bool = False,
        plot_verbose: bool = False,
        SELECTED_EVENTS: int = 10000
    ) -> List[TrackInfo]:
        """
        Split track segments into individual tracks.
        
        Args:
            x_positions: Array of x positions
            y_positions: Array of y positions
            z_positions: Array of z positions
            px_momenta: Array of px momenta
            py_momenta: Array of py momenta
            pz_momenta: Array of pz momenta
            track_segment_pathlength: Array of path lengths
            margin_theta: Angular margin for track splitting
            margin_phi: Angular margin for track splitting
            verbose: Whether to print detailed information
            plot_verbose: Whether to generate plots
            SELECTED_EVENTS: Number of events to process
            
        Returns:
            List of TrackInfo objects containing track information
        """
        try:
            tracks = []
            for event in tqdm(range(SELECTED_EVENTS), desc="Split track segments", unit="event"):
                track_segments = []
                current_track = []
                
                for i in range(len(x_positions[event])):
                    if len(current_track) == 0:
                        current_track.append(i)
                    else:
                        last_idx = current_track[-1]
                        theta_diff = abs(np.arctan2(
                            np.sqrt(px_momenta[event][i]**2 + py_momenta[event][i]**2),
                            pz_momenta[event][i]
                        ) - np.arctan2(
                            np.sqrt(px_momenta[event][last_idx]**2 + py_momenta[event][last_idx]**2),
                            pz_momenta[event][last_idx]
                        ))
                        phi_diff = abs(np.arctan2(
                            py_momenta[event][i],
                            px_momenta[event][i]
                        ) - np.arctan2(
                            py_momenta[event][last_idx],
                            px_momenta[event][last_idx]
                        ))
                        
                        if theta_diff > margin_theta or phi_diff > margin_phi:
                            track_segments.append(current_track)
                            current_track = [i]
                        else:
                            current_track.append(i)
                
                if current_track:
                    track_segments.append(current_track)
                
                track_info = TrackInfo(segments=[])
                for sid, segment in enumerate(track_segments):
                    segment_info = TrackSegmentInfo(
                        event=event,
                        x=x_positions[event][segment],
                        y=y_positions[event][segment],
                        z=z_positions[event][segment],
                        px=px_momenta[event][segment],
                        py=py_momenta[event][segment],
                        pz=pz_momenta[event][segment],
                        p=np.sqrt(px_momenta[event][segment]**2 + py_momenta[event][segment]**2 + pz_momenta[event][segment]**2),
                        pt=np.sqrt(px_momenta[event][segment]**2 + py_momenta[event][segment]**2),
                        theta=np.arctan2(
                            np.sqrt(px_momenta[event][segment]**2 + py_momenta[event][segment]**2),
                            pz_momenta[event][segment]
                        ),
                        phi=np.arctan2(py_momenta[event][segment], px_momenta[event][segment]),
                        pathlength=track_segment_pathlength[event][segment],
                        segment_id=sid
                    )
                    track_info.segments.append(segment_info)
                
                tracks.append(track_info)
                
            return tracks
        except Exception as e:
            raise RuntimeError(f"Error splitting track segments: {str(e)}")
    
    def get_track_segments_on_tof_info(
        self,
        tracks: List[TrackInfo],
        verbose: bool = False,
        plot_verbose: bool = False
    ) -> Tuple[ak.Array, ak.Array]:
        """
        Get track segments on Barrel/Endcap TOF .
        
        Args:
            tracks: List of TrackInfo objects
            verbose: Whether to print detailed information
            plot_verbose: Whether to generate plots
            
        Returns:
            Tuple of arrays containing track segments on barrel and endcap TOF
        """
        try:
            btof = {k: [] for k in (
                "event", "track_x","track_y","track_z","track_r","track_phi","track_theta","track_pathlength", "segment_id", "track_p", "track_pt"
            )}
            etof = {k: [] for k in btof}

            for trk in tqdm(tracks, desc="Get track segments on TOF", unit="track"):
                for seg in trk.segments:
                    r_arr = np.sqrt(seg.x**2 + seg.y**2)

                    # Barrel mask
                    barrel_mask = (
                        (r_arr >= 625) & (r_arr <= 660)
                        & (seg.z >= -1150) & (seg.z <= 1740)
                        & (seg.pathlength >= 580)
                    )
                    # Endcap mask
                    endcap_mask = (
                        (r_arr >= 100) & (r_arr <= 600)
                        & ((seg.z <= 1840) | (seg.z >= 1880))
                        & (seg.pathlength >= 1580)
                    )

                    for xi, yi, zi, ri, pli, pi, pti in zip(
                        seg.x[barrel_mask],
                        seg.y[barrel_mask],
                        seg.z[barrel_mask],
                        r_arr[barrel_mask],
                        seg.pathlength[barrel_mask],
                        seg.p[barrel_mask],
                        seg.pt[barrel_mask]
                    ):
                        phi_i   = np.arctan2(yi, xi)
                        theta_i = np.arctan2(ri, zi)

                        btof["event"].append(seg.event)
                        btof["track_x"].append(xi)
                        btof["track_y"].append(yi)
                        btof["track_z"].append(zi)
                        btof["track_r"].append(ri)
                        btof["track_phi"].append(phi_i)
                        btof["track_theta"].append(theta_i)
                        btof["track_pathlength"].append(pli)
                        btof["segment_id"].append(seg.segment_id)
                        btof["track_p"].append(pi)
                        btof["track_pt"].append(pti)

                    for xi, yi, zi, ri, pli, pi, pti in zip(
                        seg.x[endcap_mask],
                        seg.y[endcap_mask],
                        seg.z[endcap_mask],
                        r_arr[endcap_mask],
                        seg.pathlength[endcap_mask],
                        seg.p[endcap_mask],
                        seg.pt[endcap_mask]
                    ):
                        phi_i   = np.arctan2(yi, xi)
                        theta_i = np.arctan2(ri, zi)

                        etof["event"].append(seg.event)
                        etof["track_x"].append(xi)
                        etof["track_y"].append(yi)
                        etof["track_z"].append(zi)
                        etof["track_r"].append(ri)
                        etof["track_phi"].append(phi_i)
                        etof["track_theta"].append(theta_i)
                        etof["track_pathlength"].append(pli)
                        etof["segment_id"].append(seg.segment_id)
                        etof["track_p"].append(pi)
                        etof["track_pt"].append(pti)

            btof_dict = {k: ak.Array(v) for k, v in btof.items()}
            etof_dict = {k: ak.Array(v) for k, v in etof.items()}

            if plot_verbose:
                self.track_plotter.plot_track_segments_on_tof(btof_dict, area="btof")
                self.track_plotter.plot_track_segments_on_tof(etof_dict, area="etof")

            btof_df = pd.DataFrame(btof_dict)
            etof_df = pd.DataFrame(etof_dict)

            btof_df.to_csv(f"./out/{self.name}/track_segments_on_btof.csv", index=False)
            etof_df.to_csv(f"./out/{self.name}/track_segments_on_etof.csv", index=False)

            return btof_df, etof_df

        except Exception as e:
            raise RuntimeError(f"Error getting track segments on TOF: {str(e)}")