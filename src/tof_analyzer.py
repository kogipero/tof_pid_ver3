from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import uproot
import awkward as ak
import ROOT as r
import helper_functions as myfunc

@dataclass
class TOFHitInfo:
    """Data class to store TOF hit information."""
    pos_x: ak.Array
    pos_y: ak.Array
    pos_z: ak.Array
    time: ak.Array
    phi: ak.Array
    theta: ak.Array
    r: ak.Array

class TOFAnalyzer:
    """Class for analyzing Time-of-Flight detector data."""
    
    def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile: r.TFile):
        """
        Initialize TOF analyzer.
        
        Args:
            dis_file: Input ROOT file containing TOF data
            branch: Dictionary of branch names
            name: Name for output files
            rootfile: Output ROOT file
        """
        self.name = name
        self.rootfile = rootfile
        self.branch = branch
        self.dis_file = dis_file
        
        # Validate required branches
        self._validate_branches()

    def _branch_list(self, area: str, kind: str) -> list[str]:
        """
        Get list of branches for a given area and kind.
        
        Args:
            area: Area of the TOF detector (e.g., 'barrel', 'endcap')
            kind: Type of branch (e.g., 'raw_hits_branch', 'rec_hits_branch')
            
        Returns:
            List of branch names
        """
        try:
            return self.branch[area][kind]
        except KeyError:
            raise RuntimeError(f"Invalid area or kind: {area}, {kind}")
    
    def _validate_branches(self) -> None:
        """Validate that all required branches are present."""
        req = [
            ("barrel", "raw_hits_branch"),
            ("barrel", "rec_hits_branch"),
            ("barrel", "mc_associations_branch"),
            ("barrel", "mc_associations_ver1_24_2_branch"),
            ("endcap", "raw_hits_branch"),
            ("endcap", "rec_hits_branch"),
            ("endcap", "mc_associations_branch"),
            ("endcap", "mc_associations_ver1_24_2_branch"),
        ]
        for area, kind in req:
            _ = self._branch_list(area, kind)
        print(f"All required branches are present for {self.name}")
    
    def _get_raw_hit_info(self, area: str, selected_events: int) -> TOFHitInfo:
        """
        Extract hit information for a given TOF detector.
        
        Args:
            branch_name: Name of the branch containing hit information
            selected_events: Number of events to process
            
        Returns:
            TOFHitInfo object containing hit information
        """
        br = self._branch_list(area, "raw_hits_branch")

        time  = self.dis_file[br[0]].array(library='ak')[:selected_events]
        pos_x = self.dis_file[br[1]].array(library='ak')[:selected_events]
        pos_y = self.dis_file[br[2]].array(library='ak')[:selected_events]
        pos_z = self.dis_file[br[3]].array(library='ak')[:selected_events]

        r     = np.sqrt(pos_x**2 + pos_y**2)
        phi   = np.arctan2(pos_y, pos_x)
        theta = np.arctan2(r, pos_z)

        return TOFHitInfo(pos_x, pos_y, pos_z, time, phi, theta, r)
    
    def _plot_hit_info(self, hit_info: TOFHitInfo, area: str) -> None:
        """
        Plot TOF hit information.
        
        Args:
            hit_info: TOFHitInfo object containing hit information
            area: Name of the TOF detector area (e.g., 'btof', 'etof')
        """
        
        # Define plot configurations
        plot_configs = [
            (hit_info.pos_x, [-1000, 1000], 'x [mm]', 'tof_x'),
            (hit_info.pos_y, [-1000, 1000], 'y [mm]', 'tof_y'),
            (hit_info.pos_z, [-2000, 2000], 'z [mm]', 'tof_z'),
            (hit_info.time, [0, 100], 'time [ns]', 'tof_time'),
            (hit_info.phi, [-3.2, 3.2], 'phi [rad]', 'tof_phi'),
            (hit_info.theta, [0, 3.2], 'theta [rad]', 'tof_theta'),
            (hit_info.r, [0, 1000], 'r [mm]', 'tof_r')
        ]
        
        for data, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                ak.flatten(data),
                100,
                hist_range=hist_range,
                title=f'TOF_rec_hit_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
            
    def get_tof_info(self, selected_events: int, plot_verbose: bool = False) -> Tuple[TOFHitInfo, TOFHitInfo]:
        """
        Get TOF hit information for both barrel and endcap detectors.
        
        Args:
            selected_events: Number of events to process
            plot_verbose: Whether to generate plots
            
        Returns:
            Tuple of TOFHitInfo objects for barrel and endcap detectors
        """
        btof_info = self._get_raw_hit_info("barrel", selected_events)
        etof_info = self._get_raw_hit_info("endcap", selected_events)

        if plot_verbose:
            self._plot_hit_info(btof_info, "btof")
            self._plot_hit_info(etof_info, "etof")

        return btof_info, etof_info