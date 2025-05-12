from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import uproot
import awkward as ak
import ROOT as r
import helper_functions as myfunc
from mc_plotter import MCPlotter

@dataclass
class MCInfo:
    """Data class to store MC particle information."""
    px: ak.Array
    py: ak.Array
    pz: ak.Array
    p: ak.Array
    theta: ak.Array
    phi: ak.Array
    pdg: ak.Array
    charge: ak.Array
    generator_status: ak.Array
    vertex_x: ak.Array
    vertex_y: ak.Array
    vertex_z: ak.Array

class MCAnalyzer:
    """Class for reading and processing MC particle data."""
    
    def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile: r.TFile):
        """
        Initialize MC reader.
        
        Args:
            dis_file: Input ROOT file containing MC data
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
    
    def _validate_branches(self) -> None:
        """Validate that all required branches are present."""
        required_branches = ['mc_branch']
        for branch in required_branches:
            if branch not in self.branch:
                raise ValueError(f"Missing required branch: {branch}")
    
    def get_mc_info(self, verbose: bool = False, plot_verbose: bool = False) -> MCInfo:
        """
        Get MC particle information.
        
        Args:
            verbose: Whether to print detailed information
            plot_verbose: Whether to generate plots
            
        Returns:
            MCInfo object containing MC particle information
        """
        try:
            pdg = self.dis_file[self.branch['mc_branch'][0]].array(library='ak')
            generator_status = self.dis_file[self.branch['mc_branch'][1]].array(library='ak')
            charge = self.dis_file[self.branch['mc_branch'][2]].array(library='ak')
            px = self.dis_file[self.branch['mc_branch'][3]].array(library='ak')
            py = self.dis_file[self.branch['mc_branch'][4]].array(library='ak')
            pz = self.dis_file[self.branch['mc_branch'][5]].array(library='ak')
            vertex_x = self.dis_file[self.branch['mc_branch'][6]].array(library='ak')
            vertex_y = self.dis_file[self.branch['mc_branch'][7]].array(library='ak')
            vertex_z = self.dis_file[self.branch['mc_branch'][8]].array(library='ak')
            
            p = np.sqrt(px**2 + py**2 + pz**2)
            theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
            phi = np.arctan2(py, px)
            
            if plot_verbose:
                self._plot_mc_info(px, py, pz, p, theta, phi, pdg, charge, generator_status, vertex_x, vertex_y, vertex_z)
            
            return MCInfo(
                px=px,
                py=py,
                pz=pz,
                p=p,
                theta=theta,
                phi=phi,
                pdg=pdg,
                charge=charge,
                generator_status=generator_status,
                vertex_x=vertex_x,
                vertex_y=vertex_y,
                vertex_z=vertex_z
            )
        except Exception as e:
            raise RuntimeError(f"Error getting MC information: {str(e)}")
    
    def _plot_mc_info(self, px: ak.Array, py: ak.Array, pz: ak.Array, p: ak.Array, theta: ak.Array, phi: ak.Array, pdg: ak.Array, charge: ak.Array, generator_status: ak.Array, vertex_x: ak.Array, vertex_y: ak.Array, vertex_z: ak.Array) -> None:
        """Plot MC particle information."""
        
        plot_configs = [
            (px, [-20, 20], 100, 'px [GeV/c]', 'px'),
            (py, [-20, 20], 100, 'py [GeV/c]', 'py'),
            (pz, [-200, 400], 100, 'pz [GeV/c]', 'pz'),
            (p, [0, 5], 50, 'p [GeV/c]', 'p'),
            (theta, [0, 3.2], 50, 'theta [rad]', 'p_theta'),
            (phi, [-3.2, 3.2], 100, 'phi [rad]', 'p_phi'),
            (pdg, [-250, 250], 500, 'PDG code', 'pdg'),
            (charge, [-2, 2], 4, 'Charge', 'mc_charge'),
            (generator_status, [0, 10], 10, 'Generator status', 'generator_status'),
            (vertex_x, [-200, 200], 100, 'Vertex x [mm]', 'vertex_x'),
            (vertex_y, [-200, 200], 100, 'Vertex y [mm]', 'vertex_y'),
            (vertex_z, [-200, 200], 100, 'Vertex z [mm]', 'vertex_z')
        ]
        
        for data, hist_range, nbins,xlabel, outputname in plot_configs:
            arr = ak.to_numpy(ak.flatten(data))
            myfunc.make_histogram_root(
                arr,
                nbins,
                hist_range=hist_range,
                title=f'mc_{outputname}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        
        print('End plotting MC information')