import ROOT as r
import helper_functions as myfunc

class MatchingMCAndTOFPlotter:
    """Class for plotting MC and TOF matching information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize MC and TOF matching plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_matched_hits(self, data: dict, area: str = '') -> None:
        """
        Plot matched hit information.
        
        Args:
            data: Dictionary containing matched hit data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        print(f'Start plotting matched hits for {area}')
        
        plot_configs = [
            (data['tof_pos_x'], [-1000, 1000], 100,'Hit x [mm]', 'hit_x'),
            (data['tof_pos_x'], [-1000, 1000], 100,'Hit y [mm]', 'hit_y'),
            (data['tof_pos_x'], [-2000, 2000], 100,'Hit z [mm]', 'hit_z'),
            (data['tof_time'], [0, 100], 100,'Hit time [ns]', 'hit_time'),
            (data['tof_pos_phi'], [-3.2, 3.2], 100,'Hit phi [rad]', 'hit_phi'),
            (data['tof_pos_theta'], [0, 3.2], 100,'Hit theta [rad]', 'hit_theta'),
            (data['tof_pos_r'], [0, 2000], 100,'Hit r [mm]', 'hit_r'),
            (data['mc_momentum_x'], [-20, 20], 100,'MC px [GeV/c]', 'mc_px'),
            (data['mc_momentum_y'], [-20, 20], 100,'MC py [GeV/c]', 'mc_py'),
            (data['mc_momentum_z'], [-200, 400], 100,'MC pz [GeV/c]', 'mc_pz'),
            (data['mc_momentum'], [0, 5], 50,'MC p [GeV/c]', 'mc_p'),
            (data['mc_momentum_theta'], [0, 3.2], 50,'MC theta [rad]', 'mc_theta'),
            (data['mc_momentum_phi'], [-3.2, 3.2], 100,'MC phi [rad]', 'mc_phi'),
            (data['mc_pdg'], [-250, 250], 500,'MC PDG code', 'mc_pdg'),
            (data['mc_charge'], [-2, 2], 4,'MC charge', 'mc_charge'),
            (data['mc_generator_status'], [0, 10], 10,'MC generator status', 'mc_generator_status'),
            (data['mc_vertex_x'], [-200, 200], 100,'MC vertex x [mm]', 'mc_vertex_x'),
            (data['mc_vertex_y'], [-200, 200], 100,'MC vertex y [mm]', 'mc_vertex_y'),
            (data['mc_vertex_z'], [-200, 200], 100,'MC vertex z [mm]', 'mc_vertex_z')
        ]
        
        for data_array, hist_range, nbins, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                nbins,
                hist_range=hist_range,
                title=f'Matched_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        
        print(f'End plotting matched hits for {area}') 