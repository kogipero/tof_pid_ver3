import ROOT as r
import helper_functions as myfunc

class MatchingPlotter:
    """Class for plotting matched hit and track information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize matching plotter.
        
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
            (data['hit_x'], [-1000, 1000], 'Hit x [mm]', 'hit_x'),
            (data['hit_y'], [-1000, 1000], 'Hit y [mm]', 'hit_y'),
            (data['hit_z'], [-2000, 2000], 'Hit z [mm]', 'hit_z'),
            (data['hit_time'], [0, 100], 'Hit time [ns]', 'hit_time'),
            (data['hit_phi'], [-3.2, 3.2], 'Hit phi [rad]', 'hit_phi'),
            (data['hit_theta'], [0, 3.2], 'Hit theta [rad]', 'hit_theta'),
            (data['hit_r'], [0, 2000], 'Hit r [mm]', 'hit_r'),
            (data['mc_px'], [-10, 10], 'MC px [GeV/c]', 'mc_px'),
            (data['mc_py'], [-10, 10], 'MC py [GeV/c]', 'mc_py'),
            (data['mc_pz'], [-10, 10], 'MC pz [GeV/c]', 'mc_pz'),
            (data['mc_p'], [0, 20], 'MC p [GeV/c]', 'mc_p'),
            (data['mc_theta'], [0, 3.2], 'MC theta [rad]', 'mc_theta'),
            (data['mc_phi'], [-3.2, 3.2], 'MC phi [rad]', 'mc_phi'),
            (data['mc_pdg'], [-1000, 1000], 'MC PDG code', 'mc_pdg'),
            (data['mc_charge'], [-2, 2], 'MC charge', 'mc_charge')
        ]
        
        for data_array, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                100,
                hist_range=hist_range,
                title=f'Matched_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        
        print(f'End plotting matched hits for {area}')
    
    def plot_matched_tracks(self, data: dict, area: str = '') -> None:
        """
        Plot matched track information.
        
        Args:
            data: Dictionary containing matched track data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        print(f'Start plotting matched tracks for {area}')
        
        plot_configs = [
            (data['track_px'], [-10, 10], 'Track px [GeV/c]', 'track_px'),
            (data['track_py'], [-10, 10], 'Track py [GeV/c]', 'track_py'),
            (data['track_pz'], [-10, 10], 'Track pz [GeV/c]', 'track_pz'),
            (data['track_p'], [0, 20], 'Track p [GeV/c]', 'track_p'),
            (data['track_pt'], [0, 10], 'Track pt [GeV/c]', 'track_pt'),
            (data['track_theta'], [0, 3.2], 'Track theta [rad]', 'track_theta'),
            (data['track_phi'], [-3.2, 3.2], 'Track phi [rad]', 'track_phi'),
            (data['hit_time'], [0, 100], 'Hit time [ns]', 'hit_time'),
            (data['hit_phi'], [-3.2, 3.2], 'Hit phi [rad]', 'hit_phi'),
            (data['hit_theta'], [0, 3.2], 'Hit theta [rad]', 'hit_theta'),
            (data['hit_r'], [0, 2000], 'Hit r [mm]', 'hit_r')
        ]
        
        for data_array, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                100,
                hist_range=hist_range,
                title=f'Matched_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        
        print(f'End plotting matched tracks for {area}') 