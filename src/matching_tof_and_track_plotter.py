import ROOT as r
import helper_functions as myfunc

class MatchingTOFAndTrackPlotter:
    """Class for plotting TOF and track matching information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize TOF and track matching plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_matched_tracks(self, data: dict, area: str = '') -> None:
        """
        Plot matched track information.
        
        Args:
            data: Dictionary containing matched track data
            area: Area identifier (e.g., 'btof', 'etof')
        """        
        plot_configs = [
            (data['track_pos_x'], [-10, 10], 'Track px [GeV/c]', 'track_px'),
            (data['track_pos_y'], [-10, 10], 'Track py [GeV/c]', 'track_py'),
            (data['track_pos_z'], [-10, 10], 'Track pz [GeV/c]', 'track_pz'),
            (data['track_p'], [0, 20], 'Track p [GeV/c]', 'track_p'),
            (data['track_pt'], [0, 10], 'Track pt [GeV/c]', 'track_pt'),
            (data['track_pos_theta'], [0, 3.2], 'Track theta [rad]', 'track_theta'),
            (data['track_pos_phi'], [-3.2, 3.2], 'Track phi [rad]', 'track_phi'),
            (data['tof_time'], [0, 100], 'tof time [ns]', 'tof_time'),
            (data['tof_pos_phi'], [-3.2, 3.2], 'Hit phi [rad]', 'hit_phi'),
            (data['tof_pos_theta'], [0, 3.2], 'Hit theta [rad]', 'hit_theta'),
            (data['mc_pdg'], [-1000, 1000], 'mc_pdg', 'mc_pdg'),
            (data['mc_momentum'], [0, 10], 'mc_momentum', 'mc_momentum'),
            (data['mc_vertex_x'], [-100, 100], 'Hit y [mm]', 'hit_y'),
            (data['mc_vertex_y'], [-100, 100], 'Hit z [mm]', 'hit_z'),
            (data['mc_vertex_z'], [-100, 100], 'Hit z [mm]', 'hit_z'),
            (data['track_pathlength'], [0, 2000], 'track_pathlength [mm]', 'track_pathlength'),
            (data['delta_angle']*1000, [0, 2.0], 'delta_angle [mrad]', 'delta_angle_mrad'),
            (data['delta_angle'], [0, 2.0], 'delta_angle [rad]', 'delta_angle_rad'),
        ]

        plot_2d_configs = [
            (data['track_pos_theta'], data['track_pathlength'], [0, 3.2], [0, 2000], 'Track theta [rad]', 'Track pathlength [mm]', 'track_theta_pathlength'),
            (data['mc_vertex_z'], data['tof_time'], [-10, 10], [0, 10], 'vertex z [mm]', 'Hit time [ns]', 'hit_z_time'),
            (data['mc_vertex_z'], data['track_pathlength'], [-10, 10], [0, 2000], 'vertex z [mm]', 'Track pathlength [mm]', 'hit_z_pathlength'),
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

        for x_data, y_data, x_range, y_range, xlabel, ylabel, outputname in plot_2d_configs:
            myfunc.make_2Dhistogram_root(
                arrayx=x_data,
                arrayy=y_data,
                nbinsx=100,
                rangex=x_range,
                nbinsy=100,
                rangey=y_range,
                title=f'Matched_{outputname}_{area}',
                xlabel=xlabel,
                ylabel=ylabel,
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )