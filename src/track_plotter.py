import ROOT as r
import helper_functions as myfunc

class TrackPlotter:
    """Class for plotting track information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize track plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_track_segments(self, data: dict, area: str = '') -> None:
        """
        Plot track segment information.
        
        Args:
            data: Dictionary containing track segment data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        
        plot_configs = [
            (data['x'], [-1000, 1000], 'x [mm]', 'track_x'),
            (data['y'], [-1000, 1000], 'y [mm]', 'track_y'),
            (data['z'], [-2000, 2000], 'z [mm]', 'track_z'),
            (data['px'], [-10, 10], 'px [GeV/c]', 'track_px'),
            (data['py'], [-10, 10], 'py [GeV/c]', 'track_py'),
            (data['pz'], [-10, 10], 'pz [GeV/c]', 'track_pz'),
            (data['p'], [0, 20], 'p [GeV/c]', 'track_p'),
            (data['pt'], [0, 10], 'pt [GeV/c]', 'track_pt'),
            (data['theta'], [0, 3.2], 'theta [rad]', 'track_theta'),
            (data['phi'], [-3.2, 3.2], 'phi [rad]', 'track_phi'),
            (data['pathlength'], [0, 1000], 'pathlength [mm]', 'track_pathlength')
        ]
        
        for data_array, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                100,
                hist_range=hist_range,
                title=f'Track_segment_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        

    def plot_track_segments_on_tof(self, data: dict, area: str = '') -> None:
        """
        Plot track segment information on TOF.
        
        Args:
            data: Dictionary containing track segment data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        
        plot_configs = [
            (data['track_x'], [-1000, 1000], f'track_x_on_{area} [mm]', f'track_x_on_{area}'),
            (data['track_y'], [-1000, 1000], f'track_y_on_{area}[mm]', f'track_y_on_{area}'),
            (data['track_z'], [-2000, 2000], f'track_z_on_{area}[mm]', f'track_z_on_{area}'),
            (data['track_phi'], [-3.2, 3.2], f'track_phi_on_{area}[rad]', f'track_phi_on_{area}'),
            (data['track_theta'], [0, 3.2], f'track_theta_on_{area}[rad]', f'track_theta_on_{area}'),
            (data['track_r'], [0, 2000], f'track_r_on_{area}[mm]', f'track_r_on_{area}'),
            (data['track_pathlength'], [0, 1500], 'track_pathlength [mm]', f'track_pathlength_on_{area}'),
        ]
        
        for data_array, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                100,
                hist_range=hist_range,
                title=f'Track_segment_on_TOF_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        