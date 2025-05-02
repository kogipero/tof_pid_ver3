import ROOT as r
import helper_functions as myfunc

class TOFPlotter:
    """Class for plotting TOF hit information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize TOF plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_tof_hits(self, data: dict, area: str = '') -> None:
        """
        Plot TOF hit information.
        
        Args:
            data: Dictionary containing TOF hit data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        print(f'Start plotting TOF hits for {area}')
        
        plot_configs = [
            (data['x'], [-1000, 1000], 'x [mm]', 'tof_x'),
            (data['y'], [-1000, 1000], 'y [mm]', 'tof_y'),
            (data['z'], [-2000, 2000], 'z [mm]', 'tof_z'),
            (data['time'], [0, 100], 'Time [ns]', 'tof_time'),
            (data['phi'], [-3.2, 3.2], 'phi [rad]', 'tof_phi'),
            (data['theta'], [0, 3.2], 'theta [rad]', 'tof_theta'),
            (data['r'], [0, 2000], 'r [mm]', 'tof_r')
        ]
        
        for data_array, hist_range, xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                100,
                hist_range=hist_range,
                title=f'TOF_{outputname}_{area}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )
        
        print(f'End plotting TOF hits for {area}') 