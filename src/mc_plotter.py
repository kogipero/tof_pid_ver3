import ROOT as r
import helper_functions as myfunc

class MCPlotter:
    """Class for plotting MC particle information."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize MC plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_mc_info(self, data: dict) -> None:
        """
        Plot MC particle information.
        
        Args:
            data: Dictionary containing MC particle data
        """
        print('Start plotting MC particle information')
        
        plot_configs = [
            (data['px'], [-10, 10], 100,'px [GeV/c]', 'mc_px'),
            (data['py'], [-10, 10], 100,'py [GeV/c]', 'mc_py'),
            (data['pz'], [-10, 10], 100,'pz [GeV/c]', 'mc_pz'),
            (data['p'], [0, 20], 50,'p [GeV/c]', 'mc_p'),
            (data['theta'], [0, 3.2], 50,'theta [rad]', 'mc_theta'),
            (data['phi'], [-3.2, 3.2], 100,'phi [rad]', 'mc_phi'),
            (data['pdg'], [-1000, 1000], 500,'PDG code', 'mc_pdg'),
            (data['charge'], [-2, 2], 4,'Charge', 'mc_charge'),
            (data['generator_status'], [0, 10], 10,'Generator status', 'mc_generator_status'),
            (data['vertex_x'], [-100, 100], 100,'Vertex x [mm]', 'mc_vertex_x'),
            (data['vertex_y'], [-100, 100], 100,'Vertex y [mm]', 'mc_vertex_y'),
            (data['vertex_z'], [-100, 100], 100,'Vertex z [mm]', 'mc_vertex_z')
        ]
        
        for data_array, hist_range, nbins,xlabel, outputname in plot_configs:
            myfunc.make_histogram_root(
                data_array,
                nbins,
                hist_range=hist_range,
                title=f'MC_{outputname}',
                xlabel=xlabel,
                ylabel='Entries',
                outputname=f'{self.name}/{outputname}',
                rootfile=self.rootfile
            )