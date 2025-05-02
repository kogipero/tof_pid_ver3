import ROOT as r
import helper_functions as myfunc
import numpy as np

class TOFPIDPerformancePlotter:
    """Class for plotting TOF PID performance evaluation results."""
    
    def __init__(self, rootfile: r.TFile, name: str):
        """
        Initialize TOF PID performance plotter.
        
        Args:
            rootfile: Output ROOT file
            name: Name for output files
        """
        self.rootfile = rootfile
        self.name = name
    
    def plot_pid_performance(self, data: dict, area: str = '') -> None:
        """
        Plot PID performance evaluation results.
        
        Args:
            data: Dictionary containing PID performance data
            area: Area identifier (e.g., 'btof', 'etof')
        """
        print(f'Start plotting PID performance for {area}')
        
        # Plot time distributions for different particle types
        for particle_type in data['particle_types']:
            time_data = data[f'time_{particle_type}']
            myfunc.make_histogram_root(
                time_data,
                100,
                hist_range=[0, 100],
                title=f'Time distribution for {particle_type}',
                xlabel='Time [ns]',
                ylabel='Entries',
                outputname=f'{self.name}/time_{particle_type}',
                rootfile=self.rootfile
            )
        
        # Plot momentum distributions
        myfunc.make_histogram_root(
            data['momentum'],
            100,
            hist_range=[0, 20],
            title='Momentum distribution',
            xlabel='Momentum [GeV/c]',
            ylabel='Entries',
            outputname=f'{self.name}/momentum',
            rootfile=self.rootfile
        )
        
        # Plot separation power
        if 'separation_power' in data:
            myfunc.make_histogram_root(
                data['separation_power'],
                100,
                hist_range=[0, 10],
                title='Separation power',
                xlabel='Separation power',
                ylabel='Entries',
                outputname=f'{self.name}/separation_power',
                rootfile=self.rootfile
            )
        
        # Plot efficiency and purity
        if 'efficiency' in data and 'purity' in data:
            momentum_bins = np.linspace(0, 20, 21)
            efficiency = data['efficiency']
            purity = data['purity']
            
            # Create efficiency graph
            eff_graph = r.TGraph(len(momentum_bins)-1)
            for i in range(len(momentum_bins)-1):
                eff_graph.SetPoint(i, (momentum_bins[i] + momentum_bins[i+1])/2, efficiency[i])
            
            eff_graph.SetTitle('Efficiency vs Momentum')
            eff_graph.GetXaxis().SetTitle('Momentum [GeV/c]')
            eff_graph.GetYaxis().SetTitle('Efficiency')
            eff_graph.Write(f'{self.name}/efficiency')
            
            # Create purity graph
            pur_graph = r.TGraph(len(momentum_bins)-1)
            for i in range(len(momentum_bins)-1):
                pur_graph.SetPoint(i, (momentum_bins[i] + momentum_bins[i+1])/2, purity[i])
            
            pur_graph.SetTitle('Purity vs Momentum')
            pur_graph.GetXaxis().SetTitle('Momentum [GeV/c]')
            pur_graph.GetYaxis().SetTitle('Purity')
            pur_graph.Write(f'{self.name}/purity')
        
        print(f'End plotting PID performance for {area}') 