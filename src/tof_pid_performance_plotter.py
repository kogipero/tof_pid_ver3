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
    
    def plot_pid_performance(self, data: dict, area: str = "") -> None:

        pdg_map = {
            "pi": (211, -211),
            "k": (321, -321),
            "p": (2212, -2212),
        }

        pdg  = data.get("pdg", None)
        # ------------------------------------------------------------
        # β⁻¹ vs p   
        # ------------------------------------------------------------
        if "beta_inverse" in data and "momentum" in data:
            myfunc.make_2Dhistogram_root(
                data["momentum"],            
                data["beta_inverse"],         
                100, [0.0, 5.0],              
                100, [0.8, 1.8],              
                title     = "beta inverse vs p",    
                xlabel    = "p [GeV/c]",      
                ylabel    = "beta inverse",         
                outputname= f"beta_inv_vs_p_{area}",   
                rootfile  = self.rootfile
            )

        # ------------------------------------------------------------
        # reconstructed mass 
        # ------------------------------------------------------------
        if "calc_mass" in data:
            myfunc.make_histogram_root(
                data["calc_mass"], 120, [0, 1200],
                f"Reconstructed_Mass_{area};Mass [MeV];Entries",
                f"mass_hist_{area}", rootfile=self.rootfile
            )

        # ------------------------------------------------------------
        # momentum and β⁻¹
        # ------------------------------------------------------------
        if "momentum" in data:
            myfunc.make_histogram_root(
                data["momentum"], 100, [0, 3.5],
                f"Momentum_{area}; p [GeV/c];Entries",
                f"p_hist_{area}", rootfile=self.rootfile
            )
        if "beta_inverse" in data:
            myfunc.make_histogram_root(
                data["beta_inverse"], 100, [0, 1.8],
                f"beta inverse_{area}; beta inverse;Entries",
                f"beta_inv_hist_{area}", rootfile=self.rootfile
            )

        # ------------------------------------------------------------
        # pid each particle
        # ------------------------------------------------------------

        if len(pdg):
            for tag, pdgs in pdg_map.items():
                mask = np.isin(pdg, pdgs)

                # mass
                if mask.any():
                    myfunc.make_histogram_root(
                        data["calc_mass"][mask], 120, [0, 1200],
                        f"{tag.upper()} Mass ({area});Mass [MeV];Entries",
                        f"mass_{tag}_{area}", rootfile=self.rootfile
                    )
                    # β^-1 vs p
                    myfunc.make_2Dhistogram_root(
                        data["momentum"][mask],
                        data["beta_inverse"][mask],
                        100, [0.0, 3.5], 100, [0.8, 1.8],
                        title     = f"β^{{-1}} vs p ({tag})",
                        xlabel    = "p [GeV/c]",
                        ylabel    = "β^{-1}",
                        outputname= f"beta_inv_vs_p_{tag}_{area}",
                        rootfile  = self.rootfile
                    )

    def plot_purity_vs_momentum(
        self, bins,
        pi_norm, pi_err_norm, pi_uniq, pi_err_uniq,
        k_norm, k_err_norm, k_uniq, k_err_uniq,
        p_norm, p_err_norm, p_uniq, p_err_uniq,
    ):
        g_pi   = r.TGraphErrors(len(bins), bins, pi_norm, np.zeros_like(bins), pi_err_norm)
        g_pi.SetTitle("π purity; Momentum [GeV]; Purity")

        g_k    = r.TGraphErrors(len(bins), bins, k_norm, np.zeros_like(bins), k_err_norm)
        g_p    = r.TGraphErrors(len(bins), bins, p_norm, np.zeros_like(bins), p_err_norm)

        c = r.TCanvas(f"c_purity_{self.tag}", " ", 800, 600)
        g_pi.SetMarkerStyle(20); g_pi.Draw("AP")
        g_k.SetMarkerStyle(21); g_k.SetMarkerColor(r.kRed);   g_k.Draw("P SAME")
        g_p.SetMarkerStyle(22); g_p.SetMarkerColor(r.kBlue);  g_p.Draw("P SAME")
        c.BuildLegend()
        if self.rootfile: c.Write()