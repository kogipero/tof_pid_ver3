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
                title     = "beta_inverse_vs_p",    
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
        # momentum and β⁻¹ and resolution
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
        
        if "momentum" in data and "mc_momentum" in data:
            data["momentum_reso"] = data["momentum"] - data["mc_momentum"]
            myfunc.make_histogram_root(
                data["momentum_reso"], 100, [-0.2, 0.2],
                f"Momentum resolution_{area}; p [GeV/c];Entries",
                f"p_reso_hist_{area}", rootfile=self.rootfile
            )

        if "track_pos_phi" in data and "tof_pos_phi" in data:
            data["phi_reso"] = data["track_pos_phi"] - data["tof_pos_phi"]
            myfunc.make_histogram_root(
                data["phi_reso"], 100, [-0.2, 0.2],
                f"Phi resolution_{area}; Phi [rad];Entries",
                f"phi_reso_hist_{area}", rootfile=self.rootfile
            )

        if "track_pos_theta" in data and "tof_pos_theta" in data:
            data["theta_reso"] = data["track_pos_theta"] - data["tof_pos_theta"]
            myfunc.make_histogram_root(
                data["theta_reso"], 100, [-0.2, 0.2],
                f"Theta resolution_{area}; Theta [rad];Entries",
                f"theta_reso_hist_{area}", rootfile=self.rootfile
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
                        f"{tag.upper()}_Mass_({area});Mass [MeV];Entries",
                        f"mass_{tag}_{area}", rootfile=self.rootfile
                    )
                    # β^-1 vs p
                    myfunc.make_2Dhistogram_root(
                        data["momentum"][mask],
                        data["beta_inverse"][mask],
                        100, [0.0, 3.5], 100, [0.8, 1.8],
                        title     = f"beta_inverse_vs_p_({tag})_{area}",
                        xlabel    = "p [GeV/c]",
                        ylabel    = "beta inverse",
                        outputname= f"beta_inv_vs_p_{tag}_{area}",
                        rootfile  = self.rootfile
                    )

    def plot_separation_power_vs_momentum(
        self,
        bin_centers: np.ndarray,
        sep_pi_k   : np.ndarray,
        sep_k_p    : np.ndarray,
        area: str = "btof",
    ) -> None:
        """
        Draw two curves on the same canvas:

        * π-K  separation power
        * K-p  separation power
        """

        # ─── sanitize NaN / inf ─────────────────────────────────
        mask_pi_k = np.isfinite(sep_pi_k)
        mask_k_p  = np.isfinite(sep_k_p)

        if (not mask_pi_k.any()) and (not mask_k_p.any()):
            print(f"[warn] plot_separation_power_vs_momentum: no valid points for {area}")
            return

        x_pi_k = np.ascontiguousarray(bin_centers[mask_pi_k].astype(np.float64))
        y_pi_k = np.ascontiguousarray(sep_pi_k   [mask_pi_k].astype(np.float64))
        x_k_p  = np.ascontiguousarray(bin_centers[mask_k_p ].astype(np.float64))
        y_k_p  = np.ascontiguousarray(sep_k_p    [mask_k_p ].astype(np.float64))

        # ─── TGraph objects ─────────────────────────────────────
        g_pi_k = r.TGraph(len(x_pi_k), x_pi_k, y_pi_k)
        g_k_p  = r.TGraph(len(x_k_p ), x_k_p , y_k_p )

        for g in (g_pi_k, g_k_p):
            g.SetMarkerSize(1.2)
            g.GetXaxis().SetLimits(0.0, 3.5)
            g.GetYaxis().SetRangeUser(0.0, 1.0)

        g_pi_k.SetMarkerStyle(20)
        g_k_p.SetMarkerStyle(21)
        g_k_p.SetMarkerColor(r.kRed)

        g_pi_k.SetTitle(f"Separation Power ({area}); pT [GeV]; Separation Power")

        # ─── canvas & draw ──────────────────────────────────────
        c1 = r.TCanvas(f"c_sep_pi_k_{area}", " ", 800, 600)
        c1.SetLogy(True) 
        g_pi_k.GetYaxis().SetRangeUser(1e-3, 1e2)
        g_pi_k.GetXaxis().SetLimits(0.0, 3.5)
        g_pi_k.Draw("AP")

        c2 = r.TCanvas(f"c_sep_k_p_{area}", " ", 800, 600)
        c2.SetLogy(True)
        g_k_p.GetYaxis().SetRangeUser(1e-3, 1e2)
        g_k_p.GetXaxis().SetLimits(0.0, 3.5)
        g_k_p.Draw("AP")

        if self.rootfile:
            c1.Write()
            c2.Write()

    def plot_purity_vs_momentum(
        self, bins,
        pi_norm, pi_err_norm, pi_uniq, pi_err_uniq,
        k_norm, k_err_norm, k_uniq, k_err_uniq,
        p_norm, p_err_norm, p_uniq, p_err_uniq,
        area
    ):
        # ── safety: contiguous float64 arrays ───────────────────
        x = np.ascontiguousarray(bins.astype(np.float64))
        y_pi = np.ascontiguousarray(pi_norm.astype(np.float64))
        y_k  = np.ascontiguousarray(k_norm.astype(np.float64))
        y_p  = np.ascontiguousarray(p_norm.astype(np.float64))

        ex = np.zeros_like(x, dtype=np.float64)
        ey_pi = np.ascontiguousarray(pi_err_norm.astype(np.float64))
        ey_k  = np.ascontiguousarray(k_err_norm.astype(np.float64))
        ey_p  = np.ascontiguousarray(p_err_norm.astype(np.float64))

        # ── build graphs ────────────────────────────────────────
        g_pi = r.TGraphErrors(len(x), x, y_pi, ex, ey_pi)
        g_k  = r.TGraphErrors(len(x), x, y_k,  ex, ey_k)
        g_p  = r.TGraphErrors(len(x), x, y_p,  ex, ey_p)

        for g in (g_pi, g_k, g_p):
            g.GetXaxis().SetLimits(0.0, max(x)*1.05)
            g.GetYaxis().SetRangeUser(0.0, 1.05)

        g_pi.SetMarkerStyle(20)
        g_k .SetMarkerStyle(21); g_k.SetMarkerColor(r.kRed)
        g_p .SetMarkerStyle(22); g_p.SetMarkerColor(r.kBlue)

        g_pi.SetTitle(f"Purity vs Momentum ({area});Momentum [GeV];Purity")

        # ── canvas ──────────────────────────────────────────────
        c1 = r.TCanvas(f"c_purity_pi_{area}", "", 800, 600)
        c1.SetGrid()
        g_pi.Draw("AP")

        c2 = r.TCanvas(f"c_purity_k_{area}", "", 800, 600)
        c2.SetGrid()
        g_k.SetTitle(f"purity_k_{area};Momentum [GeV];Purity")
        g_k.Draw("AP")

        c3 = r.TCanvas(f"c_purity_p_{area}", "", 800, 600)
        c3.SetGrid()
        g_p.SetTitle(f"purity_p_{area};Momentum [GeV];Purity")
        g_p.Draw("AP")

        if self.rootfile:
            c1.Write()
            c2.Write()
            c3.Write()