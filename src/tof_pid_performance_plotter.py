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
            "e": (11, -11),
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
                title     = "P_{track} vs 1/ #beta", 
                xlabel    = "P_{track} [GeV]",      
                ylabel    = "1/ #beta",         
                outputname= f"beta_inv_vs_p_{area}",   
                rootfile  = self.rootfile
            )

        # ------------------------------------------------------------
        # reconstructed mass 
        # ------------------------------------------------------------
        if "calc_mass" in data:
            myfunc.make_histogram_root(
                data["calc_mass"], 120, [0, 1200],
                f"Reconstructed_Mass_{area}",
                "m_{reco} [MeV]", "Entries",
                rootfile=self.rootfile
            )

        # ------------------------------------------------------------
        # momentum and β⁻¹ and resolution
        # ------------------------------------------------------------
        if "momentum" in data:
            myfunc.make_histogram_root(
                data["momentum"], 100, [0, 3.5],
                f"Track_momentum_{area}",
                "Track momentum [GeV]", "Entries",
                rootfile=self.rootfile
            )
        if "beta_inverse" in data:
            myfunc.make_histogram_root(
                data["beta_inverse"], 100, [0, 1.8],
                f"Beta_inverse_{area}",
                "1/ #beta", "Entries",
                rootfile=self.rootfile
            )
        
        if "momentum" in data and "mc_momentum" in data:
            data["momentum_reso"] = (data["momentum"] - data["mc_momentum"]) / data["mc_momentum"]
            myfunc.make_histogram_root(
                data["momentum_reso"], 100, [-0.5, 0.5],
                f"Momentum_resolution_{area}",
                "Momentum resolution ", "Entries",
                rootfile=self.rootfile
            )

        if "track_pos_phi" in data and "tof_pos_phi" in data:
            data["delta_phi"] = (data["track_pos_phi"] - data["tof_pos_phi"])*1000
            myfunc.make_histogram_root(
                data["delta_phi"], 100, [-20, 20],
                f"Delta_phi_{area}",
                "Delta phi [mrad]", "Entries", 
                rootfile=self.rootfile
            )

        if "track_pos_theta" in data and "tof_pos_theta" in data:
            data["delta_theta"] = (data["track_pos_theta"] - data["tof_pos_theta"])*1000
            myfunc.make_histogram_root(
                data["delta_theta"], 100, [-20, 20],
                f"Delta_theta_{area}",
                "Delta theta [mrad]", "Entries",
                rootfile=self.rootfile
            )

        if "momentum" in data and "mc_momentum" in data and "calc_mass" in data:
            data["momentum_reso"] = (data["momentum"] - data["mc_momentum"]) / data["mc_momentum"]
            myfunc.make_2Dhistogram_root(
                data["momentum_reso"],
                data["calc_mass"],
                100, [-0.5, 0.5], 100, [0, 1200],
                title     = "Momentum resolution vs m_{reco}",
                xlabel    = "Momentum resolution",
                ylabel    = "m_{reco} [MeV]",
                outputname= f"reso_vs_mass_{area}",
                rootfile  = self.rootfile
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
                        f"{tag.upper()}_Mass_{area}",
                        "m_{reco} [MeV]", "Entries",
                        rootfile=self.rootfile
                    )
                    # β^-1 vs p
                    myfunc.make_2Dhistogram_root(
                        data["momentum"][mask],
                        data["beta_inverse"][mask],
                        100, [0.0, 3.5], 100, [0.8, 1.8],
                        title     = f"beta_inverse_vs_p_({tag})_{area}",
                        xlabel    = "P_{track} [GeV]",
                        ylabel    = "1/ #beta",
                        outputname= f"beta_inv_vs_p_{tag}_{area}",
                        rootfile  = self.rootfile
                    )

    def plot_separation_power_vs_momentum(
        self,
        bin_centers: np.ndarray,
        sep_pi_k   : np.ndarray,
        sep_k_p    : np.ndarray,
        sep_pi_k_err  : np.ndarray,
        sep_k_p_err   : np.ndarray,
        area: str = "btof",
        p_range_min: float = 0.0,
        p_range_max: float = 1.5,
    ) -> None:
        """
        Draw two curves on the same canvas:

        * π-K  separation power
        * K-p  separation power
        With error bars.
        """

        # ─── sanitize NaN / inf ─────────────────────────────────
        mask_pi_k = np.isfinite(sep_pi_k) & np.isfinite(sep_pi_k_err)
        mask_k_p  = np.isfinite(sep_k_p)  & np.isfinite(sep_k_p_err)

        if (not mask_pi_k.any()) and (not mask_k_p.any()):
            print(f"[warn] plot_separation_power_vs_momentum: no valid points for {area}")
            return

        x_pi_k = np.ascontiguousarray(bin_centers[mask_pi_k].astype(np.float64))
        y_pi_k = np.ascontiguousarray(sep_pi_k   [mask_pi_k].astype(np.float64))
        e_pi_k = np.ascontiguousarray(sep_pi_k_err[mask_pi_k].astype(np.float64))

        x_k_p  = np.ascontiguousarray(bin_centers[mask_k_p].astype(np.float64))
        y_k_p  = np.ascontiguousarray(sep_k_p    [mask_k_p ].astype(np.float64))
        e_k_p  = np.ascontiguousarray(sep_k_p_err[mask_k_p ].astype(np.float64))

        # ─── TGraphErrors objects ───────────────────────────────
        g_pi_k = r.TGraphErrors(len(x_pi_k), x_pi_k, y_pi_k, np.zeros_like(x_pi_k), e_pi_k)
        g_k_p  = r.TGraphErrors(len(x_k_p ), x_k_p , y_k_p , np.zeros_like(x_k_p ), e_k_p )

        for g in (g_pi_k, g_k_p):
            g.SetMarkerSize(1.2)
            g.GetXaxis().SetLimits(p_range_min, p_range_max)
            g.GetYaxis().SetRangeUser(1e-3, 1e2)

        g_pi_k.SetMarkerStyle(20)
        g_k_p.SetMarkerStyle(20)
        g_k_p.SetMarkerColor(r.kRed)

        g_pi_k.SetTitle(f"pi/k Separation Power {area}")

        # ─── canvas & draw ──────────────────────────────────────
        c1 = r.TCanvas(f"c_sep_power_pi_k_{area}", " ", 800, 600)
        c1.SetLogy(True)
        c1.SetGrid()
        g_pi_k.GetXaxis().SetTitle("P_{T} [GeV]")
        g_pi_k.GetYaxis().SetTitle("Separation Power")
        g_pi_k.Draw("AP")

        c2 = r.TCanvas(f"c_sep_power_k_p_{area}", " ", 800, 600)
        c2.SetLogy(True)
        c2.SetGrid()
        g_k_p.SetTitle(f"k/p Separation Power {area}")
        g_k_p.GetXaxis().SetTitle("P_{T} [GeV]")
        g_k_p.GetYaxis().SetTitle("Separation Power")
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
        g_k .SetMarkerStyle(20); g_k.SetMarkerColor(r.kRed)
        g_p .SetMarkerStyle(20); g_p.SetMarkerColor(r.kBlue)

        g_pi.SetTitle(f"Purity vs Momentum ({area});Momentum [GeV];Purity")

        # ── canvas ──────────────────────────────────────────────
        c1 = r.TCanvas(f"c_purity_pi_{area}", "", 800, 600)
        c1.SetGrid()
        g_pi.GetXaxis().SetTitle("P_{track} [GeV]")
        g_pi.GetYaxis().SetTitle("Purity")
        g_pi.Draw("AP")

        c2 = r.TCanvas(f"c_purity_k_{area}", "", 800, 600)
        c2.SetGrid()
        g_k.SetTitle(f"purity_k_{area};Momentum [GeV];Purity")
        g_k.GetXaxis().SetTitle("P_{track} [GeV]")
        g_k.GetYaxis().SetTitle("Purity")
        g_k.Draw("AP")

        c3 = r.TCanvas(f"c_purity_p_{area}", "", 800, 600)
        c3.SetGrid()
        g_p.SetTitle(f"purity_p_{area};Momentum [GeV];Purity")
        g_p.GetXaxis().SetTitle("P_{track} [GeV]")
        g_p.GetYaxis().SetTitle("Purity")
        g_p.Draw("AP")

        if self.rootfile:
            c1.Write()
            c2.Write()
            c3.Write()