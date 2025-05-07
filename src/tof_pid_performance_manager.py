from __future__ import annotations
from collections.abc import Iterable
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import ROOT as r  
import uproot  
from tof_pid_performance_plotter import TOFPIDPerformancePlotter
import itertools

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _extract_pdg(val: Any) -> int:
    """Return integer PDG code from raw field (Series / MatchedHitInfo / int)."""
    if isinstance(val, pd.Series):
        return int(val["mc_pdg"])
    if hasattr(val, "mc_pdg"):
        return int(val.mc_pdg)
    return int(val)


# -----------------------------------------------------------------------------
# Main manager class
# -----------------------------------------------------------------------------

class ToFPIDPerformanceManager:
    r"""Manage PID performance calculations for TOF tracks."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        dis_file: uproot.TTree | None,
        branch: Dict[str, Any] | None,
        name: str,
        rootfile: r.TFile | None = None,
    ) -> None:
        """Create a manager.

        Parameters
        ----------
        dis_file : uproot.TTree | None
            ROOT/UPROOT tree containing additional information (optional).
        branch : Dict | None
            Not used directly here but preserved for compatibility.
        name : str
            Tag used by the plotter and ROOT objects.
        rootfile : ROOT.TFile | None, default None
            Optional file to which histograms will be written.
        """
        self.name = name
        self.rootfile = rootfile
        self.branch = branch
        self.dis_file = dis_file
        self.tof_pid_performance_plotter = TOFPIDPerformancePlotter(rootfile, name)
        self._id_counter = itertools.count() 

    # ------------------------------------------------------------------
    # Internal converter: MatchedHitInfo → DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def _matchedhit_to_dataframe(matched_hits: Iterable[Any]) -> pd.DataFrame:
        """Convert iterable of MatchedHitInfo to a pandas.DataFrame."""
        records = [
            dict(
                tof_time=hit.tof_time,
                track_p=hit.track_p,
                track_pt=hit.track_pt,
                track_pathlength=hit.track_pathlength,
                mc_pdg=hit.mc_pdg,
            )
            for hit in matched_hits
        ]
        return pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------
    # 1) β, calculated mass, efficiency etc.
    # ------------------------------------------------------------------
    def process_pid_performance_plot(
        self,
        tof_and_track_matched_pd: pd.DataFrame | Iterable[Any],
        area: str = "btof",
        MERGIN_PI: float = 100.0,
        MERGIN_K: float = 100.0,
        MERGIN_P: float = 100.0,
        LARGE_MERGIN_PI: float = 200.0,
        LARGE_MERGIN_K: float = 200.0,
        LARGE_MERGIN_P: float = 200.0,
        MOMENTUM_RANGE: float = 2.5,
        output_txt_name: str = "pid_result.txt",
        plot_verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute PID efficiency and return core arrays.

        Parameters
        ----------
        btof_and_track_matched_pd : DataFrame | Iterable[MatchedHitInfo]
            BTOF ↔ track matching result. If not a DataFrame it will be converted.
        plot_verbose : bool, default False
            When True create diagnostic ROOT plots.

        Returns
        -------
        calc_mass : np.ndarray
            Calculated mass for each hit (MeV).
        pdg : np.ndarray[int]
            PDG codes for each hit.
        p : np.ndarray
            Total track momentum (GeV/c).
        pt : np.ndarray
            Transverse track momentum (GeV/c).
        """

        # --------------------------------------------------------------
        # Auto‑convert if necessary
        # --------------------------------------------------------------
        if not isinstance(tof_and_track_matched_pd, pd.DataFrame):
            if not isinstance(tof_and_track_matched_pd, Iterable):
                tof_and_track_matched_pd = [tof_and_track_matched_pd]
            tof_and_track_matched_pd = self._matchedhit_to_dataframe(
                tof_and_track_matched_pd
            )

        # --------------------------------------------------------------
        # Ensure int PDG column exists
        # --------------------------------------------------------------
        if "mc_pdg_val" not in tof_and_track_matched_pd.columns:
            tof_and_track_matched_pd["mc_pdg_val"] = (
                tof_and_track_matched_pd["mc_pdg"].apply(_extract_pdg).astype(int)
            )

        # --------------------------------------------------------------
        # Vectorised calculations
        # --------------------------------------------------------------
        beta = (
            tof_and_track_matched_pd["track_pathlength"]
            / tof_and_track_matched_pd["tof_time"]
        )
        beta_c = beta / 299.792458  # c in mm/ns
        beta_inv = 1.0 / beta_c

        momentum = tof_and_track_matched_pd["track_p"]
        calc_mass = 1000.0 * momentum * np.sqrt(1.0 - beta_c**2) / beta_c

        # to numpy
        calc_mass_np = calc_mass.to_numpy()
        pdg_np = tof_and_track_matched_pd["mc_pdg_val"].to_numpy(dtype=int)
        p_np = momentum.to_numpy()
        pt_np = tof_and_track_matched_pd["track_pt"].to_numpy()
        beta_inv_np = beta_inv.to_numpy()
        tof_time_np = tof_and_track_matched_pd["tof_time"].to_numpy()
        mc_momentum_np = tof_and_track_matched_pd["mc_momentum"].to_numpy()
        track_pos_phi_np = tof_and_track_matched_pd["track_pos_phi"].to_numpy()
        track_pos_theta_np = tof_and_track_matched_pd["track_pos_theta"].to_numpy()
        tof_pos_phi_np = tof_and_track_matched_pd["tof_pos_phi"].to_numpy()
        tof_pos_theta_np = tof_and_track_matched_pd["tof_pos_theta"].to_numpy()

        # --------------------------------------------------------------
        # Plot (optional)
        # --------------------------------------------------------------
        data = {
            "particle_types": ["all"],       
            "time_all": tof_time_np,           
            "momentum": p_np,                
            "beta_inverse": beta_inv_np,     
            "calc_mass": calc_mass_np,
            "pdg": pdg_np,
            "track_pos_phi": track_pos_phi_np,
            "track_pos_theta": track_pos_theta_np,
            "tof_pos_phi": tof_pos_phi_np,
            "tof_pos_theta": tof_pos_theta_np,
            "mc_momentum": mc_momentum_np,      
        }

        self.tof_pid_performance_plotter.plot_pid_performance(
            data,
            area=area,
        )

        # --------------------------------------------------------------
        # Efficiency calculation (π/K/p, normal & large window)
        # --------------------------------------------------------------
        masks = {
            "pi": (pdg_np == 211) | (pdg_np == -211),
            "k": (pdg_np == 321) | (pdg_np == -321),
            "p": (pdg_np == 2212) | (pdg_np == -2212),
        }
        masses_true = {"pi": 139.57, "k": 493.677, "p": 938.272}
        mergins = {"pi": MERGIN_PI, "k": MERGIN_K, "p": MERGIN_P}
        mergins_large = {
            "pi": LARGE_MERGIN_PI,
            "k": LARGE_MERGIN_K,
            "p": LARGE_MERGIN_P,
        }

        for key in ("pi", "k", "p"):
            mask = masks[key]
            true_mass = masses_true[key]
            n_true = mask.sum()
            if n_true == 0:
                print(f"[PID] {key} : no statistics!")
                continue

            diff = np.abs(calc_mass_np[mask] - true_mass)
            eff = (diff < mergins[key]).sum() / n_true
            eff_large = (diff < mergins_large[key]).sum() / n_true
            print(
                f"[PID] {key} Eff (±{mergins[key]:g} [MeV]) : {100 * eff:.3f}% | ±{mergins_large[key]:g} [MeV]: {100 * eff_large:.3f}% {area}"
            )

        # --------------------------------------------------------------
        # Return numpy arrays for further processing
        # --------------------------------------------------------------
        return calc_mass_np, pdg_np, p_np, pt_np

    # ------------------------------------------------------------------
    # 2) Separation power π/K and K/p vs momentum
    # ------------------------------------------------------------------
    def process_separation_power_vs_momentum(
        self,
        tof_calc_mass: np.ndarray,
        tof_pdg: np.ndarray,
        track_momentums_on_tof: np.ndarray,
        track_momentums_transverse_on_tof: np.ndarray,
        *,
        area: str = "btof",
        nbins: int = 35,
        momentum_range: tuple[float, float] = (0.0, 3.5),
        plot_verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return
        -------
        bin_centers : np.ndarray
        sep_pi_k    : np.ndarray  (π-K separation power)
        sep_k_p     : np.ndarray  (K-p separation power)
        Invalid / low-stat bins are automatically dropped.
        """

        # ─── particle masks ──────────────────────────────────────
        pi_mask = (tof_pdg ==  211) | (tof_pdg == -211)
        k_mask  = (tof_pdg ==  321) | (tof_pdg == -321)
        p_mask  = (tof_pdg == 2212) | (tof_pdg == -2212)

        # ─── pT bins ─────────────────────────────────────────────
        p_bins      = np.linspace(*momentum_range, nbins + 1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])

        sep_pi_k: List[float | None] = []
        sep_k_p : List[float | None] = []

        # -- helper: Gaussian fit with unique name ----------------
        def _fit_gauss(vals: np.ndarray, mu_guess: float) -> Tuple[float, float]:
            idx = next(self._id_counter)
            h   = r.TH1F(f"h_sep_{idx}", "", 100, 0, 1000)
            for v in vals:
                h.Fill(float(v))

            if h.GetEntries() < 5:
                h.Delete()
                return 0.0, 0.0

            f = r.TF1(f"f_sep_{idx}", "[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f.SetParameters(h.GetMaximum()*1.2, mu_guess, 20)
            h.Fit(f, "Q0")
            mu, sigma = f.GetParameter(1), abs(f.GetParameter(2))

            h.Delete(); f.Delete()
            return mu, sigma

        # ─── loop over bins ──────────────────────────────────────
        for p_low, p_high in zip(p_bins[:-1], p_bins[1:]):
            sel_bin = (
                (track_momentums_transverse_on_tof >= p_low) &
                (track_momentums_transverse_on_tof <  p_high)
            )

            pi_vals = tof_calc_mass[sel_bin & pi_mask]
            k_vals  = tof_calc_mass[sel_bin & k_mask]
            p_vals  = tof_calc_mass[sel_bin & p_mask]

            # π–K separation ------------------------------------
            if len(pi_vals) >= 5 and len(k_vals) >= 5:
                mu_pi, sigma_pi = _fit_gauss(pi_vals, 140.0)
                mu_k , sigma_k  = _fit_gauss(k_vals , 494.0)
                if sigma_pi > 1e-6 and sigma_k > 1e-6:
                    sep_val = abs(mu_pi - mu_k) / np.sqrt(0.5*(sigma_pi**2 + sigma_k**2))
                    sep_pi_k.append(sep_val)
                else:
                    sep_pi_k.append(None)
            else:
                sep_pi_k.append(None)

            # K–p separation ------------------------------------
            if len(k_vals) >= 5 and len(p_vals) >= 5:
                mu_k , sigma_k = _fit_gauss(k_vals, 494.0)
                mu_p , sigma_p = _fit_gauss(p_vals, 938.0)
                if sigma_k > 1e-6 and sigma_p > 1e-6:
                    sep_val = abs(mu_k - mu_p) / np.sqrt(0.5*(sigma_k**2 + sigma_p**2))
                    sep_k_p.append(sep_val)
                else:
                    sep_k_p.append(None)
            else:
                sep_k_p.append(None)

        # ─── remove None bins ───────────────────────────────────
        sep_pi_k_arr = np.asarray(sep_pi_k, dtype=object)
        sep_k_p_arr  = np.asarray(sep_k_p , dtype=object)
        valid_mask   = (sep_pi_k_arr != None) & (sep_k_p_arr != None)

        centers_clean = bin_centers[valid_mask].astype(float)
        pi_k_clean    = sep_pi_k_arr[valid_mask].astype(float)
        k_p_clean     = sep_k_p_arr [valid_mask].astype(float)

        # ─── plotting (optional) ────────────────────────────────
        if plot_verbose:
            self.tof_pid_performance_plotter.plot_separation_power_vs_momentum(
                centers_clean, pi_k_clean, k_p_clean, area=area
            )

    # def process_separation_power_vs_momentum(
    #     self,
    #     tof_calc_mass: np.ndarray,
    #     tof_pdg: np.ndarray,
    #     track_momentums_on_tof: np.ndarray,
    #     track_momentums_transverse_on_tof: np.ndarray,
    #     area: str = "btof",
    #     nbins: int = 35,
    #     momentum_range: tuple[float, float] = (0.0, 3.5),
    #     plot_verbose: bool = False,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Gaussian separation power (K / p) as a function of pT.
    #     Returns (bin_centers, separation_power) where invalid bins are dropped.
    #     """
    #     # ─── species masks ─────────────────────────────────────
    #     pi_mask = (tof_pdg ==  211) | (tof_pdg == -211)
    #     k_mask  = (tof_pdg ==  321) | (tof_pdg == -321)
    #     p_mask  = (tof_pdg == 2212) | (tof_pdg == -2212)

    #     # ─── pT binning ───────────────────────────────────────
    #     p_bins      = np.linspace(*momentum_range, nbins + 1)
    #     bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
    #     sep_k_p: List[float | None] = []

    #     # -- helper: Gaussian fit with unique name via counter --
    #     def _fit_gauss(vals: np.ndarray, mu_guess: float) -> Tuple[float, float]:
    #         idx = next(self._id_counter) 
    #         h   = r.TH1F(f"h_tmp{idx}", "", 100, 0, 1000)
    #         for v in vals:
    #             h.Fill(float(v))

    #         if h.GetEntries() < 3:
    #             h.Delete()
    #             return 0.0, 0.0

    #         f = r.TF1(f"f_tmp{idx}", "[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
    #         f.SetParameters(h.GetMaximum()*1.1, mu_guess, 20)
    #         h.Fit(f, "Q0")                       # Quiet, no draw

    #         mu, sigma = f.GetParameter(1), abs(f.GetParameter(2))

    #         h.Delete(); f.Delete()               # tidy-up
    #         return mu, sigma

    #     # ─── loop over pT bins ─────────────────────────────────
    #     for p_low, p_high in zip(p_bins[:-1], p_bins[1:]):
    #         sel_bin = (
    #             (track_momentums_transverse_on_tof >= p_low) &
    #             (track_momentums_transverse_on_tof <  p_high)
    #         )

    #         k_vals = tof_calc_mass[sel_bin & k_mask]
    #         p_vals = tof_calc_mass[sel_bin & p_mask]

    #         if len(k_vals) < 5 or len(p_vals) < 5:
    #             sep_k_p.append(None)
    #             continue

    #         mu_k, sigma_k = _fit_gauss(k_vals, 494.0)
    #         mu_p, sigma_p = _fit_gauss(p_vals, 938.0)

    #         if sigma_k < 1e-6 or sigma_p < 1e-6:
    #             sep_k_p.append(None)
    #             continue

    #         sep_val = abs(mu_k - mu_p) / np.sqrt(0.5 * (sigma_k**2 + sigma_p**2))
    #         sep_k_p.append(sep_val)

    #     # ─── clean up None bins ────────────────────────────────
    #     sep_arr   = np.array(sep_k_p, dtype=object)
    #     valid     = sep_arr != None
    #     centers   = bin_centers[valid].astype(float)
    #     sep_clean = sep_arr[valid].astype(float)

    #     # ─── optional plotting ────────────────────────────────
    #     if plot_verbose:
    #         self.tof_pid_performance_plotter.plot_separation_power_vs_momentum(
    #             centers,
    #             sep_clean,
    #             area=area
    #         )


    # ------------------------------------------------------------------
    # 3) Purity vs momentum (efficiency variant)
    # ------------------------------------------------------------------
    def process_purity_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        area: str = "btof",
        nbins: int = 35,
        momentum_range: tuple[float, float] = (0.0, 3.5),
        MERGIN_PI: float = 100.0,
        MERGIN_K: float = 100.0,
        MERGIN_P: float = 100.0,
        plot_verbose: bool = False,
    ) -> None:
        """Plot (and print) purity/efficiency vs momentum for π/K/p."""
        # mask by particle
        masks = {
            "pi": (btof_pdg == 211) | (btof_pdg == -211),
            "k": (btof_pdg == 321) | (btof_pdg == -321),
            "p": (btof_pdg == 2212) | (btof_pdg == -2212),
        }
        masses_true = {"pi": 139.57039, "k": 493.677, "p": 938.272}
        mergins = {"pi": MERGIN_PI, "k": MERGIN_K, "p": MERGIN_P}

        p_bins = np.linspace(momentum_range[0], momentum_range[1], nbins + 1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:]).astype(float)

        def _bin_eff(vals: np.ndarray, mass0: float, mergin: float):
            if len(vals) == 0:
                return 0.0, 0.0
            correct = np.sum(np.abs(vals - mass0) < mergin)
            eff = correct / len(vals)
            err = np.sqrt(eff * (1.0 - eff) / len(vals))
            return eff, err

        # containers
        eff_norm: Dict[str, List[float]] = {k: [] for k in masks}
        err_norm: Dict[str, List[float]] = {k: [] for k in masks}

        # per bin loop
        for p_low, p_high in zip(p_bins[:-1], p_bins[1:]):
            sel = (track_momentums_on_btof >= p_low) & (track_momentums_on_btof < p_high)
            for key in masks:
                vals = btof_calc_mass[sel & masks[key]]
                eff, err = _bin_eff(vals, masses_true[key], mergins[key])
                eff_norm[key].append(eff)
                err_norm[key].append(err)

        # to numpy
        for key in eff_norm:
            eff_norm[key] = np.array(eff_norm[key], dtype=float)
            err_norm[key] = np.array(err_norm[key], dtype=float)

        # plot
        if plot_verbose:
            self.tof_pid_performance_plotter.plot_purity_vs_momentum(
                bin_centers,
                eff_norm["pi"], err_norm["pi"], eff_norm["pi"], err_norm["pi"],
                eff_norm["k"], err_norm["k"], eff_norm["k"], err_norm["k"],
                eff_norm["p"], err_norm["p"], eff_norm["p"], err_norm["p"],
                area=area,
            )

        # log
        print("[Purity] π:", eff_norm["pi"])
        print("[Purity] K:", eff_norm["k"])
        print("[Purity] p:", eff_norm["p"])