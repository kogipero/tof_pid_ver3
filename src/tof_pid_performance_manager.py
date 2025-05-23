from __future__ import annotations
from collections.abc import Iterable
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import ROOT as r  
import uproot  
from tof_pid_performance_plotter import TOFPIDPerformancePlotter
import itertools

def _extract_pdg(val: Any) -> int:
    """Return integer PDG code from raw field (Series / MatchedHitInfo / int)."""
    if isinstance(val, pd.Series):
        return int(val["mc_pdg"])
    if hasattr(val, "mc_pdg"):
        return int(val.mc_pdg)
    return int(val)

class ToFPIDPerformanceManager:
    r"""Manage PID performance calculations for TOF tracks."""

    def __init__(
        self,
        dis_file: uproot.TTree | None,
        branch: Dict[str, Any] | None,
        name: str,
        rootfile: r.TFile | None = None,
    ) -> None:
        """Initialize the PID performance manager.

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
            Total track momentum (GeV).
        pt : np.ndarray
            Transverse track momentum (GeV).
        """

        if not isinstance(tof_and_track_matched_pd, pd.DataFrame):
            if not isinstance(tof_and_track_matched_pd, Iterable):
                tof_and_track_matched_pd = [tof_and_track_matched_pd]
            tof_and_track_matched_pd = self._matchedhit_to_dataframe(
                tof_and_track_matched_pd
            )

        if "mc_pdg_val" not in tof_and_track_matched_pd.columns:
            tof_and_track_matched_pd["mc_pdg_val"] = (
                tof_and_track_matched_pd["mc_pdg"].apply(_extract_pdg).astype(int)
            )

        beta = (
            tof_and_track_matched_pd["track_pathlength"]
            / tof_and_track_matched_pd["tof_time"]
        )
        beta_c = beta / 299.792458  
        beta_inv = 1.0 / beta_c

        momentum = tof_and_track_matched_pd["track_p"]
        calc_mass = 1000.0 * momentum * np.sqrt(1.0 - beta_c**2) / beta_c #MeV

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

        # Overall Purity each particle
        masks = {
            "pi": (pdg_np == 211) | (pdg_np == -211),
            "k": (pdg_np == 321) | (pdg_np == -321),
            "p": (pdg_np == 2212) | (pdg_np == -2212),
            "e": (pdg_np == 11) | (pdg_np == -11),
        }
        masses_true = {"pi": 139.57, "k": 493.677, "p": 938.272, "e": 0.511}
        mergins = {"pi": MERGIN_PI, "k": MERGIN_K, "p": MERGIN_P, "e": 0.1}
        mergins_large = {
            "pi": LARGE_MERGIN_PI,
            "k": LARGE_MERGIN_K,
            "p": LARGE_MERGIN_P,
            "e": 0.1,
        }

        for key in ("pi", "k", "p", "e"):
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

        return calc_mass_np, pdg_np, p_np, pt_np

    def process_separation_power_vs_momentum(
        self,
        tof_calc_mass: np.ndarray,
        tof_pdg: np.ndarray,
        track_momentums_on_tof: np.ndarray,
        track_momentums_transverse_on_tof: np.ndarray,
        *,
        area: str = "btof",
        nbins: int = 30,
        momentum_range: tuple[float, float] = (0.0, 1.5),
        plot_verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return
        -------
        bin_centers : np.ndarray
        sep_pi_k    : np.ndarray  (pi-K separation power)
        sep_k_p     : np.ndarray  (K-p separation power)
        err_pi_k    : np.ndarray  (error of pi-K separation)
        err_k_p     : np.ndarray  (error of K-p separation)
        """

        pi_mask = (tof_pdg ==  211) | (tof_pdg == -211)
        k_mask  = (tof_pdg ==  321) | (tof_pdg == -321)
        p_mask  = (tof_pdg == 2212) | (tof_pdg == -2212)

        p_bins      = np.linspace(*momentum_range, nbins + 1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])

        sep_pi_k: List[float | None] = []
        sep_k_p : List[float | None] = []
        err_pi_k: List[float | None] = []
        err_k_p : List[float | None] = []

        def _fit_gauss(vals: np.ndarray, mu_guess: float, p_low: float, p_high: float) -> Tuple[float, float, float, float]:
            idx = next(self._id_counter)
            title = f"PID separation power ({p_low:.2f} < p < {p_high:.2f})"
            h = r.TH1D(f"reco_mass_{idx}_{p_low:.2f}_{p_high:.2f}", title, 120, 0, 1200)
            for v in vals:
                h.Fill(float(v))

            h.Write()
            if h.GetEntries() < 5:
                h.Delete()
                return 0.0, 0.0, 0.0, 0.0

            f = r.TF1(f"f_sep_{idx}", "[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1200)
            f.SetParameters(h.GetMaximum()*1.2, mu_guess, 20)
            h.Fit(f, "Q0")
            f.Write()
            mu = f.GetParameter(1)
            sigma = abs(f.GetParameter(2))
            mu_err = f.GetParError(1)
            sigma_err = f.GetParError(2)
            return mu, sigma, mu_err, sigma_err

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
                mu_pi, sigma_pi, mu_err_pi, sigma_err_pi = _fit_gauss(pi_vals, 139.0, p_low, p_high)
                mu_k, sigma_k, mu_err_k, sigma_err_k = _fit_gauss(k_vals, 494.0, p_low, p_high)
                if sigma_pi > 1e-6 and sigma_k > 1e-6:
                    delta_mu = mu_pi - mu_k
                    sigma2 = 0.5 * (sigma_pi**2 + sigma_k**2)
                    S = abs(delta_mu) / np.sqrt(sigma2)
                    delta_S_sq = (mu_err_pi**2 + mu_err_k**2) / sigma2 + \
                                    (delta_mu**2 / (4 * sigma2**3)) * ((sigma_pi**2) * sigma_err_pi**2 + (sigma_k**2) * sigma_err_k**2)
                    S_err = np.sqrt(delta_S_sq)
                    sep_pi_k.append(S)
                    err_pi_k.append(S_err)
                else:
                    sep_pi_k.append(None)
                    err_pi_k.append(None)
            else:
                sep_pi_k.append(None)
                err_pi_k.append(None)

            # K–p separation ------------------------------------
            if len(k_vals) >= 5 and len(p_vals) >= 5:
                mu_k , sigma_k, mu_err_k, sigma_err_k = _fit_gauss(k_vals, 494.0, p_low, p_high)
                mu_p , sigma_p, mu_err_p, sigma_err_p = _fit_gauss(p_vals, 938.0, p_low, p_high)
                if sigma_k > 1e-6 and sigma_p > 1e-6:
                    delta_mu = mu_k - mu_p
                    sigma2 = 0.5 * (sigma_k**2 + sigma_p**2)
                    S = abs(delta_mu) / np.sqrt(sigma2)
                    delta_S_sq = (mu_err_k**2 + mu_err_p**2) / sigma2 + \
                                    (delta_mu**2 / (4 * sigma2**3)) * ((sigma_k**2) * sigma_err_k**2 + (sigma_p**2) * sigma_err_p**2)
                    S_err = np.sqrt(delta_S_sq)
                    sep_k_p.append(S)
                    err_k_p.append(S_err)
                else:
                    sep_k_p.append(None)
                    err_k_p.append(None)
            else:
                sep_k_p.append(None)
                err_k_p.append(None)

        sep_pi_k_arr = np.asarray(sep_pi_k, dtype=object)
        sep_k_p_arr  = np.asarray(sep_k_p , dtype=object)
        err_pi_k_arr = np.asarray(err_pi_k, dtype=object)
        err_k_p_arr  = np.asarray(err_k_p , dtype=object)

        valid_mask   = (sep_pi_k_arr != None) & (sep_k_p_arr != None)

        centers_clean = bin_centers[valid_mask].astype(float)
        pi_k_clean    = sep_pi_k_arr[valid_mask].astype(float)
        k_p_clean     = sep_k_p_arr [valid_mask].astype(float)
        pi_k_err_clean = err_pi_k_arr[valid_mask].astype(float)
        k_p_err_clean = err_k_p_arr[valid_mask].astype(float)

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_separation_power_vs_momentum(
                centers_clean, pi_k_clean, k_p_clean,
                pi_k_err_clean, k_p_err_clean,
                area=area
            )

        return centers_clean, pi_k_clean, k_p_clean, pi_k_err_clean, k_p_err_clean


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

        eff_norm: Dict[str, List[float]] = {k: [] for k in masks}
        err_norm: Dict[str, List[float]] = {k: [] for k in masks}

        for p_low, p_high in zip(p_bins[:-1], p_bins[1:]):
            sel = (track_momentums_on_btof >= p_low) & (track_momentums_on_btof < p_high)
            for key in masks:
                vals = btof_calc_mass[sel & masks[key]]
                eff, err = _bin_eff(vals, masses_true[key], mergins[key])
                eff_norm[key].append(eff)
                err_norm[key].append(err)

        for key in eff_norm:
            eff_norm[key] = np.array(eff_norm[key], dtype=float)
            err_norm[key] = np.array(err_norm[key], dtype=float)

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_purity_vs_momentum(
                bin_centers,
                eff_norm["pi"], err_norm["pi"], eff_norm["pi"], err_norm["pi"],
                eff_norm["k"], err_norm["k"], eff_norm["k"], err_norm["k"],
                eff_norm["p"], err_norm["p"], eff_norm["p"], err_norm["p"],
                area=area,
            )

        print("[Purity] π:", eff_norm["pi"])
        print("[Purity] K:", eff_norm["k"])
        print("[Purity] p:", eff_norm["p"])