import numpy as np
import pandas as pd
import awkward as ak
import uproot
import ROOT as r
import numbers
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
from tqdm.auto import tqdm

import helper_functions as myfunc
from tof_analyzer import TOFHitInfo
from mc_analyzer import MCInfo
from matching_mc_and_tof_plotter import MatchingMCAndTOFPlotter

@dataclass
class MatchedHitInfo:
    """Data class to store matched hit information."""
    df: pd.DataFrame
    ak_array: ak.Array

class MatchingMCAndTOF:
    """MC and TOF matching class."""
    def __init__(
        self,
        branch: Dict[str, Any],
        version: str,
        rootfile: r.TFile,
        name: str,
        dis_file: uproot.TTree
    ):
        self.branch = branch
        self.version = version
        self.rootfile = rootfile
        self.name = name
        self.dis_file = dis_file
        self.plotter = MatchingMCAndTOFPlotter(rootfile, name)

        tof_br = branch['tof']
        self.btof_rec_x_branch = tof_br['barrel']['rec_hits_branch'][1]
        self.etof_rec_x_branch = tof_br['endcap']['rec_hits_branch'][1]

        self.btof_raw_hit_mc_associaction_branch = (
            tof_br['barrel']['mc_associations_ver1_24_2_branch']
            if version=='1.24.2'
            else tof_br['barrel']['mc_associations_branch']
        )

        self.etof_raw_hit_mc_associaction_branch = (
            tof_br['endcap']['mc_associations_ver1_24_2_branch']
            if version=='1.24.2'
            else tof_br['endcap']['mc_associations_branch']
        )

    def matching_mc_and_tof(
        self,
        mc_info: MCInfo,
        btof_info: TOFHitInfo,
        etof_info: TOFHitInfo,
        selected_events: int,
        verbose: bool=False,
        plot_verbose: bool=False
    ) -> Tuple[MatchedHitInfo, MatchedHitInfo]:
        """
        1) Associate MC particles with TOF hits.
        2) Filter stable particles and vertex cut. (generator_status==1, charge!=0, |vertex_z|<5)
        3) Filter reconstructed hits.
        """
        # ── 1) Associate MC particles with truth TOF hits ──
        b_raw = self._build_hit_info(mc_info, btof_info,
                                     self.btof_raw_hit_mc_associaction_branch,
                                     selected_events, area="btof_raw")
        e_raw = self._build_hit_info(mc_info, etof_info,
                                     self.etof_raw_hit_mc_associaction_branch,
                                     selected_events, area="etof_raw")

        if plot_verbose:
            self._plot_matched_hits(b_raw, area="btof_raw")
            self._plot_matched_hits(e_raw, area="etof_raw")

        # ── 2) stable-particle filter ──
        b_stable_df, e_stable_df = self.filtered_stable_particle_hit_and_generated_point(
            b_raw.df, e_raw.df, plot_verbose=plot_verbose
        )
        b_stable = MatchedHitInfo(df=b_stable_df,
                                  ak_array=ak.Array(b_stable_df.to_dict("list")))
        e_stable = MatchedHitInfo(df=e_stable_df,
                                  ak_array=ak.Array(e_stable_df.to_dict("list")))

        if plot_verbose:
            self._plot_matched_hits(b_stable, area="btof_stable")
            self._plot_matched_hits(e_stable, area="etof_stable")

        # ── 3) Reconstructed-only hits or conversion reconstructed hits ──
        if self.version == '1.24.2':
            b_reco_df, e_reco_df = self.replace_tof_with_reconstructed(b_stable_df, e_stable_df, plot_verbose=plot_verbose)
        else:
            b_reco_df, e_reco_df = self.isReconstructedHit(b_stable_df, e_stable_df, plot_verbose=plot_verbose)
        
        b_reco = MatchedHitInfo(df=b_reco_df,
                                ak_array=ak.Array(b_reco_df.to_dict("list")))
        e_reco = MatchedHitInfo(df=e_reco_df,
                                ak_array=ak.Array(e_reco_df.to_dict("list")))

        if plot_verbose:
            self._plot_matched_hits(b_reco, area="btof_reco")
            self._plot_matched_hits(e_reco, area="etof_reco")

        return b_reco, e_reco

    def _build_hit_info(
        self,
        mc: MCInfo,
        tof: TOFHitInfo,
        assoc_branch: list[str],
        n_evt: int,
        area: str
    ) -> MatchedHitInfo:
        """ return matched hit information """
        # MC assoc_branch[0] is associated with the TOF hits index
        mc_index_awk = self.dis_file[assoc_branch[0]].array(library="ak")[:n_evt]
        rows = []
        for ev in tqdm(range(n_evt), desc=f"{area} match", unit="evt"):
            sel = mc_index_awk[ev] >= 0
            if not ak.any(sel): continue

            mc_sel = mc_index_awk[ev][sel]
            # TOF info
            hx, hy, hz = tof.pos_x[ev][sel], tof.pos_y[ev][sel], tof.pos_z[ev][sel]
            htime     = tof.time[ev][sel]
            hphi, htheta, hr = tof.phi[ev][sel], tof.theta[ev][sel], tof.r[ev][sel]
            # MC info
            mpdg = mc.pdg[ev][mc_sel]
            mstat= mc.generator_status[ev][mc_sel]
            mchg = mc.charge[ev][mc_sel]
            mvx, mvy, mvz = mc.vertex_x[ev][mc_sel], mc.vertex_y[ev][mc_sel], mc.vertex_z[ev][mc_sel]
            mpx, mpy, mpz = mc.px[ev][mc_sel], mc.py[ev][mc_sel], mc.pz[ev][mc_sel]
            mp   = mc.p[ev][mc_sel]
            mphi, mtheta  = mc.phi[ev][mc_sel], mc.theta[ev][mc_sel]

            df_ev = pd.DataFrame({
                "event":               int(ev),
                "mc_index":            ak.to_numpy(mc_sel),
                "mc_pdg":              ak.to_numpy(mpdg),
                "mc_generator_status": ak.to_numpy(mstat),
                "mc_charge":           ak.to_numpy(mchg),
                "mc_vertex_x":         ak.to_numpy(mvx),
                "mc_vertex_y":         ak.to_numpy(mvy),
                "mc_vertex_z":         ak.to_numpy(mvz),
                "mc_momentum_x":       ak.to_numpy(mpx),
                "mc_momentum_y":       ak.to_numpy(mpy),
                "mc_momentum_z":       ak.to_numpy(mpz),
                "mc_momentum":         ak.to_numpy(mp),
                "mc_momentum_phi":     ak.to_numpy(mphi),
                "mc_momentum_theta":   ak.to_numpy(mtheta),
                "tof_time":            ak.to_numpy(htime),
                "tof_pos_x":           ak.to_numpy(hx),
                "tof_pos_y":           ak.to_numpy(hy),
                "tof_pos_z":           ak.to_numpy(hz),
                "tof_pos_phi":         ak.to_numpy(hphi),
                "tof_pos_theta":       ak.to_numpy(htheta),
                "tof_pos_r":           ak.to_numpy(hr),
            })
            rows.append(df_ev)

        df_all = pd.concat(rows, ignore_index=True)
        out_csv = f"./out/{self.name}/{area}_hit_info.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"[{area}] saved → {out_csv}")

        return MatchedHitInfo(df=df_all, ak_array=ak.Array(df_all.to_dict("list")))

    def _plot_matched_hits(self, matched: MatchedHitInfo, area: str) -> None:
        """ draw matched hit information """
        df = matched.df
        print(f"Plotting {area} matched hits")
        configs = [
            (df["tof_pos_x"],   [-1000,1000], 'x [mm]',     'hit_x'),
            (df["tof_pos_y"],   [-1000,1000], 'y [mm]',     'hit_y'),
            (df["tof_pos_z"],   [-2000,2000], 'z [mm]',     'hit_z'),
            (df["tof_time"],    [0,100],      'time [ns]',  'hit_time'),
            (df["tof_pos_phi"], [-3.2,3.2],   'phi [rad]',  'hit_phi'),
            (df["tof_pos_theta"],[0,3.2],     'theta [rad]','hit_theta'),
            (df["tof_pos_r"],   [0,1000],     'r [mm]',     'hit_r'),
            (df["mc_momentum_x"],[-20,20],    'px [GeV/c]', 'mc_px'),
            (df["mc_momentum_y"],[-20,20],    'py [GeV/c]', 'mc_py'),
            (df["mc_momentum_z"],[-20,20],    'pz [GeV/c]', 'mc_pz'),
            (df["mc_momentum"], [0,20],       'p [GeV/c]',  'mc_p'),
            (df["mc_pdg"],      [-500,500],   'PDG code',   'mc_pdg'),
            (df["mc_charge"],   [-2,2],       'charge',     'mc_charge'),
        ]
        for data, hr, xl, nm in configs:
            myfunc.make_histogram_root(
                data, nbins=100, hist_range=hr,
                title=f"{area}_{nm}", xlabel=xl, ylabel="Entries",
                outputname=f"{self.name}/{area}_{nm}",
                rootfile=self.rootfile
            )
        print(f"Done plotting {area}")

    def plot_replacement_difference(
        self,
        df_truth: pd.DataFrame,
        df_replaced: pd.DataFrame,
        detector_label: str,
        root_dir: str | None = None,
        tol: float = 1e-1,                       
    ):
        """
        Robust difference plot: rows are matched by (event, tof_pos_x)
        within ±tol.  Unmatched truth hits are ignored.
        """
        # Return values want to covert pd.DataFrame ,so I use assign method. Maybe another way would be better.
        root_dir = root_dir or f"./out/{self.name}"
        key_truth = df_truth["tof_pos_x"].round(int(-np.log10(tol)))
        key_repl  = df_replaced["tof_pos_x"].round(int(-np.log10(tol)))
        df_truth_ = df_truth.assign(_key=key_truth)
        df_repl_  = df_replaced.assign(_key=key_repl)

        merged = (
            df_truth_
            .merge(
                df_repl_,
                on=["event", "_key"],
                suffixes=("_tru", "_rec"),
                how="inner",
            )
        )
        if merged.empty:
            print(f"[{detector_label}] no matched rows → nothing to plot")
            return

        dx = merged["tof_pos_x_tru"] - merged["tof_pos_x_rec"]
        dy = merged["tof_pos_y_tru"] - merged["tof_pos_y_rec"]
        dz = merged["tof_pos_z_tru"] - merged["tof_pos_z_rec"]
        dt = (merged["tof_time_tru"]  - merged["tof_time_rec"]) * 1000  #ps

        cfgs = [
            (dx, [-2, 2],  "Δx [mm]", "dx"),
            (dy, [-2, 2],  "Δy [mm]", "dy"),
            (dz, [-2, 2], "Δz [mm]", "dz"),
            (dt, [-100, 100],  "Δt [ps]", "dt"),
        ]
        for data, hr, xl, tag in cfgs:
            myfunc.make_histogram_root(
                data,
                nbins=100,
                hist_range=hr,
                title=f"{detector_label}_{tag}",
                xlabel=xl,
                ylabel="Entries",
                outputname=f"{root_dir}/{detector_label}_{tag}",
                rootfile=self.rootfile,
            )
        print(f"[{detector_label}] difference plots done ({len(merged)} matches).")


    def filtered_stable_particle_hit_and_generated_point(
        self,
        btof_df: pd.DataFrame,
        etof_df: pd.DataFrame,
        plot_verbose: bool=False
    ) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        generator_status==1, charge!=0, |vertex_z|<5 
        """
        # barrel
        b_mask = (
            (btof_df.mc_generator_status == 1) &
            (btof_df.mc_charge           != 0) &
            (btof_df.mc_vertex_z         > -5) &
            (btof_df.mc_vertex_z         <  5)
        )
        b_stable = btof_df[b_mask].reset_index(drop=True)
        b_stable.to_csv(f'./out/{self.name}/stable_particle_btof_hit.csv', index=False)

        # endcap
        e_mask = (
            (etof_df.mc_generator_status == 1) &
            (etof_df.mc_charge           != 0) &
            (etof_df.mc_vertex_z         > -5) &
            (etof_df.mc_vertex_z         <  5)
        )
        e_stable = etof_df[e_mask].reset_index(drop=True)
        e_stable.to_csv(f'./out/{self.name}/stable_particle_etof_hit.csv', index=False)

        if plot_verbose:
            self._plot_matched_hits(MatchedHitInfo(df=b_stable, ak_array=ak.Array(b_stable.to_dict("list"))),
                                    area="btof_stable")
            self._plot_matched_hits(MatchedHitInfo(df=e_stable, ak_array=ak.Array(e_stable.to_dict("list"))),
                                    area="etof_stable")

        return b_stable, e_stable

    def isReconstructedHit(
        self,
        b_stable_df: pd.DataFrame,
        e_stable_df: pd.DataFrame,
        plot_verbose: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Checks if a hit is reconstructed.

        Args:
            b_stable_df: dataframe of stable barrel TOF hits 
            e_stable_df: dataframe of stable endcap TOF hits 
            plot_verbose: True to plot the matched hits

        Returns:
            filtered_btof: dataframe of barrel TOF hits that are reconstructed
            filtered_etof: dataframe of endcap TOF hits that are reconstructed
        """

        # barrel
        if self.version == '1.24.2':
            btof_rec_x_arr = self.dis_file[
                self.branch['tof']['barrel']['rec_hits_branch'][1]
            ].array(library="ak")
        else:
            btof_rec_x_arr = self.dis_file[
                self.branch['tof']['barrel']['rec_hits_branch_old'][1]
            ].array(library="ak")

        filtered_rows_btof = []

        for event in b_stable_df['event'].unique():
            df_evt = b_stable_df[b_stable_df['event'] == event].reset_index(drop=True)
            new_x = df_evt['tof_pos_x'].values.astype(float)
            rec_x = np.array(btof_rec_x_arr[event], dtype=float)

            matching_idxs = []
            for x in rec_x:
                idx = np.where(np.isclose(new_x, x, atol=1e-1))[0]
                if idx.size > 0:
                    closest = idx[np.argmin(np.abs(new_x[idx] - x))]
                    matching_idxs.append(closest)
            if matching_idxs:
                filtered_rows_btof.append(df_evt.iloc[matching_idxs])

        filtered_btof = (
            pd.concat(filtered_rows_btof, ignore_index=True)
            if filtered_rows_btof else pd.DataFrame(columns=b_stable_df.columns)
        )
        filtered_btof.to_csv(
            f"./out/{self.name}/filtered_stable_btof_hit_info.csv",
            index=False
        )

        # endcap
        etof_rec_x_arr = self.dis_file[
            self.branch['tof']['endcap']['rec_hits_branch'][1]
        ].array(library="ak")
        filtered_rows_etof = []

        for event in e_stable_df['event'].unique():
            df_evt = e_stable_df[e_stable_df['event'] == event].reset_index(drop=True)
            new_x = df_evt['tof_pos_x'].values.astype(float)
            rec_x = np.array(etof_rec_x_arr[event], dtype=float)

            matching_idxs = []
            for x in rec_x:
                idx = np.where(np.isclose(new_x, x, atol=1e-1))[0]
                if idx.size > 0:
                    closest = idx[np.argmin(np.abs(new_x[idx] - x))]
                    matching_idxs.append(closest)
            if matching_idxs:
                filtered_rows_etof.append(df_evt.iloc[matching_idxs])

        filtered_etof = (
            pd.concat(filtered_rows_etof, ignore_index=True)
            if filtered_rows_etof else pd.DataFrame(columns=e_stable_df.columns)
        )
        filtered_etof.to_csv(
            f"./out/{self.name}/filtered_stable_etof_hit_info.csv",
            index=False
        )

        if plot_verbose:
            b_reco_info = MatchedHitInfo(
                df=filtered_btof,
                ak_array=ak.Array(filtered_btof.to_dict("list"))
            )
            e_reco_info = MatchedHitInfo(
                df=filtered_etof,
                ak_array=ak.Array(filtered_etof.to_dict("list"))
            )
            self._plot_matched_hits(b_reco_info, area="btof_reco")
            self._plot_matched_hits(e_reco_info, area="etof_reco")

        return filtered_btof, filtered_etof
    

    # Replace TOF hits with reconstructed hits
    def replace_tof_with_reconstructed(
        self,
        truth_btof: pd.DataFrame,
        truth_etof: pd.DataFrame,
        plot_verbose: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters
        ----------
        truth_btof, truth_etof : DataFrames
            Stable-particle truth hits (after vertex/charge cuts).
        Returns
        -------
        df_b_reco, df_e_reco : DataFrames
            Reconstructed hits with truth hits replaced by reconstructed hits.
        """

        # Converted Numpy/Awkward to Python
        def _native(val: Any) -> Any:
            """Convert awkward array to native Python type."""
            if isinstance(val, (np.generic, numbers.Integral, numbers.Real)):
                return val.item()
            if isinstance(val, np.ndarray):
                try:          # 0-d ndarray → scalar
                    return val.item()
                except ValueError:
                    return val.tolist()
            if isinstance(val, ak.highlevel.Array):
                return val.tolist()
            return val

        # Process each region separately
        def _process_region(truth_df: pd.DataFrame,
                            rec_tags: List[str]) -> pd.DataFrame:

            rec_buf: Dict[str, ak.Array] = {
                tag: self.dis_file[tag].array(library="ak") for tag in rec_tags
            }
            # Replaced values will be stored in this list
            out_chunks: List[pd.DataFrame] = []

            for ev in sorted(truth_df["event"].unique()):
                # Re-number lines for truth hits
                tru_evt = truth_df.query("event == @ev").copy().reset_index(drop=True)

                rec_x = np.asarray(rec_buf[rec_tags[1]][ev], dtype=float)
                # Find the closest reconstructed hit to each truth hit within 0.1 mm
                diff = np.abs(tru_evt["tof_pos_x"].to_numpy()[:, None] - rec_x[None, :])
                # Find the indices of the closest reconstructed hits
                truth_idx, rec_idx = np.where(diff < 1e-1)
                if truth_idx.size == 0:        
                    continue

                # Get the truth hit information to associate with the reconstructed hit
                matched = tru_evt.iloc[truth_idx].copy()
                colmap = {
                    "time":"tof_time",
                    "position.x": "tof_pos_x",
                    "position.y": "tof_pos_y",
                    "position.z": "tof_pos_z",
                }
                for full_tag in rec_tags:
                    short = full_tag.split(".")[-1]       
                    if short not in colmap:              
                        continue
                    # Convert the reconstructed hit information to native Python types
                    new_vals = [_native(rec_buf[full_tag][ev][i]) for i in rec_idx]
                    matched[colmap[short]] = new_vals

                out_chunks.append(matched)

            return (
                pd.concat(out_chunks, ignore_index=True)
                if out_chunks else
                pd.DataFrame(columns=truth_df.columns)
            )
        barrel_rec_tags = self.branch["tof"]["barrel"]["rec_hits_branch"]
        endcap_rec_tags = self.branch["tof"]["endcap"]["rec_hits_branch"]

        df_b_reco = _process_region(truth_btof, barrel_rec_tags)
        df_e_reco = _process_region(truth_etof, endcap_rec_tags)

        outdir = f"./out/{self.name}"
        df_b_reco.to_csv(f"{outdir}/btof_reconstructed_only.csv", index=False)
        df_e_reco.to_csv(f"{outdir}/etof_reconstructed_only.csv", index=False)

        if plot_verbose:
            info_b = MatchedHitInfo(df_b_reco, ak.Array(df_b_reco.to_dict("records")))
            info_e = MatchedHitInfo(df_e_reco, ak.Array(df_e_reco.to_dict("records")))
            self._plot_matched_hits(info_b, area="btof_reco")
            self._plot_matched_hits(info_e, area="etof_reco")
            self.plot_replacement_difference(
                truth_btof, df_b_reco, "btof", root_dir=outdir
            )
            self.plot_replacement_difference(
                truth_etof, df_e_reco, "etof", root_dir=outdir
            )
            
        return df_b_reco, df_e_reco