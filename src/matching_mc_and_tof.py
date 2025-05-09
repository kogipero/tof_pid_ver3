import numpy as np
import pandas as pd
import awkward as ak
import uproot
import ROOT as r
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
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
        self.branch   = branch
        self.version  = version
        self.rootfile = rootfile
        self.name     = name
        self.dis_file = dis_file
        self.plotter  = MatchingMCAndTOFPlotter(rootfile, name)

        tof_br = branch['tof']

        self.btof_rec_x_branch = tof_br['barrel']['rec_hits_branch'][1]
        self.etof_rec_x_branch = tof_br['endcap']['rec_hits_branch'][1]
        # Barrel TOF
        self.btof_raw_hit_mc_associaction_branch = (
            tof_br['barrel']['mc_associations_ver1_24_2_branch']
            if version=='1.24.2'
            else tof_br['barrel']['mc_associations_branch']
        )
        # Endcap TOF
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
        1) Match MC particles with TOF hits.
        2) Filter stable particles.
        3) Filter reconstructed hits.
        """
        # ── 1) raw matching ──
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

        # ── 3) Reconstructed-only filter ──
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
        mc_index_awk = self.dis_file[assoc_branch[0]].array(library="ak")[:n_evt]
        rows = []
        for ev in tqdm(range(n_evt), desc=f"{area} match", unit="evt"):
            sel = mc_index_awk[ev] >= 0
            if not ak.any(sel): continue

            mc_sel = mc_index_awk[ev][sel]
            # TOF
            hx, hy, hz = tof.pos_x[ev][sel], tof.pos_y[ev][sel], tof.pos_z[ev][sel]
            htime     = tof.time[ev][sel]
            hphi, htheta, hr = tof.phi[ev][sel], tof.theta[ev][sel], tof.r[ev][sel]
            # MC
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
        btof_rec_x_arr = self.dis_file[
            self.branch['tof']['barrel']['rec_hits_branch'][1]
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