from typing import Any
from tof_analyzer import TOFAnalyzer, TOFHitInfo
from track_analyzer import TrackAnalyzer, TrackInfo
from mc_analyzer import MCAnalyzer, MCInfo
from matching_mc_and_tof import MatchingMCAndTOF, MatchedHitInfo
from matching_tof_and_track import MatchingTOFAndTrack, MatchedTrackInfo
from tof_pid_performance_manager import ToFPIDPerformanceManager
from typing import List

def process_events(selected_events: int,
                   analyzers: list[Any]) -> None:
    """
    Very thin orchestration layer.
    - This function is responsible for calling the analyzers in the correct order.
    - It does not contain any analysis logic itself.
    """
    # --- 0) Get analyzers -----------------------------------------------
    mc_an          = next(a for a in analyzers if isinstance(a, MCAnalyzer))
    track_an       = next(a for a in analyzers if isinstance(a, TrackAnalyzer))
    tof_an         = next(a for a in analyzers if isinstance(a, TOFAnalyzer))
    match_mc_tof   = next(a for a in analyzers if isinstance(a, MatchingMCAndTOF))
    match_tof_trk  = next(a for a in analyzers if isinstance(a, MatchingTOFAndTrack))
    perf_manager   = next(a for a in analyzers if isinstance(a, ToFPIDPerformanceManager))

    # --- 1) MC --------------------------------------------------------------
    print("process_events : start MC analysis")
    mc_info = mc_an.get_mc_info(plot_verbose=True)

    # --- 2) TOF -------------------------------------------------------------
    print("process_events : start TOF analysis")
    btof_info, etof_info = tof_an.get_tof_info(
        selected_events = selected_events,
        plot_verbose    = True  
    )

    # --- 3) Track -----------------------------------------------------------
    print("process_events : start Track analysis")
    x, y, z = track_an.get_track_segments_pos(plot_verbose=True)
    px, py, pz, p, pt, th, ph, pl = track_an.get_track_segments_momentum(plot_verbose=True)
    tracks = track_an.split_track_segments(
        x, y, z, px, py, pz, pl,
        SELECTED_EVENTS = selected_events
    )
    tracks_on_btof, tracks_on_etof = track_an.get_track_segments_on_tof_info(
        tracks = tracks,
        plot_verbose    = True
    )

    # --- 4) MC ↔︎ TOF matching -------------------------------------------
    print("process_events : start MC ↔︎ TOF matching")
    matched_btof, matched_etof = match_mc_tof.matching_mc_and_tof(
        mc_info      = mc_info,
        btof_info    = btof_info,
        etof_info    = etof_info,
        selected_events = selected_events,
        plot_verbose  = True
    )

    # --- 5) TOF ↔︎ Track matching ----------------------------------------
    print("process_events : start TOF ↔︎ Track matching")
    matched_tof_and_track_btof, matched_tof_and_track_etof = match_tof_trk.matching_tof_and_track(
        track_segments_on_btof_df = tracks_on_btof,
        filtered_stable_btof_hit_info = matched_btof.df,
        track_segments_on_etof_df = tracks_on_etof,
        filtered_stable_etof_hit_info = matched_etof.df,
        plot_verbose=True
    )

    # --- 6) PID performance -------------------------------------
    print("process_events : start PID performance analysis")
    calc_mass_btof,pdg_btof,p_btof,pt_btof = perf_manager.process_pid_performance_plot(
        tof_and_track_matched_pd = matched_tof_and_track_btof,
        area="btof",
        plot_verbose=True
    )

    calc_mass_etof,pdg_etof,p_etof,pt_etof= perf_manager.process_pid_performance_plot(
        tof_and_track_matched_pd = matched_tof_and_track_etof,
        area="etof",
        plot_verbose=True
    )

    perf_manager.process_separation_power_vs_momentum(
        tof_calc_mass = calc_mass_btof,
        tof_pdg       = pdg_btof,
        track_momentums_on_tof         = p_btof,
        track_momentums_transverse_on_tof        = pt_btof,
        area = "btof",
        plot_verbose=True
    )

    # etof is invalid because it has not yet been analyzed
    # perf_manager.process_separation_power_vs_momentum(
    #     tof_calc_mass = calc_mass_etof,
    #     tof_pdg       = pdg_etof,
    #     track_momentums_on_tof         = p_etof,
    #     track_momentums_transverse_on_tof        = pt_etof,
    #     area = "etof",
    #     plot_verbose=True
    # )

    perf_manager.process_purity_vs_momentum(
        btof_calc_mass = calc_mass_btof,
        btof_pdg       = pdg_btof,
        track_momentums_on_btof         = p_btof,
        track_momentums_transverse_on_btof        = pt_btof,
        area = "btof",
        plot_verbose=True
    )

    print("process_events : all analyzers have finished.")