import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

import uproot
import numpy as np
import ROOT as r
import argparse
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from tof_analyzer import TOFAnalyzer, TOFHitInfo
from track_analyzer import TrackAnalyzer, TrackInfo
from mc_analyzer import MCAnalyzer, MCInfo
from matching_mc_and_tof import MatchingMCAndTOF, MatchedHitInfo
from matching_tof_and_track import MatchingTOFAndTrack, MatchedTrackInfo
from tof_pid_performance_manager import ToFPIDPerformanceManager
from pipeline_utils import process_events

@dataclass
class AnalysisConfig:
    """Data class to store analysis configuration."""
    directory_name: str
    analysis_event_type: str
    selected_events: int
    verbose: bool
    plot_verbose: bool
    detail_plot_verbose: bool
    version: str
    vertex_cuts: Dict[str, float]
    file_paths: Dict[str, str]
    branches: Dict[str, Dict[str, Any]]

def load_config(config_path: str) -> AnalysisConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        AnalysisConfig object containing validated configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if 'analysis' not in config:
        raise ValueError("Missing required section: analysis")
    if 'vertex_cuts' not in config:
        raise ValueError("Missing required section: vertex_cuts")
    if 'file_paths' not in config:
        raise ValueError("Missing required section: file_paths")
    if 'branches' not in config:
        raise ValueError("Missing required section: branches")
    
    analysis_config = config['analysis']
    required_analysis_fields = ['directory_name', 'analysis_event_type', 'selected_events', 'version']
    for field in required_analysis_fields:
        if field not in analysis_config:
            raise ValueError(f"Missing required field in analysis section: {field}")
    
    return AnalysisConfig(
        directory_name=analysis_config['directory_name'],
        analysis_event_type=analysis_config['analysis_event_type'],
        selected_events=analysis_config['selected_events'],
        verbose=analysis_config.get('verbose', False),
        plot_verbose=analysis_config.get('plot_verbose', False),
        detail_plot_verbose=analysis_config.get('detail_plot_verbose', False),
        version=analysis_config['version'],
        vertex_cuts=config['vertex_cuts'],
        file_paths=config['file_paths'],
        branches=config['branches']
    )

def analyzer(config: AnalysisConfig, output_file: str):
    """Main analysis function"""
    # Find matching file path based on analysis_event_type
    file_path = None
    for path_info in config.file_paths.values():
        if config.analysis_event_type in path_info['description']:
            file_path = path_info['path']
            break
    
    if not file_path:
        raise ValueError(f"No matching file path found for analysis_event_type: {config.analysis_event_type}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output ROOT file
    rootfile = r.TFile(output_file, "RECREATE")
    tree = uproot.open(file_path)
    try:
        tree = tree['events']
    except KeyError:
        raise RuntimeError(f"Key 'events' not found in file: {file_path}")
    
    name = config.directory_name
    
    # Initialize analyzers
    print(f"Starting Initialization of analyzers for {name}")
    mc_analyzer = MCAnalyzer(
        dis_file=tree,
        branch=config.branches['mc'],
        name=name,
        rootfile=rootfile
    )
    track_analyzer = TrackAnalyzer(
        dis_file=tree,
        branch=config.branches['track'],
        name=name,
        rootfile=rootfile
    )
    tof_analyzer = TOFAnalyzer(
        dis_file=tree,
        branch=config.branches['tof'],
        name=name,
        rootfile=rootfile
    )
    matching_mc_and_tof = MatchingMCAndTOF(
        branch=config.branches,
        version=config.version,
        rootfile=rootfile,
        name=name,
        dis_file=tree
    )
    matching_tof_and_track = MatchingTOFAndTrack(
        tof=tof_analyzer,
        track=track_analyzer,
        rootfile=rootfile,
        name=name
    )
    tof_pid_performance = ToFPIDPerformanceManager(
        dis_file=tree,
        branch=config.branches['tof'],
        name=name,
        rootfile=rootfile
    )
    # Process events
    process_events(
        selected_events=config.selected_events,
        analyzers=[
            mc_analyzer,
            track_analyzer,
            tof_analyzer,
            matching_mc_and_tof,
            matching_tof_and_track,
            tof_pid_performance
        ]
    )
        
    # Close output file
    rootfile.Close()

def main():
    """Main function to execute the analysis."""
    parser = argparse.ArgumentParser(description='TOF PID Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True, help='Output ROOT file name')
    parser.add_argument('--filetype', type=str, choices=['NCDIS', 'single_particle_pion'], required=True,
                      help='Type of input file (NCDIS or single_particle_pion)')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override filetype in config
        config.analysis_event_type = args.filetype
        
        # Create output directory if it doesn't exist
        os.makedirs(f'./out/{config.directory_name}', exist_ok=True)
        
        # Create output ROOT file
        output_file = f'./out/{config.directory_name}/{args.output}'
        
        # Run analysis
        analyzer(config, output_file)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 