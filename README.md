# TOF PID Analysis (Version 3)

This is a refactored version of the TOF PID analysis code with the following improvements:

1. **Unified Configuration**
   - All configuration settings are now in a single `config.yaml` file
   - Better structured and more maintainable configuration

2. **Improved Code Organization**
   - Combined TOF reader and plotter into a single `TOFAnalyzer` class
   - Better type hints and documentation
   - More robust error handling

3. **Data Structure Improvements**
   - Using dataclasses for better data organization
   - More consistent data flow between components

## Directory Structure

```
tof_pid_ver3/
├── config/
│   └── config.yaml          # Unified configuration file
├── src/
│   ├── tof_analyzer.py      # Combined TOF analyzer and plotter
│   ├── track_reader.py      # Track data reader
│   ├── mc_reader.py         # MC data reader
│   ├── matching_mc_and_tof.py  # MC-TOF matching
│   ├── matching_tof_and_track.py  # TOF-track matching
│   ├── tof_pid_performance_manager.py  # PID performance analysis
│   ├── utility_function.py  # Utility functions
│   └── helper_functions.py  # Helper functions
├── out/                     # Output directory
└── analyze_script.py        # Main analysis script
```

## Usage

1. Configure the analysis in `config/config.yaml`
2. Run the analysis:
   ```bash
   python analyze_script.py --rootfile output.root
   ```

## Configuration

The `config.yaml` file contains all configuration settings:

- Analysis parameters (events, verbosity, etc.)
- Vertex cuts
- File paths
- Branch names

## Dependencies

- Python 3.6+
- ROOT
- uproot
- numpy
- awkward 