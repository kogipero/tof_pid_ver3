import numpy as np
import os
import uproot
import vector
import mplhep as hep
import sys
import yaml
from typing import List, Tuple, Dict
import ROOT as r

vector.register_awkward()
hep.style.use(hep.style.ROOT)

# Utility functions
def load_yaml_config(file_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML configuration as a dictionary.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_tree_file(filename: str) -> uproot.TTree:
    """
    Loads a ROOT file and retrieves the 'events' tree.

    Args:
        filename (str): Path to the ROOT file.

    Returns:
        uproot.TTree: The 'events' tree from the ROOT file.
    """
    if not os.path.exists(filename):
        print(f'File {filename} does not exist')
        sys.exit()
    print(f'File {filename} opened')
    file = uproot.open(filename)

    return file['events']

def make_directory(directory_name: str):
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

# Define the angular distance function manually
def angular_distance(phi1, theta1, phi2, theta2):   
    delta_phi = phi1 - phi2

    return np.arccos(
        np.sin(theta1) * np.sin(theta2) +
        np.cos(theta1) * np.cos(theta2) * np.cos(delta_phi)
    )

def gaussian(x, A, mu, sigma):
    return A * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def calc_delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi

def calc_delta_theta(theta1, theta2):
    return theta1 - theta2
