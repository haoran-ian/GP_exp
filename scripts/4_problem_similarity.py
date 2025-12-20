# fmt: off
import os
import sys
import ioh
import torch
import warnings
import numpy as np
sys.path.insert(0, os.getcwd())
from ioh import get_problem, logger
from LLaMEA.llamea import LLaMEA, OpenAI_LLM
from LLaMEA.misc import aoc_logger, correct_aoc, OverBudgetException
from utils.extract_top_funcs import extract_top_funcs
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
# fmt: on

