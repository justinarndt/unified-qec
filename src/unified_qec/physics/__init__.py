"""
Physics Layer

Hardware physics models: leakage, cosmic rays, burst errors.
"""

from unified_qec.physics.leakage import LeakageTracker
from unified_qec.physics.cosmic_ray import CosmicRaySimulator
from unified_qec.physics.burst_error import BurstErrorDetector
