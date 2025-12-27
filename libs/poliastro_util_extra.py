from astropy import units as u
from astropy import time
import astropy
import numpy as np

from poliastro import iod
from poliastro.bodies import Body,Mars, Earth, Venus, Jupiter, Saturn, Uranus, Neptune, Sun, Europa, Ganymede, Callisto, Io, Titan
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.frames import Planes
from poliastro.frames.fixed import JupiterFixed
from poliastro.frames.equatorial import JupiterICRS
from astroquery.jplhorizons import Horizons
from copy import deepcopy

from scipy import ndimage
from scipy.spatial.transform import Rotation as scipyRot

from poliastro.plotting import OrbitPlotter3D, StaticOrbitPlotter
import math
import matplotlib.pyplot as plt
# More info: https://plotly.com/python/renderers/
import plotly.io as pio
from poliastro.util import norm, time_range
pio.renderers.default = "plotly_mimetype+notebook_connected"
import weakref
from astropy.coordinates import solar_system_ephemeris
from collections.abc import Mapping

solar_system_ephemeris.set("jpl")

def match_astro_query_num(body):
    if body==Io:
        return 501
    if body==Europa:
        return 502
    if body==Ganymede:
        return 503
    if body==Callisto:
        return 504
    else:
        return False

#tries to find the lowest DV burn at periapsis of the initial orbit that 
#leads to encounter with target body
def get_single_burn_elliptical_hohmann(attractor, target, initial_orb, min_tof, max_tof, tof_step=0.1*u.day):
    time_till_pe=initial_orb.period-initial_orb.t_p
    periapsis_epoch=initial_orb.epoch+time_till_pe

    sim_range=time_range(start=periapsis_epoch,end=periapsis_epoch+max_tof,periods=50)
    min_dv=50000*u.m/u.s
    final_orb=None
    final_date=None
    final_burn=None
    final_targ_orb=None
    periapsis_orb=initial_orb.propagate(periapsis_epoch)
    body_num=match_astro_query_num(target)
    body_ephem=Ephem.from_horizons(body_num,epochs=sim_range, attractor=attractor, plane=Planes.EARTH_ECLIPTIC)
    body_orb_placeholder = Orbit.from_ephem(attractor, body_ephem, periapsis_epoch)
    tof_range=np.arange(min_tof.to(u.day).value,max_tof.to(u.day).value,tof_step.to(u.day).value)*u.day
    print(f"Getting ephems from {periapsis_epoch}.")
    for i in range(len(tof_range)):
        arrival_date=periapsis_epoch+tof_range[i]
        body_orb=body_orb_placeholder.propagate(arrival_date+1*u.s)
        # print(f"{initial_orb} and {body_orb}")
        lambert=Maneuver.lambert(periapsis_orb,body_orb,M=0)
        burn=lambert[0]
        dv=np.linalg.norm(burn[1])
        if dv<min_dv:
            min_dv=dv
            final_orb,dummy=periapsis_orb.apply_maneuver(lambert,intermediate=True)
            final_date=arrival_date
            final_burn=burn
            final_targ_orb=body_orb
    return min_dv,final_orb,final_date,final_burn, final_targ_orb

##find lowest possible period orbit that is still a multiple of ganymede's period, this can continue until period approaches Ganymedes
def search_for_resonant_orbit(body_orb, body, inc_orb_vel, r_p_min, r_p_max, num_samples=150, max_resonance_ratio=10,lower=True):
    """
    Find the gravity assist that produces the lowest period orbit that is still a resonant multiple of the body's period.
    
    Parameters:
    body_orb: Orbit of the body we're slingshotting around
    body: The body itself (e.g., Ganymede)
    inc_orb_vel: Incoming velocity vector of spacecraft relative to Jupiter
    r_p_min, r_p_max: Min/max periapsis distances for flyby
    num_samples: Number of periapsis values to scan
    max_resonance_ratio: Maximum numerator or denominator to check (default 10)
    
    Returns:
    best_orbit: The resonant orbit with lowest period
    best_resonance: Tuple (n_sc, n_body) representing the resonance ratio
    best_rp: Periapsis distance that achieves this
    """
    
    slingshot_epoch = body_orb.epoch
    body_period = body_orb.period
    
    # Relative velocity & speed to body we are slingshotting around
    rel_vel = inc_orb_vel - body_orb.rv()[1]
    rel_speed = np.linalg.norm(rel_vel)
    
    # Sample periapsis distances
    r_ps = body.R + np.linspace(r_p_min.value, r_p_max.value, num=num_samples) * u.km
    
    # Calculate deflection angles for each periapsis
    rot_angs = 2 * np.arcsin(1 / ((r_ps * rel_speed**2 / body.k) + 1 * u.one))
    
    # Rotation axis in ecliptic plane
    axis = np.array([0, 0, 1.0])
    axis = axis / np.linalg.norm(axis)
    
    best_orbit = None
    best_resonance = None
    best_rp = None
    if lower:
        best_period = np.inf * u.day
    else:
        best_period = 0 * u.day
    
    # Try both rotation directions (leading/trailing side flybys)
    for sign in [1,-1]:
        for i, rot_ang in enumerate(rot_angs):
            # Rotate the relative velocity
            rot = scipyRot.from_rotvec(sign * axis * rot_ang.value)
            rel_vel_out = rot.apply(rel_vel.value) * u.km / u.s
            
            # Calculate post-assist velocity in Jupiter frame
            post_assist_vel = rel_vel_out + body_orb.rv()[1]
            
            # Create orbit from post-assist state
            test_orb = Orbit.from_vectors(
                Jupiter, 
                body_orb.r, 
                post_assist_vel, 
                slingshot_epoch,
                plane=Planes.EARTH_ECLIPTIC
            )
            
            #only consider orbits that don't crash into the planet
            if test_orb.r_p < Jupiter.R:
                continue
            
            # Only consider bound orbits
            if test_orb.ecc >= 1:
                continue
            
            # Check if orbit crosses body's orbital radius (necessary for resonance)
            body_radius = body_orb.a
            orbit_crosses = (test_orb.r_p < body_radius < test_orb.r_a)
            
            if not orbit_crosses:
                continue
            
            # Calculate the period ratio
            period_ratio = test_orb.period / body_period
            # print(period_ratio)
            # Check for integer resonances by testing simple fractions
            found_resonance = False
            for n_body in range(1, max_resonance_ratio + 1):
                for n_sc in range(1, max_resonance_ratio + 1):
                    expected_ratio = n_body / n_sc
                    ratio_error = abs(period_ratio.value - expected_ratio) / expected_ratio
                    
                    if ratio_error < 0.03 :
                        # Check if this is better than current best
                        is_better = (test_orb.period < best_period) if lower else (test_orb.period > best_period)
                        if is_better:
                            # Additional check for lower=False: only accept if period is LONGER than body period
                            if not lower and test_orb.period <= body_period:
                                continue
                            
                            best_period = test_orb.period
                            best_orbit = test_orb
                            best_resonance = (n_sc, n_body)
                            best_rp = r_ps[i]
                            print(f"Found {n_sc}:{n_body} resonance: period={test_orb.period.to(u.day):.2f}, r_p={r_ps[i]:.0f}")
                            found_resonance = True
                            break
                if found_resonance:
                    break
    
    if best_orbit is None:
        print("No resonant orbit found!")
        return None, None, None
    
    print(f"\nBest resonance: {best_resonance[0]}:{best_resonance[1]}")
    print(f"Period: {best_period.to(u.day):.2f} (target body: {body_period.to(u.day):.2f})")
    print(f"Periapsis: {best_rp:.0f}")
    
    return best_orbit, best_resonance, best_rp
    
    
    
def match_orbit_plane(source_orbit, target_orbit):
    """
    Create a new orbit with the same orbital parameters as source_orbit
    but with the inclination and RAAN from target_orbit (making it coplanar).
    
    Parameters
    ----------
    source_orbit : Orbit
        Orbit whose parameters (a, ecc, argp, nu) will be used
    target_orbit : Orbit
        Orbit whose plane (inc, raan) will be used
        
    Returns
    -------
    Orbit
        New orbit with source parameters in target's plane
    """
    
    # Get orbital elements from source
    a = source_orbit.a
    ecc = source_orbit.ecc
    argp = source_orbit.argp
    nu = source_orbit.nu
    
    # Get plane orientation from target
    inc = target_orbit.inc
    raan = target_orbit.raan
    
    # Create new orbit with combined parameters
    new_orbit = Orbit.from_classical(
        attractor=source_orbit.attractor,
        a=a,
        ecc=ecc,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu,
        epoch=source_orbit.epoch,
        plane=source_orbit.plane
    )
    
    return new_orbit
