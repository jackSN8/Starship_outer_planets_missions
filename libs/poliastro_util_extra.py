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
    print(initial_orb.period.to(u.day))
    print(initial_orb.t_p.to(u.day))
    
    print(initial_orb.period-initial_orb.t_p.to(u.day))

    # print(time_till_pe.to(u.day))

    # print(periapsis_epoch)
    sim_range=time_range(start=periapsis_epoch,end=periapsis_epoch+max_tof,periods=50)
    min_dv=10000*u.m/u.s
    final_orb=None
    final_date=None
    periapsis_orb=initial_orb.propagate(periapsis_epoch)
    body_num=match_astro_query_num(target)
    body_ephem=Ephem.from_horizons(body_num,epochs=sim_range, attractor=attractor, plane=Planes.EARTH_ECLIPTIC)
    body_orb_placeholder = Orbit.from_ephem(attractor, body_ephem, periapsis_epoch)
    tof_range=np.arange(min_tof.to(u.day).value,max_tof.to(u.day).value,tof_step.to(u.day).value)*u.day
    
    for i in range(len(tof_range)):
        arrival_date=periapsis_epoch+tof_range[i]
        body_orb=body_orb_placeholder.propagate(arrival_date+1*u.s)
        # print(f"{initial_orb} and {body_orb}")
        lambert=Maneuver.lambert(periapsis_orb,body_orb,M=0)
        dv=np.linalg.norm(lambert[0][1])
        if dv<min_dv:
            min_dv=dv
            final_orb,dummy=periapsis_orb.apply_maneuver(lambert,intermediate=True)
            final_date=arrival_date
    return min_dv,final_orb,final_date
    
    
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
