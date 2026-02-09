import re
import numpy as np

def parse_spenvis_spex(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    energies = None
    apogee_km = perigee_km = period_hr = None

    for ln in lines:
        s = ln.strip()
        if energies is None and s.startswith("'ENERGY'"):
            parts = [p.strip() for p in s.split(",")]
            nE = int(parts[1])
            energies = np.array([float(x) for x in parts[2:2+nE]], dtype=float)

        m = re.search(r"Apogee:\s*([0-9.+-Ee]+)\s*km", s)
        if m: apogee_km = float(m.group(1))
        m = re.search(r"Perigee:\s*([0-9.+-Ee]+)\s*km", s)
        if m: perigee_km = float(m.group(1))
        m = re.search(r"Period:\s*([0-9.+-Ee]+)\s*hrs", s)
        if m: period_hr = float(m.group(1))

    if energies is None:
        raise ValueError(f"Could not find ENERGY bins in {path}")

    nE = len(energies)
    rows = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("-1.0000E+00"):
            vals = [float(x) for x in s.split(",")]
            if len(vals) == 2 + nE:
                rows.append(vals[2:])  # drop B,L

    if not rows:
        raise ValueError(f"No data rows parsed from {path}")

    Jint = np.asarray(rows, dtype=float)   # (N, nE): integral flux J(>E_i)
    return energies, Jint, apogee_km, perigee_km, period_hr

def kepler_r_from_apo_peri_period(apogee_km, perigee_km, period_hr, dt_s, N):
    ra, rp = float(apogee_km), float(perigee_km)
    a = 0.5 * (ra + rp)
    e = (ra - rp) / (ra + rp)
    T = float(period_hr) * 3600.0
    n = 2.0 * np.pi / T

    t = np.arange(N, dtype=float) * dt_s
    M = (n * t) % (2.0 * np.pi)  # assumes t=0 at perijove

    E = M.copy()
    for _ in range(30):
        dE = -(E - e*np.sin(E) - M) / (1.0 - e*np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-13:
            break

    r_km = a * (1.0 - e*np.cos(E))
    return t, r_km

def bin_average_by_radius(r_RJ, Y, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    nbin = len(centers)
    ncol = Y.shape[1]
    bin_id = np.clip(np.digitize(r_RJ, edges) - 1, 0, nbin - 1)

    counts = np.bincount(bin_id, minlength=nbin).astype(float)
    Yb = np.empty((nbin, ncol), dtype=float)
    for j in range(ncol):
        sums = np.bincount(bin_id, weights=Y[:, j], minlength=nbin).astype(float)
        Yb[:, j] = sums / np.maximum(counts, 1.0)

    good = counts > 0
    if not np.all(good):
        for j in range(ncol):
            Yb[~good, j] = np.interp(centers[~good], centers[good], Yb[good, j])

    return centers, Yb

def build_fast_e_p_spectrum_lookup(
    spe_path, spp_path,
    dt_s=360.0,
    RJ_km=71492.0,
    r_bin_width_RJ=0.05
):
    Ee, Je, apogee_km, perigee_km, period_hr = parse_spenvis_spex(spe_path)
    Ep, Jp, ap2, pe2, pr2 = parse_spenvis_spex(spp_path)

    apogee_km = apogee_km if apogee_km is not None else ap2
    perigee_km = perigee_km if perigee_km is not None else pe2
    period_hr = period_hr if period_hr is not None else pr2
    if any(v is None for v in (apogee_km, perigee_km, period_hr)):
        raise ValueError("Could not parse orbit apogee/perigee/period from file headers.")

    N = min(Je.shape[0], Jp.shape[0])
    Je = Je[:N]
    Jp = Jp[:N]

    _, r_km = kepler_r_from_apo_peri_period(apogee_km, perigee_km, period_hr, dt_s, N)
    r_RJ = r_km / RJ_km

    rmin, rmax = float(r_RJ.min()), float(r_RJ.max())
    edges = np.arange(rmin, rmax + r_bin_width_RJ, r_bin_width_RJ)
    r_centers, Je_b = bin_average_by_radius(r_RJ, Je, edges)
    _,         Jp_b = bin_average_by_radius(r_RJ, Jp, edges)

    # Differential approx from integral thresholds: j ~ (J(>Ei)-J(>Ei+1))/dE
    dEe = np.diff(Ee); Ee_mid = 0.5*(Ee[:-1] + Ee[1:])
    dEp = np.diff(Ep); Ep_mid = 0.5*(Ep[:-1] + Ep[1:])
    je_b = np.clip((Je_b[:, :-1] - Je_b[:, 1:]) / dEe[None, :], 0.0, None)
    jp_b = np.clip((Jp_b[:, :-1] - Jp_b[:, 1:]) / dEp[None, :], 0.0, None)

    def interp_spectrum(rq, X, Y2d):
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        out = np.empty((rq.size, Y2d.shape[1]), dtype=float)
        for j in range(Y2d.shape[1]):
            out[:, j] = np.interp(rq, X, Y2d[:, j], left=Y2d[0, j], right=Y2d[-1, j])
        return out[0] if out.shape[0] == 1 else out

    def flux_at_radius(r_RJ_query):
        return {
            "e": {
                "E_thr_MeV": Ee,
                "Jint_gt_E": interp_spectrum(r_RJ_query, r_centers, Je_b),
                "E_mid_MeV": Ee_mid,
                "jd_MeV":    interp_spectrum(r_RJ_query, r_centers, je_b),
            },
            "p": {
                "E_thr_MeV": Ep,
                "Jint_gt_E": interp_spectrum(r_RJ_query, r_centers, Jp_b),
                "E_mid_MeV": Ep_mid,
                "jd_MeV":    interp_spectrum(r_RJ_query, r_centers, jp_b),
            },
        }

    return {
        "r_centers": r_centers,
        "electron": {"E_thr_MeV": Ee, "Jint_binned": Je_b, "E_mid_MeV": Ee_mid, "jd_binned": je_b},
        "proton":   {"E_thr_MeV": Ep, "Jint_binned": Jp_b, "E_mid_MeV": Ep_mid, "jd_binned": jp_b},
        "flux_at_radius": flux_at_radius,
        "orbit_params": {"apogee_km": apogee_km, "perigee_km": perigee_km, "period_hr": period_hr},
    }

def electron_flux_gt2MeV(r_list_RJ, model,maxE=1):#maxE in Mev
    r = np.asarray(r_list_RJ, dtype=float)

    # electron threshold grid + pick bin closest to maxE
    Ee = model["electron"]["E_thr_MeV"]
    i2 = int(np.argmin(np.abs(Ee - maxE)))

    # use binned integral table directly (vectorized)
    r_centers = model["r_centers"]
    Je2 = model["electron"]["Jint_binned"][:, i2]  # J(>~2MeV) vs r_centers

    # flux at your radii
    return np.interp(r, r_centers, Je2, left=Je2[0], right=Je2[-1])

def electron_spectrum_at_radius(model, r_RJ):
    """
    Return the electron spectrum at radius r_RJ using the precomputed SPENVIS-based model.

    Parameters
    ----------
    model : dict
        Output of build_fast_e_p_spectrum_lookup(...)
    r_RJ : float or array-like
        Radius (Jupiter radii)

    Returns
    -------
    spec : dict
        {
          "r_RJ": r_RJ,
          "E_thr_MeV": (nE,) thresholds from SPENVIS,
          "Jint_gt_E": (nE,) or (N,nE) integral flux above thresholds,
          "E_mid_MeV": (nE-1,) mid-bin energies,
          "jd_MeV":    (nE-1,) or (N,nE-1) approx differential flux per MeV
        }
    """
    # If you kept model["flux_at_radius"] from earlier, use it (it already interpolates).
    if "flux_at_radius" in model:
        out = model["flux_at_radius"](r_RJ)["e"]
        return {
            "r_RJ": r_RJ,
            "E_thr_MeV": out["E_thr_MeV"],
            "Jint_gt_E": out["Jint_gt_E"],
            "E_mid_MeV": out["E_mid_MeV"],
            "jd_MeV": out["jd_MeV"],
        }

    # Otherwise: do it directly from binned tables (still fast, fully vectorized in radius)
    r_centers = model["r_centers"]
    Ee = model["electron"]["E_thr_MeV"]
    Je_b = model["electron"]["Jint_binned"]
    Ee_mid = model["electron"]["E_mid_MeV"]
    je_b = model["electron"]["jd_binned"]

    r = np.atleast_1d(np.asarray(r_RJ, dtype=float))

    # Interp each energy column in radius
    J = np.empty((r.size, Je_b.shape[1]), dtype=float)
    for j in range(Je_b.shape[1]):
        col = Je_b[:, j]
        J[:, j] = np.interp(r, r_centers, col, left=col[0], right=col[-1])

    jd = np.empty((r.size, je_b.shape[1]), dtype=float)
    for j in range(je_b.shape[1]):
        col = je_b[:, j]
        jd[:, j] = np.interp(r, r_centers, col, left=col[0], right=col[-1])

    # If input was scalar, return 1D arrays
    if np.isscalar(r_RJ):
        J = J[0]
        jd = jd[0]

    return {
        "r_RJ": r_RJ,
        "E_thr_MeV": Ee,
        "Jint_gt_E": J,
        "E_mid_MeV": Ee_mid,
        "jd_MeV": jd,
    }
