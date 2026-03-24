"""
Lambert's Problem: The Vacant Focus Hyperbola
==============================================
Animates how the locus of the vacant (empty) focus of all possible
transfer ellipses between two points forms a hyperbola.

Key insight:
  For any transfer ellipse with occupied focus F1 (central body) and
  vacant focus F2, the ellipse definition gives:
      |P1 - F2| + r1 = 2a
      |P2 - F2| + r2 = 2a
  Subtracting:
      |P1 - F2| - |P2 - F2| = r2 - r1  (a constant!)

  This is the equation of a hyperbola with foci at P1 and P2.
  Each point on this hyperbola defines a unique transfer orbit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Gravitational parameter (normalised) ──────────────────────────
mu = 1.0

# ── Terminal positions ────────────────────────────────────────────
theta1, theta2 = np.radians(20), np.radians(155)   # true anomalies
r1_mag, r2_mag = 1.0, 1.8                          # radial distances

P1 = r1_mag * np.array([np.cos(theta1), np.sin(theta1)])
P2 = r2_mag * np.array([np.cos(theta2), np.sin(theta2)])

c_chord = np.linalg.norm(P2 - P1)       # chord length
s = (r1_mag + r2_mag + c_chord) / 2.0   # semi-perimeter

# ── Hyperbola of vacant-focus positions ───────────────────────────
#  |P1-F2| - |P2-F2| = r2 - r1  =  constant "diff"
diff = r2_mag - r1_mag                   # signed difference

h_center = (P1 + P2) / 2.0
h_c = c_chord / 2.0                     # half-distance between foci
h_a = abs(diff) / 2.0                   # semi-major axis of hyperbola
h_b = np.sqrt(h_c**2 - h_a**2)          # semi-minor axis

# Rotation from x-axis to P1→P2 direction
direction = P2 - P1
phi = np.arctan2(direction[1], direction[0])
cos_phi, sin_phi = np.cos(phi), np.sin(phi)


def hyperbola_point(t):
    """Return (x, y) on the relevant branch of the vacant-focus hyperbola."""
    # In local frame aligned with P1-P2:
    #   branch closer to P2 when diff > 0, closer to P1 when diff < 0
    sign = 1.0 if diff > 0 else -1.0
    x_loc = sign * h_a * np.cosh(t)
    y_loc = h_b * np.sinh(t)
    # Rotate to world frame and translate to hyperbola centre
    x = cos_phi * x_loc - sin_phi * y_loc + h_center[0]
    y = sin_phi * x_loc + cos_phi * y_loc + h_center[1]
    return np.array([x, y])


def transfer_ellipse_points(F2, n=300):
    """Given vacant focus F2, return array of (x,y) points on the transfer ellipse."""
    a_orb = (r1_mag + np.linalg.norm(P1 - F2)) / 2.0
    c_orb = np.linalg.norm(F2) / 2.0
    if a_orb <= c_orb:
        return None, None
    b_orb = np.sqrt(a_orb**2 - c_orb**2)
    center = F2 / 2.0
    psi = np.arctan2(F2[1], F2[0])

    E = np.linspace(0, 2 * np.pi, n)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    x = center[0] + a_orb * np.cos(E) * cos_psi - b_orb * np.sin(E) * sin_psi
    y = center[1] + a_orb * np.cos(E) * sin_psi + b_orb * np.sin(E) * cos_psi
    return x, y


def time_of_flight(a_orb):
    """Lambert TOF:  t = sqrt(a³/μ) [ (α - sin α) - (β - sin β) ]"""
    if a_orb <= 0:
        return np.inf
    val_a = s / (2.0 * a_orb)
    val_b = (s - c_chord) / (2.0 * a_orb)
    if val_a > 1.0 or val_b > 1.0:
        return np.inf
    alpha = 2.0 * np.arcsin(np.sqrt(np.clip(val_a, 0, 1)))
    beta  = 2.0 * np.arcsin(np.sqrt(np.clip(val_b, 0, 1)))
    return np.sqrt(a_orb**3 / mu) * ((alpha - np.sin(alpha)) - (beta - np.sin(beta)))


# ── Precompute the full hyperbola curve for drawing ───────────────
t_vals = np.linspace(-2.0, 2.0, 500)
hyp_pts = np.array([hyperbola_point(t) for t in t_vals])

# ── Animation parameter sweep ────────────────────────────────────
N_FRAMES = 180
t_anim = np.linspace(-1.6, 1.6, N_FRAMES)

# ── Figure setup ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0a0a2a')
ax.set_facecolor('#0a0a2a')

# Colours
CLR_HYP   = '#ff6f61'   # coral – hyperbola
CLR_ORBIT = '#66ccff'   # sky blue – transfer ellipse
CLR_ARC   = '#ffdd44'   # gold – the actual transfer arc
CLR_FOCUS = '#ff3366'   # pink – vacant focus dot
CLR_PT    = '#44ff88'   # green – terminal points
CLR_BODY  = '#ffffff'   # white – central body
CLR_CIRC  = '#445566'   # grey – auxiliary circles
CLR_TEXT  = '#cccccc'

# Static elements
ax.plot(*hyp_pts.T, color=CLR_HYP, lw=1.5, alpha=0.5, label='Vacant focus locus (hyperbola)')
ax.plot(0, 0, 'o', color=CLR_BODY, ms=10, zorder=10, label='Central body (F₁)')
ax.plot(*P1, 's', color=CLR_PT, ms=9, zorder=10, label=f'P₁  r₁={r1_mag:.1f}')
ax.plot(*P2, 'D', color=CLR_PT, ms=9, zorder=10, label=f'P₂  r₂={r2_mag:.1f}')

# Dashed lines from origin to terminal points
ax.plot([0, P1[0]], [0, P1[1]], '--', color=CLR_PT, alpha=0.3, lw=0.8)
ax.plot([0, P2[0]], [0, P2[1]], '--', color=CLR_PT, alpha=0.3, lw=0.8)

# Animated artists
(line_orbit,)  = ax.plot([], [], color=CLR_ORBIT, lw=1.2, alpha=0.6)
(line_arc,)    = ax.plot([], [], color=CLR_ARC,   lw=2.5, alpha=0.9)
(dot_focus,)   = ax.plot([], [], 'o', color=CLR_FOCUS, ms=8, zorder=9)
(line_f1f2,)   = ax.plot([], [], '--', color=CLR_FOCUS, lw=0.8, alpha=0.4)
(circ1_line,)  = ax.plot([], [], color=CLR_CIRC, lw=0.6, alpha=0.35)
(circ2_line,)  = ax.plot([], [], color=CLR_CIRC, lw=0.6, alpha=0.35)

info_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                    fontsize=11, color=CLR_TEXT, va='top', fontfamily='monospace')

equation_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                        fontsize=9, color='#888888', va='bottom', fontfamily='monospace')

ax.set_xlim(-3.0, 2.5)
ax.set_ylim(-2.0, 3.0)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=9, facecolor='#111133',
          edgecolor='#333355', labelcolor=CLR_TEXT)
ax.set_title("Lambert's Problem — Vacant Focus Hyperbola",
             color=CLR_TEXT, fontsize=14, pad=12)
ax.tick_params(colors='#555555')
for spine in ax.spines.values():
    spine.set_color('#333355')


def find_transfer_arc(ex, ey, F2):
    """Identify the segment of the ellipse between P1 and P2 (short way)."""
    psi = np.arctan2(F2[1], F2[0])
    center = F2 / 2.0

    # Eccentric anomaly of P1 and P2 on the ellipse
    def eccentric_anomaly(P):
        dp = P - center
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        x_loc =  cos_psi * dp[0] + sin_psi * dp[1]
        y_loc = -sin_psi * dp[0] + cos_psi * dp[1]
        return np.arctan2(y_loc, x_loc)

    E1 = eccentric_anomaly(P1)
    E2 = eccentric_anomaly(P2)

    a_orb = (r1_mag + np.linalg.norm(P1 - F2)) / 2.0
    c_orb = np.linalg.norm(F2) / 2.0
    b_orb = np.sqrt(max(a_orb**2 - c_orb**2, 1e-12))

    # Generate arc from E1 to E2 going counter-clockwise (short way)
    if E2 < E1:
        E2 += 2 * np.pi
    E_arc = np.linspace(E1, E2, 150)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    x = center[0] + a_orb * np.cos(E_arc) * cos_psi - b_orb * np.sin(E_arc) * sin_psi
    y = center[1] + a_orb * np.cos(E_arc) * sin_psi + b_orb * np.sin(E_arc) * cos_psi
    return x, y


def init():
    line_orbit.set_data([], [])
    line_arc.set_data([], [])
    dot_focus.set_data([], [])
    line_f1f2.set_data([], [])
    circ1_line.set_data([], [])
    circ2_line.set_data([], [])
    info_text.set_text('')
    equation_text.set_text('')
    return line_orbit, line_arc, dot_focus, line_f1f2, circ1_line, circ2_line, info_text, equation_text


def update(frame):
    t = t_anim[frame]
    F2 = hyperbola_point(t)

    # ── Transfer ellipse ──────────────────────────────────────────
    ex, ey = transfer_ellipse_points(F2)
    if ex is None:
        return (line_orbit, line_arc, dot_focus, line_f1f2,
                circ1_line, circ2_line, info_text, equation_text)

    a_orb = (r1_mag + np.linalg.norm(P1 - F2)) / 2.0
    tof = time_of_flight(a_orb)

    # Full ellipse (ghost)
    line_orbit.set_data(ex, ey)

    # Transfer arc (highlighted)
    ax_arc, ay_arc = find_transfer_arc(ex, ey, F2)
    line_arc.set_data(ax_arc, ay_arc)

    # Vacant focus dot
    dot_focus.set_data([F2[0]], [F2[1]])

    # F1 – F2 line
    line_f1f2.set_data([0, F2[0]], [0, F2[1]])

    # Auxiliary circles from P1 and P2 showing the focus definition
    ang = np.linspace(0, 2 * np.pi, 120)
    rad1 = np.linalg.norm(P1 - F2)
    rad2 = np.linalg.norm(P2 - F2)
    circ1_line.set_data(P1[0] + rad1 * np.cos(ang), P1[1] + rad1 * np.sin(ang))
    circ2_line.set_data(P2[0] + rad2 * np.cos(ang), P2[1] + rad2 * np.sin(ang))

    # ── Info text ─────────────────────────────────────────────────
    e_orb = np.linalg.norm(F2) / (2.0 * a_orb)
    info_text.set_text(
        f'Semi-major axis  a = {a_orb:.3f}\n'
        f'Eccentricity     e = {e_orb:.3f}\n'
        f'Time of flight   T = {tof:.3f}\n'
        f'|P₁−F₂|         = {np.linalg.norm(P1-F2):.3f}\n'
        f'|P₂−F₂|         = {np.linalg.norm(P2-F2):.3f}\n'
        f'Difference       = {np.linalg.norm(P1-F2)-np.linalg.norm(P2-F2):.3f}  '
        f'(r₂−r₁ = {diff:.3f})'
    )

    equation_text.set_text(
        '|P₁ − F₂| − |P₂ − F₂|  =  r₂ − r₁   (hyperbola definition)\n'
        'Each point on the hyperbola → a unique transfer ellipse'
    )

    return (line_orbit, line_arc, dot_focus, line_f1f2,
            circ1_line, circ2_line, info_text, equation_text)


anim = FuncAnimation(fig, update, frames=N_FRAMES,
                     init_func=init, interval=60, blit=True)

plt.tight_layout()
plt.show()
