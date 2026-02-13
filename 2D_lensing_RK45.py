# Python code to run an improved CPU black-hole raytracing demo with adaptive RK45,
# better impact-parameter sampling focused near the critical impact parameter,
# and frame-saving to a video (mp4 if ffmpeg available, else a gif).
#
# It will create a file at /mnt/data/blackhole_rk45.(mp4|gif) that you can download.
#
# Notes: This is a visualization/educational demo. It uses a per-ray adaptive RK45 integrator
# (Cash-Karp coefficients) and stores trails. Tune `N_RAYS`, `FRAMES`, and `STEPS_PER_FRAME`
# for speed/quality tradeoffs.
#
# Run-time: moderate. I've kept parameters conservative so it completes in the notebook.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from math import sqrt

# Physical constants (SI)
c = 299792458.0
G = 6.67430e-11

# --- Black hole and ray classes ---
class BlackHole:
    def __init__(self, mass, pos=np.array([0.0,0.0])):
        self.mass = float(mass)
        self.pos = np.array(pos, dtype=float)
        self.r_s = 2.0 * G * self.mass / (c**2)

class Ray:
    def __init__(self, pos, dir_vec, bh: BlackHole):
        # pos, dir_vec: numpy arrays (x,y) in meters and m/s for dir
        self.x, self.y = float(pos[0]), float(pos[1])
        self.bh = bh
        # polar
        self.r = np.hypot(self.x - bh.pos[0], self.y - bh.pos[1])
        self.phi = np.arctan2(self.y - bh.pos[1], self.x - bh.pos[0])
        # velocity components in polar basis
        dx, dy = dir_vec[0], dir_vec[1]
        urx, ury = np.cos(self.phi), np.sin(self.phi)
        utx, uty = -ury, urx
        self.dr = dx * urx + dy * ury
        self.dphi = (dx * utx + dy * uty) / max(self.r, 1e-30)
        # conserved quantities
        self.L = (self.r**2) * self.dphi
        f = 1.0 - bh.r_s / max(self.r, 1e-30)
        td = np.sqrt( (self.dr*self.dr)/(f*f) + (self.r*self.r * self.dphi*self.dphi)/f )
        self.E = f * td
        self.trail = [(self.x, self.y)]
        self.alive = True  # ray active unless it hits horizon or escapes

    def inside_horizon(self):
        return self.r <= self.bh.r_s

# --- Geodesic RHS (same physics as the C++ code) ---
def geodesic_rhs_state(y, E, rs):
    # y = [r, phi, dr, dphi]
    r, phi, dr, dphi = y
    f = 1.0 - rs / r
    out = np.zeros(4, dtype=float)
    out[0] = dr
    out[1] = dphi
    dt_dlambda = E / f
    out[2] = - (rs/(2*r*r)) * f * (dt_dlambda*dt_dlambda) \
             + (rs/(2*r*r*f)) * (dr*dr) \
             + (r - rs) * (dphi*dphi)
    out[3] = -2.0 * dr * dphi / r
    return out

# --- Cash-Karp RK45 (embedded) coefficients ---
# Coeffs from the Cash-Karp tableau
a2 = 1/5
a3 = 3/10
a4 = 3/5
a5 = 1.0
a6 = 7/8

b21 = 1/5
b31 = 3/40; b32 = 9/40
b41 = 3/10; b42 = -9/10; b43 = 6/5
b51 = -11/54; b52 = 5/2; b53 = -70/27; b54 = 35/27
b61 = 1631/55296; b62 = 175/512; b63 = 575/13824; b64 = 44275/110592; b65 = 253/4096

c1 = 37/378; c3 = 250/621; c4 = 125/594; c6 = 512/1771  # 5th-order solution
# c2 and c5 are zero in this representation
# Error estimate coefficients (difference between 5th and 4th order)
dc1 = c1 - 2825/27648
dc3 = c3 - 18575/48384
dc4 = c4 - 13525/55296
dc5 = -277/14336
dc6 = c6 - 1/4

def rk45_cash_karp_step(y, E, rs, h):
    # y: state array length 4
    k1 = geodesic_rhs_state(y, E, rs)
    k2 = geodesic_rhs_state(y + h*b21*k1, E, rs)
    k3 = geodesic_rhs_state(y + h*(b31*k1 + b32*k2), E, rs)
    k4 = geodesic_rhs_state(y + h*(b41*k1 + b42*k2 + b43*k3), E, rs)
    k5 = geodesic_rhs_state(y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4), E, rs)
    k6 = geodesic_rhs_state(y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5), E, rs)

    y5 = y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6)  # 5th order
    # 4th-order estimate via coefficients (from tableau)
    y4 = y + h*(2825/27648*k1 + 18575/48384*k3 + 13525/55296*k4 + 277/14336*k5 + 1/4*k6)
    err = y5 - y4
    return y5, err

# --- Adaptive stepper per ray ---
def adaptive_rk45_step(ray, h, tol):
    # ray: Ray object; h: initial step; tol: relative tolerance
    if not ray.alive:
        return 0.0  # no progress
    rs = ray.bh.r_s
    if ray.inside_horizon():
        ray.alive = False
        return 0.0
    # state vector
    y = np.array([ray.r, ray.phi, ray.dr, ray.dphi], dtype=float)
    E = ray.E

    max_retries = 12
    safety = 0.9
    for _ in range(max_retries):
        y5, err = rk45_cash_karp_step(y, E, rs, h)
        # compute error metric: relative error scaled per-component
        sc = np.maximum(np.abs(y5), np.abs(y)) + 1e-10
        err_norm = np.max(np.abs(err) / sc)
        # acceptable if err_norm < tol
        if err_norm <= tol:
            # accept step
            ray.r, ray.phi, ray.dr, ray.dphi = y5
            ray.x = ray.bh.pos[0] + ray.r * np.cos(ray.phi)
            ray.y = ray.bh.pos[1] + ray.r * np.sin(ray.phi)
            ray.trail.append((ray.x, ray.y))
            # safety-based step scaling
            if err_norm == 0:
                h_new = h * 5.0
            else:
                h_new = h * min(5.0, safety * (tol / err_norm)**0.2)
            return h_new  # return suggested new step
        else:
            # reduce step and retry
            h = h * max(0.1, safety * (tol / err_norm)**0.25)
    # if we get here, step failed; mark ray dead to avoid infinite loop
    ray.alive = False
    return h * 0.1

# --- Sampling impact parameter near critical value ---
def sample_impact_parameters(rs, N, spread=0.6):
    # critical impact parameter b_c = (3 * sqrt(3) / 2) * r_s
    b_c = (3.0 * np.sqrt(3.0) / 2.0) * rs
    # We'll sample more densely near b_c using a mixture of log-space and normal bump
    # Create base linear grid then add clustering around b_c
    base = np.linspace(-2.5*b_c, 2.5*b_c, N)
    # apply a warp to cluster near +/- b_c
    # use tanh-based warp: compress extremes, expand near center, then add concentration near b_c
    t = np.tanh(np.linspace(-1.5, 1.5, N))
    warped = b_c * 2.6 * t  # roughly in range
    # Mix base and warped: closer to warped for mid-range indices
    mix = 0.55
    pts = (1-mix)*base + mix*warped
    # Also add a small gaussian cluster centered on +b_c and -b_c
    cluster = np.concatenate([np.random.normal(b_c, spread*b_c, int(N*0.06)),
                              np.random.normal(-b_c, spread*b_c, int(N*0.06))])
    pts = np.concatenate([pts, cluster])
    # sort
    pts = np.sort(pts)
    return pts

# --- Build rays with initial positions and directions ---
def build_rays(bh, N_rays, x_start_factor=3.0):
    X0 = -x_start_factor * bh.r_s
    offsets = sample_impact_parameters(bh.r_s, N_rays)
    rays = []
    for oy in offsets:
        pos = np.array([X0, oy])
        dir_vec = np.array([c, 0.0])  # initial direction toward +x
        r = Ray(pos, dir_vec, bh)
        rays.append(r)
    return rays

# --- Simulation and animation saving ---
def run_and_save(mass=8.54e36, N_RAYS=180, FRAMES=220, STEPS_PER_FRAME=8, out_path='/mnt/data/blackhole_rk45.mp4'):
    bh = BlackHole(mass)
    rays = build_rays(bh, N_RAYS)
    # plotting setup
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_aspect('equal')
    limit = 3.0 * bh.r_s
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit*0.75, limit*0.75)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Schwarzschild ray tracing — adaptive RK45')

    bh_circle = plt.Circle((0,0), bh.r_s)
    ax.add_artist(bh_circle)
    # one Line2D per ray
    lines = [ax.plot([], [], lw=0.9, alpha=0.7)[0] for _ in rays]

    # initial step sizes per ray (heuristic)
    h_arr = np.array([0.02 * bh.r_s / c for _ in rays])  # affine-step initial guess
    tol = 1e-6

    frames = []  # for possible saving without FuncAnimation
    # We'll produce frames by integrating rays a few adaptive steps per frame
    for frame_idx in range(FRAMES):
        for i, ray in enumerate(rays):
            if not ray.alive:
                continue
            # take several adaptive steps per frame
            for _ in range(STEPS_PER_FRAME):
                h_new = adaptive_rk45_step(ray, h_arr[i], tol)
                # if step returns 0 progress, ray is dead
                if h_new == 0.0:
                    break
                # clamp step to reasonable bounds
                h_arr[i] = max(1e-12, min(h_new, 1e-1 * bh.r_s / c))
                # terminate if far away
                if ray.r > 100.0 * bh.r_s:
                    ray.alive = False
                    break

        # update lines
        for i, ray in enumerate(rays):
            xs = [p[0] for p in ray.trail]
            ys = [p[1] for p in ray.trail]
            lines[i].set_data(xs, ys)

        # capture frame by drawing canvas to an image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image.copy())

    plt.close(fig)

    # Try saving MP4 using ffmpeg; fallback to GIF using Pillow
    saved = False
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='auto'), bitrate=2000)
        print("Saving MP4 with ffmpeg to", out_path)
        # recreate a writer animation (this is lighter than FuncAnimation)
        fig2, ax2 = plt.subplots(figsize=(9,6))
        ax2.set_aspect('equal')
        ax2.imshow(frames[0])
        ax2.axis('off')
        im = ax2.imshow(frames[0])
        def update_frame(i):
            im.set_data(frames[i])
            return [im]
        ani = animation.FuncAnimation(fig2, update_frame, frames=len(frames), blit=True)
        ani.save(out_path, writer=writer)
        plt.close(fig2)
        saved = True
    except Exception as e:
        print("ffmpeg save failed:", e)
    if not saved:
        try:
            import imageio
            gif_path = out_path.rsplit('.',1)[0] + '.gif'
            print("Saving GIF to", gif_path)
            imageio.mimsave(gif_path, frames, fps=20)
            out_path = gif_path
            saved = True
        except Exception as e:
            print("GIF save failed:", e)

    if saved:
        return out_path
    else:
        return None

# Run with conservative parameters so it completes in this environment
output_path = run_and_save(N_RAYS=160, FRAMES=200, STEPS_PER_FRAME=6, out_path='/mnt/data/blackhole_rk45.mp4')

print("Done. Output file:", output_path)

