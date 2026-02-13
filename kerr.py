import glfw
from OpenGL.GL import * # type: ignore
from OpenGL.GLU import * # type: ignore
import numpy as np
import math

# Constants
c = 299792458.0  # Speed of light
G = 6.67430e-11  # Gravitational constant

class Engine:
    def __init__(self):
        self.WIDTH = 800
        self.HEIGHT = 600
        self.width = 100000000000.0  # Width of viewport in meters
        self.height = 75000000000.0  # Height of viewport in meters
        
        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        
        self.window = glfw.create_window(self.WIDTH, self.HEIGHT, "Kerr Black Hole Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.WIDTH, self.HEIGHT)
    
    def run(self):
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        left = -self.width
        right = self.width
        bottom = -self.height
        top = self.height
        
        glOrtho(left, right, bottom, top, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

class KerrBlackHole:
    def __init__(self, position, mass, spin):
        self.position = np.array(position)
        self.mass = mass
        self.spin = spin  # Spin parameter a (0 to 1, where 1 is maximal)
        
        # Schwarzschild radius
        self.r_s = 2.0 * G * mass / (c * c)
        
        # Kerr parameters
        self.a = spin * self.r_s / 2.0  # Angular momentum per unit mass
        
        # Event horizons (outer and inner)
        self.r_outer = self.r_s / 2.0 * (1.0 + math.sqrt(1.0 - spin**2))
        self.r_inner = self.r_s / 2.0 * (1.0 - math.sqrt(1.0 - spin**2))
        
        # Ergosphere (stationary limit)
        self.r_ergo = self.r_s / 2.0 * (1.0 + math.sqrt(1.0 - spin**2 * 0.0))  # At equator
    
    def draw(self):
        # Draw outer event horizon (red)
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(1.0, 0.0, 0.0)
        glVertex2f(0.0, 0.0)
        for i in range(101):
            angle = 2.0 * math.pi * i / 100
            x = self.r_outer * math.cos(angle)
            y = self.r_outer * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
        
        # Draw ergosphere boundary (yellow circle)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glColor3f(1.0, 1.0, 0.0)
        for i in range(100):
            angle = 2.0 * math.pi * i / 100
            x = self.r_s * math.cos(angle)
            y = self.r_s * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
    
    def draw_spacetime_grid(self, time):
        # Draw rotating spacetime grid to show frame dragging
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        
        # Draw multiple circles at different radii showing frame dragging
        for ring in range(3, 8):
            radius = ring * self.r_s / 2.0
            
            # Calculate frame dragging rotation at this radius
            omega = (2.0 * self.a * self.r_s * radius) / ((radius * radius + self.a * self.a) ** 2)
            rotation = omega * time * 50.0  # Amplify for visibility
            
            # Stronger effect closer to black hole
            alpha = 0.3 * (1.0 - radius / (8.0 * self.r_s))
            
            glBegin(GL_LINE_STRIP)
            glColor4f(0.0, 0.5, 1.0, alpha)
            
            # Draw spiral pattern showing rotation
            num_points = 100
            for i in range(num_points + 1):
                angle = 2.0 * math.pi * i / num_points + rotation
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                glVertex2f(x, y)
            glEnd()
            
            # Draw radial lines showing twist
            num_spokes = 8
            for spoke in range(num_spokes):
                angle = 2.0 * math.pi * spoke / num_spokes + rotation
                glBegin(GL_LINES)
                glColor4f(0.0, 0.5, 1.0, alpha * 0.5)
                glVertex2f(self.r_s * math.cos(angle), self.r_s * math.sin(angle))
                glVertex2f(radius * math.cos(angle), radius * math.sin(angle))
                glEnd()
        
        glDisable(GL_BLEND)

class Ray:
    def __init__(self, pos, dir_vec, black_hole):
        self.x = pos[0]
        self.y = pos[1]
        self.black_hole = black_hole
        
        # Convert to polar coordinates
        self.r = math.sqrt(self.x**2 + self.y**2)
        self.phi = math.atan2(self.y, self.x)
        
        # Seed velocities
        self.dr = dir_vec[0] * math.cos(self.phi) + dir_vec[1] * math.sin(self.phi)
        self.dphi = (-dir_vec[0] * math.sin(self.phi) + dir_vec[1] * math.cos(self.phi)) / self.r
        
        # Store conserved quantities (approximate for Kerr)
        self.L = self.r * self.r * self.dphi
        f = 1.0 - black_hole.r_s / self.r
        dt_dlambda = math.sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dlambda
        
        # Start trail
        self.trail = [(self.x, self.y)]
    
    def step(self, dlambda):
        if self.r <= self.black_hole.r_outer:
            return  # Stop if inside outer event horizon
        
        kerr_rk4_step(self, dlambda, self.black_hole)
        
        # Convert back to cartesian
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)
        
        # Record trail
        self.trail.append((self.x, self.y))

def kerr_geodesic_rhs(ray, bh):
    r = float(ray.r)
    dr = float(ray.dr)
    dphi = float(ray.dphi)
    E = float(ray.E)
    L = float(ray.L)
    
    rs = bh.r_s
    a = bh.a
    
    # Kerr metric components (simplified 2D equatorial plane)
    rho2 = r * r + a * a
    delta = r * r - rs * r + a * a
    
    # Effective potential terms
    f = 1.0 - rs / r
    
    # Frame dragging effect - adds rotation to dphi
    omega = (2.0 * a * rs * r) / (rho2 * rho2)  # Frame dragging angular velocity
    
    rhs = np.zeros(4)
    rhs[0] = dr  # dr/dλ
    rhs[1] = dphi  # dφ/dλ
    
    # Modified geodesic equations for Kerr (approximate)
    # Frame dragging adds extra phi velocity
    frame_drag_factor = omega * 0.5
    
    # d²r/dλ² - modified by rotation
    dt_dlambda = E / f if f > 0.01 else E / 0.01
    
    rhs[2] = (
        -(rs / (2.0 * r * r)) * f * (dt_dlambda * dt_dlambda) +
        (rs / (2.0 * r * r * f)) * (dr * dr) +
        (r - rs) * (dphi * dphi) +
        frame_drag_factor * (dphi * dphi)  # Frame dragging term
    )
    
    # d²φ/dλ² - modified by frame dragging
    rhs[3] = -2.0 * dr * dphi / r + frame_drag_factor * dr / r
    
    return rhs

def kerr_rk4_step(ray, dlambda, bh):
    dlambda_f = float(dlambda)
    y0 = np.array([float(ray.r), float(ray.phi), float(ray.dr), float(ray.dphi)])
    
    k1 = kerr_geodesic_rhs(ray, bh)
    
    temp = y0 + k1 * (dlambda_f / 2.0)
    r2 = Ray.__new__(Ray)
    r2.r, r2.phi, r2.dr, r2.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r2.E = float(ray.E)
    r2.L = float(ray.L)
    k2 = kerr_geodesic_rhs(r2, bh)
    
    temp = y0 + k2 * (dlambda_f / 2.0)
    r3 = Ray.__new__(Ray)
    r3.r, r3.phi, r3.dr, r3.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r3.E = float(ray.E)
    r3.L = float(ray.L)
    k3 = kerr_geodesic_rhs(r3, bh)
    
    temp = y0 + k3 * dlambda_f
    r4 = Ray.__new__(Ray)
    r4.r, r4.phi, r4.dr, r4.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r4.E = float(ray.E)
    r4.L = float(ray.L)
    k4 = kerr_geodesic_rhs(r4, bh)
    
    ray.r += float((dlambda_f / 6.0) * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]))
    ray.phi += float((dlambda_f / 6.0) * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]))
    ray.dr += float((dlambda_f / 6.0) * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]))
    ray.dphi += float((dlambda_f / 6.0) * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]))

def draw_rays(rays):
    # Draw current ray positions as points
    glPointSize(2.0)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    for ray in rays:
        glVertex2f(ray.x, ray.y)
    glEnd()
    
    # Enable blending for trails
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    
    # Draw each trail with fading alpha
    for ray in rays:
        N = len(ray.trail)
        if N < 2:
            continue
        
        glBegin(GL_LINE_STRIP)
        for i, point in enumerate(ray.trail):
            alpha = i / (N - 1)
            glColor4f(1.0, 1.0, 1.0, max(alpha, 0.05))
            glVertex2f(point[0], point[1])
        glEnd()
    
    glDisable(GL_BLEND)

def main():
    engine = Engine()
    
    # Create a Kerr black hole with spin parameter 0.8 (fast rotation)
    SagA = KerrBlackHole([0.0, 0.0, 0.0], 8.54e36, 0.8)
    rays = []
    
    # Add multiple rays at different heights
    start_x = -1e11
    for i in range(8):
        y_offset = -3e10 + i * 1e10
        rays.append(Ray([start_x, y_offset], [c, 0.0], SagA))
    
    time = 0.0
    while not glfw.window_should_close(engine.window):
        engine.run()
        
        # Draw rotating spacetime grid first (background)
        SagA.draw_spacetime_grid(time)
        
        # Draw black hole on top
        SagA.draw()
        
        for ray in rays:
            ray.step(1.0)
        
        draw_rays(rays)
        
        glfw.swap_buffers(engine.window)
        glfw.poll_events()
        
        time += 0.01
    
    glfw.terminate()

if __name__ == "__main__":
    main()