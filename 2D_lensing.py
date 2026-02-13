import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
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
        
        # Navigation state
        self.offsetX = 0.0
        self.offsetY = 0.0
        self.zoom = 1.0
        self.middleMousePressed = False
        self.lastMouseX = 0.0
        self.lastMouseY = 0.0
        
        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        
        self.window = glfw.create_window(self.WIDTH, self.HEIGHT, "Black Hole Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.WIDTH, self.HEIGHT)
    
    def run(self):
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        left = -self.width + self.offsetX
        right = self.width + self.offsetX
        bottom = -self.height + self.offsetY
        top = self.height + self.offsetY
        
        glOrtho(left, right, bottom, top, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

class BlackHole:
    def __init__(self, position, mass):
        self.position = np.array(position)
        self.mass = mass
        self.r_s = 2.0 * G * mass / (c * c)  # Schwarzschild radius
    
    def draw(self):
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(1.0, 0.0, 0.0)  # Red color
        glVertex2f(0.0, 0.0)  # Center
        
        for i in range(101):
            angle = 2.0 * math.pi * i / 100
            x = self.r_s * math.cos(angle)
            y = self.r_s * math.sin(angle)
            glVertex2f(x, y)
        
        glEnd()

class Ray:
    def __init__(self, pos, dir_vec, black_hole):
        self.x = pos[0]
        self.y = pos[1]
        
        # Convert to polar coordinates
        self.r = math.sqrt(self.x**2 + self.y**2)
        self.phi = math.atan2(self.y, self.x)
        
        # Seed velocities
        self.dr = dir_vec[0] * math.cos(self.phi) + dir_vec[1] * math.sin(self.phi)
        self.dphi = (-dir_vec[0] * math.sin(self.phi) + dir_vec[1] * math.cos(self.phi)) / self.r
        
        # Store conserved quantities
        self.L = self.r * self.r * self.dphi
        f = 1.0 - black_hole.r_s / self.r
        dt_dlambda = math.sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dlambda
        
        # Start trail
        self.trail = [(self.x, self.y)]
    
    def step(self, dlambda, rs):
        if self.r <= rs:
            return  # Stop if inside event horizon
        
        rk4_step(self, dlambda, rs)
        
        # Convert back to cartesian
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)
        
        # Record trail
        self.trail.append((self.x, self.y))

def geodesic_rhs(ray, rs):
    r = float(ray.r)
    dr = float(ray.dr)
    dphi = float(ray.dphi)
    E = float(ray.E)
    rs_f = float(rs)
    
    f = 1.0 - rs_f / r
    dt_dlambda = E / f
    
    rhs = np.zeros(4)
    rhs[0] = dr  # dr/dλ
    rhs[1] = dphi  # dφ/dλ
    
    # d²r/dλ² from Schwarzschild null geodesic
    rhs[2] = (
        -(rs_f / (2.0 * r * r)) * f * (dt_dlambda * dt_dlambda) +
        (rs_f / (2.0 * r * r * f)) * (dr * dr) +
        (r - rs_f) * (dphi * dphi)
    )
    
    # d²φ/dλ²
    rhs[3] = -2.0 * dr * dphi / r
    
    return rhs

def rk4_step(ray, dlambda, rs):
    dlambda_f = float(dlambda)
    y0 = np.array([float(ray.r), float(ray.phi), float(ray.dr), float(ray.dphi)])
    
    k1 = geodesic_rhs(ray, rs)
    
    temp = y0 + k1 * (dlambda_f / 2.0)
    r2 = Ray.__new__(Ray)
    r2.r, r2.phi, r2.dr, r2.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r2.E = float(ray.E)
    k2 = geodesic_rhs(r2, rs)
    
    temp = y0 + k2 * (dlambda_f / 2.0)
    r3 = Ray.__new__(Ray)
    r3.r, r3.phi, r3.dr, r3.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r3.E = float(ray.E)
    k3 = geodesic_rhs(r3, rs)
    
    temp = y0 + k3 * dlambda_f
    r4 = Ray.__new__(Ray)
    r4.r, r4.phi, r4.dr, r4.dphi = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
    r4.E = float(ray.E)
    k4 = geodesic_rhs(r4, rs)
    
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
    SagA = BlackHole([0.0, 0.0, 0.0], 8.54e36)  # Sagittarius A*
    rays = []
    
    # Add multiple rays at different heights
    start_x = -1e11
    for i in range(8):
        # Vary the y-position to create rays at different heights
        y_offset = -3e10 + i * 1e10  # From -30 billion to +40 billion meters
        rays.append(Ray([start_x, y_offset], [c, 0.0], SagA))
    
    while not glfw.window_should_close(engine.window):
        engine.run()
        SagA.draw()
        
        for ray in rays:
            ray.step(1.0, SagA.r_s)
        
        draw_rays(rays)
        
        glfw.swap_buffers(engine.window)
        glfw.poll_events()
    
    glfw.terminate()

if __name__ == "__main__":
    main()