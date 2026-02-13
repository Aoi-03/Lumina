import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import math
from ctypes import c_float, c_int, sizeof, c_void_p
import struct

# Constants
c = 299792458.0
G = 6.67430e-11
SagA_mass = 8.54e36
SagA_rs = 2.0 * G * SagA_mass / (c * c)

class Camera:
    def __init__(self):
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = 6.34194e10
        self.minRadius = 1e10
        self.maxRadius = 1e12
        self.azimuth = 0.0
        self.elevation = math.pi / 2.0
        self.orbitSpeed = 0.01
        self.zoomSpeed = 25e9
        self.dragging = False
        self.moving = False
        self.lastX = 0.0
        self.lastY = 0.0
    
    def position(self):
        elevation = np.clip(self.elevation, 0.01, math.pi - 0.01)
        return np.array([
            self.radius * math.sin(elevation) * math.cos(self.azimuth),
            self.radius * math.cos(elevation),
            self.radius * math.sin(elevation) * math.sin(self.azimuth)
        ], dtype=np.float32)

# Object data
objects = [
    {'pos': [4e11, 0.0, 0.0], 'radius': 4e10, 'color': [1, 1, 0, 1], 'mass': 1.98892e30, 'velocity': [0, 0, 0]},
    {'pos': [0.0, 0.0, 4e11], 'radius': 4e10, 'color': [1, 0, 0, 1], 'mass': 1.98892e30, 'velocity': [0, 0, 0]},
    {'pos': [0.0, 0.0, 0.0], 'radius': SagA_rs, 'color': [0, 0, 0, 1], 'mass': SagA_mass, 'velocity': [0, 0, 0]},
]

gravity_enabled = False

class Engine:
    def __init__(self):
        self.WIDTH = 800
        self.HEIGHT = 600
        self.COMPUTE_WIDTH = 100  # Reduced from 200
        self.COMPUTE_HEIGHT = 75  # Reduced from 150
        
        if not glfw.init():
            raise Exception("GLFW init failed")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.WIDTH, self.HEIGHT, "Black Hole Simulator", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create window")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable vsync
        print(f"OpenGL {glGetString(GL_VERSION).decode()}")
        
        self.create_shaders()
        self.create_texture()
        self.create_quad()
        self.create_ubos()
        self.create_grid()
    
    def create_shaders(self):
        # Display shader
        vert_src = """
        #version 430 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }"""
        
        frag_src = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }"""
        
        self.display_program = shaders.compileProgram(
            shaders.compileShader(vert_src, GL_VERTEX_SHADER),
            shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
        )
        
        # Grid shader
        grid_vert = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 viewProj;
        void main() {
            gl_Position = viewProj * vec4(aPos, 1.0);
        }"""
        
        grid_frag = """
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.5, 0.5, 0.5, 0.7);
        }"""
        
        self.grid_program = shaders.compileProgram(
            shaders.compileShader(grid_vert, GL_VERTEX_SHADER),
            shaders.compileShader(grid_frag, GL_FRAGMENT_SHADER)
        )
        
        # Compute shader - optimized
        comp_src = f"""#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) writeonly uniform image2D outImage;
layout(std140, binding = 1) uniform Camera {{
    vec3 camPos; float _pad0;
    vec3 camRight; float _pad1;
    vec3 camUp; float _pad2;
    vec3 camForward; float _pad3;
    float tanHalfFov;
    float aspect;
    bool moving;
    int _pad4;
}} cam;
layout(std140, binding = 2) uniform Disk {{
    float disk_r1;
    float disk_r2;
    float disk_num;
    float thickness;
}};
layout(std140, binding = 3) uniform Objects {{
    int numObjects;
    vec4 objPosRadius[16];
    vec4 objColor[16];
    float mass[16];
}};
const float SagA_rs = {SagA_rs};
const float D_LAMBDA = 1e7;
const double ESCAPE_R = 1e30;
vec4 objectColor = vec4(0.0);
vec3 hitCenter = vec3(0.0);
float hitRadius = 0.0;
struct Ray {{
    float x, y, z, r, theta, phi;
    float dr, dtheta, dphi;
    float E, L;
}};
Ray initRay(vec3 pos, vec3 dir) {{
    Ray ray;
    ray.x = pos.x; ray.y = pos.y; ray.z = pos.z;
    ray.r = length(pos);
    ray.theta = acos(clamp(pos.z / ray.r, -1.0, 1.0));
    ray.phi = atan(pos.y, pos.x);
    float dx = dir.x, dy = dir.y, dz = dir.z;
    ray.dr = sin(ray.theta)*cos(ray.phi)*dx + sin(ray.theta)*sin(ray.phi)*dy + cos(ray.theta)*dz;
    ray.dtheta = (cos(ray.theta)*cos(ray.phi)*dx + cos(ray.theta)*sin(ray.phi)*dy - sin(ray.theta)*dz) / ray.r;
    ray.dphi = (-sin(ray.phi)*dx + cos(ray.phi)*dy) / (ray.r * max(sin(ray.theta), 0.001));
    ray.L = ray.r * ray.r * sin(ray.theta) * ray.dphi;
    float f = 1.0 - SagA_rs / ray.r;
    float dt_dL = sqrt((ray.dr*ray.dr)/f + ray.r*ray.r*(ray.dtheta*ray.dtheta + sin(ray.theta)*sin(ray.theta)*ray.dphi*ray.dphi));
    ray.E = f * dt_dL;
    return ray;
}}
bool intercept(Ray ray, float rs) {{ return ray.r <= rs; }}
bool interceptObject(Ray ray) {{
    vec3 P = vec3(ray.x, ray.y, ray.z);
    for (int i = 0; i < numObjects; ++i) {{
        vec3 center = objPosRadius[i].xyz;
        float radius = objPosRadius[i].w;
        if (distance(P, center) <= radius) {{
            objectColor = objColor[i];
            hitCenter = center;
            hitRadius = radius;
            return true;
        }}
    }}
    return false;
}}
void geodesicRHS(Ray ray, out vec3 d1, out vec3 d2) {{
    float r = ray.r, theta = ray.theta;
    float dr = ray.dr, dtheta = ray.dtheta, dphi = ray.dphi;
    float f = 1.0 - SagA_rs / r;
    float dt_dL = ray.E / f;
    d1 = vec3(dr, dtheta, dphi);
    d2.x = -(SagA_rs / (2.0*r*r)) * f * dt_dL * dt_dL + (SagA_rs / (2.0*r*r*f)) * dr*dr + r*(dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);
    d2.y = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;
    d2.z = -2.0*dr*dphi/r - 2.0*cos(theta)/(max(sin(theta), 0.001))*dtheta*dphi;
}}
void rk4Step(inout Ray ray, float dL) {{
    vec3 k1a, k1b;
    geodesicRHS(ray, k1a, k1b);
    ray.r += dL * k1a.x;
    ray.theta += dL * k1a.y;
    ray.phi += dL * k1a.z;
    ray.dr += dL * k1b.x;
    ray.dtheta += dL * k1b.y;
    ray.dphi += dL * k1b.z;
    ray.x = ray.r * sin(ray.theta) * cos(ray.phi);
    ray.y = ray.r * sin(ray.theta) * sin(ray.phi);
    ray.z = ray.r * cos(ray.theta);
}}
bool crossesEquatorialPlane(vec3 oldPos, vec3 newPos) {{
    bool crossed = (oldPos.y * newPos.y < 0.0);
    float r = length(vec2(newPos.x, newPos.z));
    return crossed && (r >= disk_r1 && r <= disk_r2);
}}
void main() {{
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imgSize = imageSize(outImage);
    if (pix.x >= imgSize.x || pix.y >= imgSize.y) return;
    float u = (2.0 * (pix.x + 0.5) / imgSize.x - 1.0) * cam.aspect * cam.tanHalfFov;
    float v = (1.0 - 2.0 * (pix.y + 0.5) / imgSize.y) * cam.tanHalfFov;
    vec3 dir = normalize(u * cam.camRight - v * cam.camUp + cam.camForward);
    Ray ray = initRay(cam.camPos, dir);
    vec4 color = vec4(0.0);
    vec3 prevPos = vec3(ray.x, ray.y, ray.z);
    bool hitBlackHole = false;
    bool hitDisk = false;
    bool hitObject = false;
    int maxSteps = cam.moving ? 5000 : 15000;
    for (int i = 0; i < maxSteps; ++i) {{
        if (intercept(ray, SagA_rs)) {{ hitBlackHole = true; break; }}
        rk4Step(ray, D_LAMBDA);
        vec3 newPos = vec3(ray.x, ray.y, ray.z);
        if (crossesEquatorialPlane(prevPos, newPos)) {{ hitDisk = true; break; }}
        if (interceptObject(ray)) {{ hitObject = true; break; }}
        prevPos = newPos;
        if (ray.r > ESCAPE_R) break;
    }}
    if (hitDisk) {{
        float r = length(vec3(ray.x, ray.y, ray.z)) / disk_r2;
        vec3 diskColor = vec3(1.0, r, 0.2);
        color = vec4(diskColor, r);
    }} else if (hitBlackHole) {{
        color = vec4(0.0, 0.0, 0.0, 1.0);
    }} else if (hitObject) {{
        vec3 P = vec3(ray.x, ray.y, ray.z);
        vec3 N = normalize(P - hitCenter);
        vec3 V = normalize(cam.camPos - P);
        float ambient = 0.1;
        float diff = max(dot(N, V), 0.0);
        float intensity = ambient + (1.0 - ambient) * diff;
        vec3 shaded = objectColor.rgb * intensity;
        color = vec4(shaded, objectColor.a);
    }} else {{
        color = vec4(0.0);
    }}
    imageStore(outImage, pix, color);
}}"""
        
        self.compute_program = shaders.compileProgram(
            shaders.compileShader(comp_src, GL_COMPUTE_SHADER)
        )
    
    def create_texture(self):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.COMPUTE_WIDTH, self.COMPUTE_HEIGHT,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    
    def create_quad(self):
        vertices = np.array([
            -1, 1, 0, 1,  -1, -1, 0, 0,  1, -1, 1, 0,
            -1, 1, 0, 1,   1, -1, 1, 0,  1,  1, 1, 1
        ], dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(8))
        glEnableVertexAttribArray(1)
    
    def create_grid(self):
        self.grid_vao = glGenVertexArrays(1)
        self.grid_vbo = glGenBuffers(1)
        self.grid_ebo = glGenBuffers(1)
        self.grid_needs_update = True
        self.update_grid()
    
    def update_grid(self):
        if not self.grid_needs_update:
            return
            
        gridSize = 20  # Reduced from 25
        spacing = 1.5e10  # Increased spacing
        vertices = []
        
        for z in range(gridSize + 1):
            for x in range(gridSize + 1):
                worldX = (x - gridSize // 2) * spacing
                worldZ = (z - gridSize // 2) * spacing
                y = 0.0
                
                for obj in objects:
                    mass = obj['mass']
                    r_s = 2.0 * G * mass / (c * c)
                    dx = worldX - obj['pos'][0]
                    dz = worldZ - obj['pos'][2]
                    dist = math.sqrt(dx**2 + dz**2)
                    
                    if dist > r_s:
                        deltaY = 2.0 * math.sqrt(r_s * (dist - r_s))
                        y += deltaY - 3e10
                    else:
                        y += 2.0 * math.sqrt(r_s * r_s) - 3e10
                
                vertices.extend([worldX, y, worldZ])
        
        indices = []
        for z in range(gridSize):
            for x in range(gridSize):
                i = z * (gridSize + 1) + x
                indices.extend([i, i + 1, i, i + gridSize + 1])
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        glBindVertexArray(self.grid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.grid_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, c_void_p(0))
        glEnableVertexAttribArray(0)
        
        self.grid_index_count = len(indices)
        self.grid_needs_update = False
    
    def create_ubos(self):
        self.camera_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 128, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.camera_ubo)
        
        self.disk_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 16, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.disk_ubo)
        
        self.objects_ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        glBufferData(GL_UNIFORM_BUFFER, 2048, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, self.objects_ubo)
    
    def upload_camera_ubo(self, camera):
        pos = camera.position()
        fwd = (camera.target - pos) / np.linalg.norm(camera.target - pos)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        
        data = bytearray(128)
        struct.pack_into('ffff', data, 0, *pos, 0)
        struct.pack_into('ffff', data, 16, *right, 0)
        struct.pack_into('ffff', data, 32, *up, 0)
        struct.pack_into('ffff', data, 48, *fwd, 0)
        struct.pack_into('ff', data, 64, math.tan(math.radians(30)), self.WIDTH / self.HEIGHT)
        struct.pack_into('i', data, 72, 1 if camera.moving else 0)
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.camera_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_disk_ubo(self):
        data = struct.pack('ffff', SagA_rs * 2.2, SagA_rs * 5.2, 2.0, 1e9)
        glBindBuffer(GL_UNIFORM_BUFFER, self.disk_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def upload_objects_ubo(self):
        data = bytearray(2048)
        struct.pack_into('i', data, 0, len(objects))
        for i, obj in enumerate(objects[:16]):
            offset = 16 + i * 16
            struct.pack_into('ffff', data, offset, *obj['pos'], obj['radius'])
        for i, obj in enumerate(objects[:16]):
            offset = 16 + 256 + i * 16
            struct.pack_into('ffff', data, offset, *obj['color'])
        for i, obj in enumerate(objects[:16]):
            offset = 16 + 512 + i * 4
            struct.pack_into('f', data, offset, obj['mass'])
        
        glBindBuffer(GL_UNIFORM_BUFFER, self.objects_ubo)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, len(data), data)
    
    def dispatch_compute(self, camera):
        glUseProgram(self.compute_program)
        self.upload_camera_ubo(camera)
        self.upload_disk_ubo()
        self.upload_objects_ubo()
        
        glBindImageTexture(0, self.texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)
        glDispatchCompute(math.ceil(self.COMPUTE_WIDTH / 16), math.ceil(self.COMPUTE_HEIGHT / 16), 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    
    def render(self):
        glUseProgram(self.display_program)
        glBindVertexArray(self.vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDisable(GL_DEPTH_TEST)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glEnable(GL_DEPTH_TEST)
    
    def draw_grid(self, camera):
        pos = camera.position()
        fwd = (camera.target - pos) / np.linalg.norm(camera.target - pos)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        view = np.eye(4, dtype=np.float32)
        z = -fwd
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        view[:3, :3] = np.column_stack([x, y, z])
        view[:3, 3] = -view[:3, :3] @ pos
        
        proj = np.zeros((4, 4), dtype=np.float32)
        fov = math.radians(60)
        aspect = self.WIDTH / self.HEIGHT
        f = 1.0 / math.tan(fov / 2)
        near, far = 1e9, 1e14
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1
        
        vp = proj @ view
        
        glUseProgram(self.grid_program)
        glUniformMatrix4fv(glGetUniformLocation(self.grid_program, "viewProj"), 1, GL_FALSE, vp.T)
        glBindVertexArray(self.grid_vao)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawElements(GL_LINES, self.grid_index_count, GL_UNSIGNED_INT, None)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

def main():
    global gravity_enabled
    engine = Engine()
    camera = Camera()
    
    def mouse_button_callback(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                camera.dragging = True
                camera.lastX, camera.lastY = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                camera.dragging = False
                camera.moving = False
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            global gravity_enabled
            gravity_enabled = (action == glfw.PRESS)
    
    def cursor_pos_callback(window, xpos, ypos):
        if camera.dragging:
            dx = xpos - camera.lastX
            dy = ypos - camera.lastY
            camera.azimuth += dx * camera.orbitSpeed
            camera.elevation -= dy * camera.orbitSpeed
            camera.elevation = np.clip(camera.elevation, 0.01, math.pi - 0.01)
            camera.lastX, camera.lastY = xpos, ypos
            camera.moving = True
    
    def scroll_callback(window, xoffset, yoffset):
        camera.radius -= yoffset * camera.zoomSpeed
        camera.radius = np.clip(camera.radius, camera.minRadius, camera.maxRadius)
    
    def key_callback(window, key, scancode, action, mods):
        global gravity_enabled
        if key == glfw.KEY_G and action == glfw.PRESS:
            gravity_enabled = not gravity_enabled
            print(f"Gravity {'ON' if gravity_enabled else 'OFF'}")
    
    glfw.set_mouse_button_callback(engine.window, mouse_button_callback)
    glfw.set_cursor_pos_callback(engine.window, cursor_pos_callback)
    glfw.set_scroll_callback(engine.window, scroll_callback)
    glfw.set_key_callback(engine.window, key_callback)
    
    frame_count = 0
    while not glfw.window_should_close(engine.window):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Gravity simulation - only update every 2 frames
        if gravity_enabled and frame_count % 2 == 0:
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i == j: continue
                    dx = obj2['pos'][0] - obj1['pos'][0]
                    dy = obj2['pos'][1] - obj1['pos'][1]
                    dz = obj2['pos'][2] - obj1['pos'][2]
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if dist > 0:
                        force = G * obj1['mass'] * obj2['mass'] / (dist**2)
                        acc = force / obj1['mass']
                        obj1['velocity'][0] += (dx / dist) * acc
                        obj1['velocity'][1] += (dy / dist) * acc
                        obj1['velocity'][2] += (dz / dist) * acc
                obj1['pos'][0] += obj1['velocity'][0]
                obj1['pos'][1] += obj1['velocity'][1]
                obj1['pos'][2] += obj1['velocity'][2]
            engine.grid_needs_update = True
        
        engine.update_grid()
        engine.dispatch_compute(camera)
        engine.render()
        engine.draw_grid(camera)
        
        glfw.swap_buffers(engine.window)
        glfw.poll_events()
        frame_count += 1
    
    glfw.terminate()

if __name__ == "__main__":
    main()