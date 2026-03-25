# LUMINA - Gravitational Lensing Simulator

A black hole simulation project featuring gravitational lensing, accretion disks, and spacetime curvature visualization using ray-tracing and GPU compute shaders

## Features

- Ray-tracing simulation of gravitational lensing around black holes
- Accretion disk visualization with realistic light bending
- Spacetime curvature demonstration using dynamic grids
- GPU-accelerated computation using OpenGL compute shaders
- Multiple simulation modes: 2D lensing, 3D visualization, Kerr black holes
- Interactive camera controls with orbit and zoom

## Demo

Watch the detailed explanation: [YouTube Video](https://www.youtube.com/watch?v=8-B6ryuBkCM)

## Project Structure

```
.
├── black_hole.cpp          # Main 3D black hole simulator (GPU-accelerated)
├── 2D_lensing.cpp          # 2D gravitational lensing (C++)
├── black_hole.py           # Python version with interactive controls
├── 2D_lensing.py           # 2D lensing simulation (Python)
├── 2D_lensing_2_body.py    # Two-body gravitational lensing
├── kerr.py                 # Kerr (rotating) black hole simulation
├── geodesic.comp           # GPU compute shader for geodesic calculations
├── grid.vert/frag          # Spacetime grid visualization shaders
├── CMakeLists.txt          # Build configuration
└── vcpkg.json              # C++ dependencies
```

## Building Requirements

### C++ Version

1. C++ Compiler supporting C++17 or newer
2. [CMake](https://cmake.org/) (version 3.21+)
3. [Vcpkg](https://vcpkg.io/en/) (for dependency management)
4. [Git](https://git-scm.com/)

### Python Version

- Python 3.8+
- See `requirements.txt` for dependencies

## Build Instructions

### C++ Build

1. Clone the repository:
   ```bash
   git clone https://github.com/Aoi-03/Lumina.git
   cd Lumina
   ```

2. Install dependencies with Vcpkg:
   ```bash
   vcpkg install
   ```

3. Get the vcpkg cmake toolchain file path:
   ```bash
   vcpkg integrate install
   ```
   This outputs: `CMake projects should use: "-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"`

4. Create a build directory:
   ```bash
   mkdir build
   ```

5. Configure project with CMake:
   ```bash
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
   ```

6. Build the project:
   ```bash
   cmake --build build
   ```

7. Run the executables from the build folder

### Alternative: Debian/Ubuntu

If you prefer native packages over vcpkg:

```bash
sudo apt update
sudo apt install build-essential cmake \
    libglew-dev libglfw3-dev libglm-dev libgl1-mesa-dev
cmake -B build -S .
cmake --build build
```

### Python Setup

```bash
pip install -r requirements.txt
python black_hole.py
```

## How It Works

### 2D Simulations
Simple ray-tracing through curved spacetime using Schwarzschild geodesic equations. Run `2D_lensing.cpp` or `2D_lensing.py` with dependencies installed.

### 3D Simulations
`black_hole.cpp` and `geodesic.comp` work together for GPU-accelerated simulation:
- CPU sends camera and object data via Uniform Buffer Objects (UBOs)
- GPU compute shader calculates geodesic paths using RK4 integration
- Results rendered in real-time with interactive controls

### Controls

- Left Mouse: Rotate camera
- Right Mouse / G key: Toggle gravity simulation
- Scroll: Zoom in/out

## Physics

The simulator uses:
- Schwarzschild metric for non-rotating black holes
- Kerr metric for rotating black holes (kerr.py)
- RK4 integration for geodesic equations
- Conserved quantities (Energy E, Angular momentum L)

## Data Files

The `data/` folder contains FITS astronomical images for testing gravitational lensing effects on real astronomical data.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Special thanks to everyone who checked out the project and the YouTube video explaining the implementation details.
