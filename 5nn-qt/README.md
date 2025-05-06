# Neural Network Water Potability Analyzer

A Rust-based water potability analysis tool using neural networks with multiple frontend options:
- Native Rust GUI with egui 
- Cross-platform Qt interface

## Features

- Machine learning model for predicting water potability
- Real-time visualization of training process
- Interactive water parameter input for prediction
- Visualized prediction results with confidence scores
- Cross-platform support

## Technical Details

This project demonstrates:
- Neural networks implementation in Rust
- Multi-threaded architecture for concurrent training and UI
- Two GUI implementations:
  - Native Rust GUI using egui/eframe
  - C++/Qt integration using Rust FFI bindings

## Requirements

### For egui frontend:
- Rust 2021 edition or newer
- Cargo package manager

### For Qt frontend:
- Qt 5.15.2 or newer
- CMake 3.10 or newer
- C++ compiler with C++17 support

## Building and Running

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/yourusername/water-potability-nn.git
cd water-potability-nn

# Set Qt environment variables (if using Qt frontend)
export QT_DIR=/path/to/qt/installation
```

### Building with egui frontend (default)

```bash
cargo build --release
cargo run --release
```

### Building with Qt frontend

```bash
# Build with Qt features enabled
cargo build --release --features qt_frontend --no-default-features

# Run Qt version
cargo run --release --bin water_potability_nn_qt
```

## Project Structure

```
.
├── src/                # Main Rust source code
│   ├── main.rs         # Entry point for egui version
│   └── lib.rs          # Core neural network implementation
├── qt_integration.rs   # Qt integration code
├── main_qt.rs          # Entry point for Qt version
├── build.rs            # Build script for Qt integration
├── CMakeLists.txt      # CMake configuration for Qt components
├── wrapper.h           # C++ header for Qt bindings
└── data/               # Training and test data
```

## License

MIT License

## Acknowledgments

- Qt and QCustomPlot for visualization components
- egui/eframe for native Rust GUI
- Contributors to the Rust-Qt binding projects 