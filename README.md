# Rubik's Cube Solver Benchmark: Installation and Reproduction Guide

## Overview

This project implements and benchmarks two classical **Rubik's Cube** solving algorithms (**Thistlethwaite** and **Kociemba**) with comprehensive performance analysis. This guide provides complete instructions for environment setup, dependency installation, and result reproduction.

---

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version **3.8** or higher
- **Memory**: Minimum **4GB RAM** (8GB recommended)
- **Storage**: **2GB** free space for tables and results
- **Processor**: x86-64 architecture

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/cck181851/rubiks-cube-benchmark.git
cd cube-solver
```

### 2.  Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `seaborn>=0.11.0`
- `psutil>=5.8.0`

## Reproducing Benchmark Results

### 1. Run Comprehensive Benchmark
Execute the full benchmarking pipeline:

```bash
python benchmark.py
```

**This will:**
- Generate 200 test cubes across difficulty levels
- Execute both solvers on each cube
- Collect performance metrics (time, moves, memory, nodes expanded)
- Generate comprehensive analysis plots
- Produce statistical reports

## 2. Verification of Results

**Compare your results with expected performance characteristics:**

Expected Performance Ranges:
- Thistlethwaite: 20-35 moves, 0.01-0.05 seconds average
- Kociemba: 15-25 moves, 0.05-0.3 seconds average

Check the generated benchmark_results.txt for detailed statistics matching these ranges.
You can modify benchmark.py to adjust experimental parameters. In main execution block:
```
cubes = categorizer.generate_test_cubes(num_cubes=100)  # Reduce/increase test set
```

You can also customize visualizations in plot_comprehensive_analysis.py:
```
# Change color schemes
palette = sns.color_palette("Set2", n_colors=8)

# Modify figure sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
```

## Reproducibility Notes
- Results are deterministic with fixed random seed (3074742867)
- All timing uses wall-clock measurements for real-world performance
- Memory tracking uses tracemalloc for accurate peak usage
- System configuration is automatically logged for verification

## Support 
For issues with reproduction:
- Check the system_info section in benchmark_results.txt
- Verify all table files are present in tables/ directory
- Ensure Python environment matches requirements
- Consult the troubleshooting section above

## Acknowledgement

The implementations of Kociemba and Thistlethwaite Algorithm are based on [cube-solver](https://github.com/itsdaveba/cube-solver)
by Dave Barragan, licensed under the MIT License.

The original code have been modified and extended to suit the requirements of this graduation project.

*Last updated: October 25, 2025 | Compatible with Python 3.8-3.11*
