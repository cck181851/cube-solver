import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
import platform
import psutil
import sys
import os
from datetime import datetime
from src.cube_solver.cube.cube import Cube
from src.cube_solver.solver.thistlethwaite import Thistlethwaite
from src.cube_solver.solver.kociemba import Kociemba
from cube_difficulty_categorizer import CubeDifficultyCategorizer, CubeDifficulty

# store results
all_results = []

def create_comprehensive_plots(results, output_filename="plots"):
    """Create comprehensive plots for benchmark results."""
    from plot_comprehensive_analysis import PlotComprehensiveAnalysis

    if not os.path.exists(output_filename):
        os.makedirs(output_filename)

    plotter = PlotComprehensiveAnalysis(results, output_folder=output_filename)
    plotter.generate_all_plots()

# System information for reproducibility
def get_system_info():
    """Get comprehensive system information for reproducibility."""
    return {
        'timestamp': datetime.now().isoformat(),
        'hardware': f"{platform.processor()} ({psutil.virtual_memory().total / (1024**3):.1f}GB RAM)",
        'os': f"{platform.system()} {platform.release()}",
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'numpy_version': np.__version__,
        'matplotlib_version': plt.matplotlib.__version__,
        'random_seed': random.getstate()[1][0] if hasattr(random.getstate(), '__getitem__') else 'unknown',
        'timing_method': 'wall-clock time (time.perf_counter())',
        'memory_tracking': 'tracemalloc peak memory',
        'table_files': {
            'transition': 'tables/transition.npz',
            'pruning_thistlethwaite': 'tables/pruning_thistlethwaite.npz',
            'pruning_kociemba': 'tables/pruning_kociemba.npz'
        }
    }

def calculate_comprehensive_stats(data):
    """Calculate comprehensive statistics including medians, IQR, percentiles, and confidence intervals."""
    if not data or len(data) == 0:
        return {
            'N': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'iqr': np.nan, 'q1': np.nan, 'q3': np.nan,
            'p5': np.nan, 'p95': np.nan, 'p10': np.nan, 'p90': np.nan,
            'min': np.nan, 'max': np.nan,
            'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
            'ci_99_lower': np.nan, 'ci_99_upper': np.nan
        }
    
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        return {
            'N': 0, 'mean': np.nan, 'median': np.nan, 'std': np.nan,
            'iqr': np.nan, 'q1': np.nan, 'q3': np.nan,
            'p5': np.nan, 'p95': np.nan, 'p10': np.nan, 'p90': np.nan,
            'min': np.nan, 'max': np.nan,
            'ci_95_lower': np.nan, 'ci_95_upper': np.nan,
            'ci_99_lower': np.nan, 'ci_99_upper': np.nan
        }
    
    # Basic statistics
    stats = {
        'N': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    # Quartiles and IQR
    stats['q1'] = np.percentile(data, 25)
    stats['q3'] = np.percentile(data, 75)
    stats['iqr'] = stats['q3'] - stats['q1']
    
    # Percentiles
    stats['p5'] = np.percentile(data, 5)
    stats['p10'] = np.percentile(data, 10)
    stats['p90'] = np.percentile(data, 90)
    stats['p95'] = np.percentile(data, 95)
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    stats['ci_95_lower'] = np.percentile(bootstrap_means, 2.5)
    stats['ci_95_upper'] = np.percentile(bootstrap_means, 97.5)
    stats['ci_99_lower'] = np.percentile(bootstrap_means, 0.5)
    stats['ci_99_upper'] = np.percentile(bootstrap_means, 99.5)
    
    return stats

def run_solver(solver, cube: Cube, name: str, per_phase=False, optimal=False):
    print(f"\n--- {name} ---")

    tracemalloc.start()
    start_time = time.perf_counter()

    if per_phase:
        solution_parts = solver.solve(cube, verbose=2, optimal=optimal)
        if not solution_parts:
            print("Solver returned nothing!")
            solution = ""
        else:
            solution = " ".join(solution_parts)
    else:
        solution = solver.solve(cube, optimal=optimal)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    moves = solution.split()
    move_count = len(moves)
    elapsed = end_time - start_time
    mem_peak = peak / 1024

    print(f"Solution: {solution}")
    print(f"Move count: {move_count}")
    print(f"Time taken: {elapsed:.4f} sec")
    print(f"Memory peak: {mem_peak:.2f} KB")

    all_results.append({
        "solver": name,
        "moves": move_count,
        "time": elapsed,
        "memory": mem_peak,
    })

def run_solver_quiet(solver, cube: Cube, name: str, per_phase=False, optimal=False):
    """Run solver without printing detailed output and collect comprehensive metrics."""
    tracemalloc.start()
    start_time = time.perf_counter()
    cpu_start = time.process_time()

    if per_phase:
        solution_parts = solver.solve(cube, verbose=2, optimal=optimal)
        if not solution_parts:
            solution = ""
        else:
            solution = " ".join(solution_parts)
    else:
        solution = solver.solve(cube, optimal=optimal)

    end_time = time.perf_counter()
    cpu_end = time.process_time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    moves = solution.split()
    move_count = len(moves)
    elapsed_wall = end_time - start_time
    elapsed_cpu = cpu_end - cpu_start
    mem_peak = peak / 1024

    # Collect solver-specific metrics
    nodes_expanded = sum(solver.nodes) if hasattr(solver, 'nodes') else 0
    table_lookups = sum(solver.checks) if hasattr(solver, 'checks') else 0
    pruned_nodes = sum(solver.prunes) if hasattr(solver, 'prunes') else 0

    result = {
        "solver": name,
        "moves": move_count,
        "time": elapsed_wall,
        "time_cpu": elapsed_cpu,
        "memory": mem_peak,
        "nodes_expanded": nodes_expanded,
        "table_lookups": table_lookups,
        "pruned_nodes": pruned_nodes,
        "solution": solution,
        "success": solution != "" and move_count > 0
    }

    all_results.append(result)
    return result

def run_test_quiet(cube_difficulty: CubeDifficulty):
    """Run benchmark test quietly and return results."""
    cube = cube_difficulty.cube
    
    thistlethwaite_result = run_solver_quiet(Thistlethwaite(), cube.copy(), "Thistlethwaite", per_phase=True)
    kociemba_result = run_solver_quiet(Kociemba(), cube.copy(), "Kociemba", per_phase=True)
    
    return {
        'cube_info': cube_difficulty,
        'thistlethwaite': thistlethwaite_result,
        'kociemba': kociemba_result
    }

def group_results_by_difficulty_and_metric(results):
    """Group results by difficulty level and metric, merging easy and very easy groups."""
    grouped = {}
    
    for result in results:
        cube_info = result['cube_info']
        
        # Get difficulty categories for each metric
        manhattan_cat = cube_info.manhattan_category
        hamming_cat = cube_info.hamming_category
        orientation_cat = cube_info.orientation_category
        solution_cat = cube_info.solution_category
        
        # Merge easy and very easy groups
        if manhattan_cat == "very easy":
            manhattan_cat = "easy"
        if hamming_cat == "very easy":
            hamming_cat = "easy"
        if orientation_cat == "very easy":
            orientation_cat = "easy"
        if solution_cat == "very easy":
            solution_cat = "easy"
        
        # Group by Manhattan distance difficulty
        if manhattan_cat not in grouped:
            grouped[manhattan_cat] = {}
        if 'manhattan' not in grouped[manhattan_cat]:
            grouped[manhattan_cat]['manhattan'] = []
        grouped[manhattan_cat]['manhattan'].append(result)
        
        # Group by Hamming distance difficulty
        if hamming_cat not in grouped:
            grouped[hamming_cat] = {}
        if 'hamming' not in grouped[hamming_cat]:
            grouped[hamming_cat]['hamming'] = []
        grouped[hamming_cat]['hamming'].append(result)
        
        # Group by Orientation distance difficulty
        if orientation_cat not in grouped:
            grouped[orientation_cat] = {}
        if 'orientation' not in grouped[orientation_cat]:
            grouped[orientation_cat]['orientation'] = []
        grouped[orientation_cat]['orientation'].append(result)
        
        # Group by Solution length difficulty
        if solution_cat not in grouped:
            grouped[solution_cat] = {}
        if 'solution' not in grouped[solution_cat]:
            grouped[solution_cat]['solution'] = []
        grouped[solution_cat]['solution'].append(result)
    
    return grouped

def calculate_statistics(data):
    """Calculate comprehensive statistics for a dataset."""
    return calculate_comprehensive_stats(data)

def calculate_group_statistics(results):
    """Calculate comprehensive statistics for a group of results."""
    if not results:
        return {
            'thistlethwaite': {'moves': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                              'time': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                              'memory': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}},
            'kociemba': {'moves': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                        'time': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                        'memory': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}}
        }
    
    thistlethwaite_results = [r['thistlethwaite'] for r in results]
    kociemba_results = [r['kociemba'] for r in results]
    
    return {
        'thistlethwaite': {
            'moves': calculate_statistics([r['moves'] for r in thistlethwaite_results]),
            'time': calculate_statistics([r['time'] for r in thistlethwaite_results]),
            'memory': calculate_statistics([r['memory'] for r in thistlethwaite_results])
        },
        'kociemba': {
            'moves': calculate_statistics([r['moves'] for r in kociemba_results]),
            'time': calculate_statistics([r['time'] for r in kociemba_results]),
            'memory': calculate_statistics([r['memory'] for r in kociemba_results])
        }
    }

def print_group_statistics(grouped_results, output_file):
    """Print comprehensive statistics for each difficulty-metric combination to file."""
    output_file.write("\n" + "="*100 + "\n")
    output_file.write("DETAILED STATISTICS BY DIFFICULTY LEVEL AND METRIC\n")
    output_file.write("="*100 + "\n")
    
    for difficulty in sorted(grouped_results.keys()):
        output_file.write(f"\n{difficulty.upper()} DIFFICULTY LEVEL\n")
        output_file.write("-" * 50 + "\n")
        
        for metric_name in sorted(grouped_results[difficulty].keys()):
            results = grouped_results[difficulty][metric_name]
            if not results:
                continue
            
            stats = calculate_group_statistics(results)
            num_tests = len(results)
            
            output_file.write(f"\n{metric_name.title()} Distance Metric ({num_tests} tests):\n")
            output_file.write("  Thistlethwaite Solver:\n")
            output_file.write(f"    Moves:    Mean={stats['thistlethwaite']['moves']['mean']:.2f}, "
                             f"Median={stats['thistlethwaite']['moves']['median']:.2f}, "
                             f"Min={stats['thistlethwaite']['moves']['min']:.2f}, "
                             f"Max={stats['thistlethwaite']['moves']['max']:.2f}, "
                             f"Std={stats['thistlethwaite']['moves']['std']:.2f}\n")
            output_file.write(f"    Time:     Mean={stats['thistlethwaite']['time']['mean']:.4f}, "
                             f"Median={stats['thistlethwaite']['time']['median']:.4f}, "
                             f"Min={stats['thistlethwaite']['time']['min']:.4f}, "
                             f"Max={stats['thistlethwaite']['time']['max']:.4f}, "
                             f"Std={stats['thistlethwaite']['time']['std']:.4f}\n")
            output_file.write(f"    Memory:   Mean={stats['thistlethwaite']['memory']['mean']:.2f}, "
                             f"Median={stats['thistlethwaite']['memory']['median']:.2f}, "
                             f"Min={stats['thistlethwaite']['memory']['min']:.2f}, "
                             f"Max={stats['thistlethwaite']['memory']['max']:.2f}, "
                             f"Std={stats['thistlethwaite']['memory']['std']:.2f}\n")
            
            output_file.write("  Kociemba Solver:\n")
            output_file.write(f"    Moves:    Mean={stats['kociemba']['moves']['mean']:.2f}, "
                             f"Median={stats['kociemba']['moves']['median']:.2f}, "
                             f"Min={stats['kociemba']['moves']['min']:.2f}, "
                             f"Max={stats['kociemba']['moves']['max']:.2f}, "
                             f"Std={stats['kociemba']['moves']['std']:.2f}\n")
            output_file.write(f"    Time:     Mean={stats['kociemba']['time']['mean']:.4f}, "
                             f"Median={stats['kociemba']['time']['median']:.4f}, "
                             f"Min={stats['kociemba']['time']['min']:.4f}, "
                             f"Max={stats['kociemba']['time']['max']:.4f}, "
                             f"Std={stats['kociemba']['time']['std']:.4f}\n")
            output_file.write(f"    Memory:   Mean={stats['kociemba']['memory']['mean']:.2f}, "
                             f"Median={stats['kociemba']['memory']['median']:.2f}, "
                             f"Min={stats['kociemba']['memory']['min']:.2f}, "
                             f"Max={stats['kociemba']['memory']['max']:.2f}, "
                             f"Std={stats['kociemba']['memory']['std']:.2f}\n")

def print_overall_statistics(grouped_results, output_file):
    """Print overall statistics across all groups for each difficulty level to file."""
    output_file.write("\n" + "="*100 + "\n")
    output_file.write("OVERALL STATISTICS BY DIFFICULTY LEVEL (ACROSS ALL METRICS)\n")
    output_file.write("="*100 + "\n")
    
    for difficulty in sorted(grouped_results.keys()):
        output_file.write(f"\n{difficulty.upper()} DIFFICULTY LEVEL\n")
        output_file.write("-" * 50 + "\n")
        
        # Collect all results for this difficulty level across all metrics
        all_results_for_difficulty = []
        for metric_name in grouped_results[difficulty].keys():
            all_results_for_difficulty.extend(grouped_results[difficulty][metric_name])
        
        if not all_results_for_difficulty:
            continue
        
        stats = calculate_group_statistics(all_results_for_difficulty)
        total_tests = len(all_results_for_difficulty)
        
        output_file.write(f"Total tests across all metrics: {total_tests}\n")
        output_file.write("\n  Thistlethwaite Solver:\n")
        output_file.write(f"    Moves:    Mean={stats['thistlethwaite']['moves']['mean']:.2f}, "
                         f"Median={stats['thistlethwaite']['moves']['median']:.2f}, "
                         f"Min={stats['thistlethwaite']['moves']['min']:.2f}, "
                         f"Max={stats['thistlethwaite']['moves']['max']:.2f}, "
                         f"Std={stats['thistlethwaite']['moves']['std']:.2f}\n")
        output_file.write(f"    Time:     Mean={stats['thistlethwaite']['time']['mean']:.4f}, "
                         f"Median={stats['thistlethwaite']['time']['median']:.4f}, "
                         f"Min={stats['thistlethwaite']['time']['min']:.4f}, "
                         f"Max={stats['thistlethwaite']['time']['max']:.4f}, "
                         f"Std={stats['thistlethwaite']['time']['std']:.4f}\n")
        output_file.write(f"    Memory:   Mean={stats['thistlethwaite']['memory']['mean']:.2f}, "
                         f"Median={stats['thistlethwaite']['memory']['median']:.2f}, "
                         f"Min={stats['thistlethwaite']['memory']['min']:.2f}, "
                         f"Max={stats['thistlethwaite']['memory']['max']:.2f}, "
                         f"Std={stats['thistlethwaite']['memory']['std']:.2f}\n")
        
        output_file.write("\n  Kociemba Solver:\n")
        output_file.write(f"    Moves:    Mean={stats['kociemba']['moves']['mean']:.2f}, "
                         f"Median={stats['kociemba']['moves']['median']:.2f}, "
                         f"Min={stats['kociemba']['moves']['min']:.2f}, "
                         f"Max={stats['kociemba']['moves']['max']:.2f}, "
                         f"Std={stats['kociemba']['moves']['std']:.2f}\n")
        output_file.write(f"    Time:     Mean={stats['kociemba']['time']['mean']:.4f}, "
                         f"Median={stats['kociemba']['time']['median']:.4f}, "
                         f"Min={stats['kociemba']['time']['min']:.4f}, "
                         f"Max={stats['kociemba']['time']['max']:.4f}, "
                         f"Std={stats['kociemba']['time']['std']:.4f}\n")
        output_file.write(f"    Memory:   Mean={stats['kociemba']['memory']['mean']:.2f}, "
                         f"Median={stats['kociemba']['memory']['median']:.2f}, "
                         f"Min={stats['kociemba']['memory']['min']:.2f}, "
                         f"Max={stats['kociemba']['memory']['max']:.2f}, "
                         f"Std={stats['kociemba']['memory']['std']:.2f}\n")


def save_comprehensive_results(results, grouped_results, output_filename="benchmark_results.txt"):
    """Save comprehensive results with system information and detailed statistics."""
    system_info = get_system_info()
    
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        # Write system information header
        output_file.write("="*100 + "\n")
        output_file.write("CUBE SOLVER BENCHMARK WITH COMPREHENSIVE ANALYSIS\n")
        output_file.write("="*100 + "\n")
        output_file.write(f"Generated on: {system_info['timestamp']}\n")
        output_file.write(f"Hardware: {system_info['hardware']}\n")
        output_file.write(f"OS: {system_info['os']}\n")
        output_file.write(f"Python version: {system_info['python_version']}\n")
        output_file.write(f"NumPy version: {system_info['numpy_version']}\n")
        output_file.write(f"Matplotlib version: {system_info['matplotlib_version']}\n")
        output_file.write(f"Random seed: {system_info['random_seed']}\n")
        output_file.write(f"Timing method: {system_info['timing_method']}\n")
        output_file.write(f"Memory tracking: {system_info['memory_tracking']}\n")
        output_file.write(f"Table files: {system_info['table_files']}\n")
        output_file.write(f"Total cubes analyzed: {len(results)}\n\n")
        
        # Write detailed statistics for each group
        print("Calculating and writing comprehensive statistics...")
        print_group_statistics(grouped_results, output_file)
        
        # Write overall statistics by difficulty level
        print("Calculating and writing overall statistics...")
        print_overall_statistics(grouped_results, output_file)
        
        # Write comprehensive statistics summary
        output_file.write(f"\n{'='*100}\n")
        output_file.write("COMPREHENSIVE STATISTICS SUMMARY\n")
        output_file.write(f"{'='*100}\n")
        
        # Calculate comprehensive stats for each solver
        thistlethwaite_results = [r['thistlethwaite'] for r in results]
        kociemba_results = [r['kociemba'] for r in results]
        
        # Thistlethwaite comprehensive stats
        thistle_times = [r['time'] for r in thistlethwaite_results]
        thistle_moves = [r['moves'] for r in thistlethwaite_results]
        thistle_memory = [r['memory'] for r in thistlethwaite_results]
        
        output_file.write(f"\nTHISTLETHWAITE SOLVER STATISTICS:\n")
        output_file.write(f"Runtime (seconds):\n")
        time_stats = calculate_comprehensive_stats(thistle_times)
        output_file.write(f"  N: {time_stats['N']}, Mean: {time_stats['mean']:.4f}, Median: {time_stats['median']:.4f}\n")
        output_file.write(f"  Std: {time_stats['std']:.4f}, IQR: {time_stats['iqr']:.4f}\n")
        output_file.write(f"  5th-95th percentile: {time_stats['p5']:.4f} - {time_stats['p95']:.4f}\n")
        output_file.write(f"  95% CI: [{time_stats['ci_95_lower']:.4f}, {time_stats['ci_95_upper']:.4f}]\n")
        
        output_file.write(f"Solution Length (moves):\n")
        moves_stats = calculate_comprehensive_stats(thistle_moves)
        output_file.write(f"  N: {moves_stats['N']}, Mean: {moves_stats['mean']:.2f}, Median: {moves_stats['median']:.2f}\n")
        output_file.write(f"  Std: {moves_stats['std']:.2f}, IQR: {moves_stats['iqr']:.2f}\n")
        output_file.write(f"  5th-95th percentile: {moves_stats['p5']:.2f} - {moves_stats['p95']:.2f}\n")
        output_file.write(f"  95% CI: [{moves_stats['ci_95_lower']:.2f}, {moves_stats['ci_95_upper']:.2f}]\n")
        
        # Kociemba comprehensive stats
        koci_times = [r['time'] for r in kociemba_results]
        koci_moves = [r['moves'] for r in kociemba_results]
        koci_memory = [r['memory'] for r in kociemba_results]
        
        output_file.write(f"\nKOCIEMBA SOLVER STATISTICS:\n")
        output_file.write(f"Runtime (seconds):\n")
        time_stats = calculate_comprehensive_stats(koci_times)
        output_file.write(f"  N: {time_stats['N']}, Mean: {time_stats['mean']:.4f}, Median: {time_stats['median']:.4f}\n")
        output_file.write(f"  Std: {time_stats['std']:.4f}, IQR: {time_stats['iqr']:.4f}\n")
        output_file.write(f"  5th-95th percentile: {time_stats['p5']:.4f} - {time_stats['p95']:.4f}\n")
        output_file.write(f"  95% CI: [{time_stats['ci_95_lower']:.4f}, {time_stats['ci_95_upper']:.4f}]\n")
        
        output_file.write(f"Solution Length (moves):\n")
        moves_stats = calculate_comprehensive_stats(koci_moves)
        output_file.write(f"  N: {moves_stats['N']}, Mean: {moves_stats['mean']:.2f}, Median: {moves_stats['median']:.2f}\n")
        output_file.write(f"  Std: {moves_stats['std']:.2f}, IQR: {moves_stats['iqr']:.2f}\n")
        output_file.write(f"  5th-95th percentile: {moves_stats['p5']:.2f} - {moves_stats['p95']:.2f}\n")
        output_file.write(f"  95% CI: [{moves_stats['ci_95_lower']:.2f}, {moves_stats['ci_95_upper']:.2f}]\n")
        
        # Write comprehensive plot information
        output_file.write(f"\n{'='*100}\n")
        output_file.write("COMPREHENSIVE PLOT ANALYSIS\n")
        output_file.write(f"{'='*100}\n")
        
        output_file.write("\nThe following 8 comprehensive plots have been generated in the 'plots' folder:\n\n")
        
        output_file.write("1. GROUPED RUNTIME BOXPLOT\n")
        output_file.write("   File: plots/fig_grouped_runtime_boxplot.png\n")
        output_file.write("   Purpose: Compare runtime distributions of solvers across difficulties in a single view\n")
        output_file.write("   Caption: 'Runtime distribution by solver and difficulty. Median and IQR shown; outliers visualized.'\n")
        output_file.write("   Interpretation: Shorter boxes = more consistent runtime. Lower median = faster. Heavy tails upward indicate rare slow cases.\n\n")
        
        output_file.write("2. MOVES VS RUNTIME SCATTER WITH DIFFICULTY\n")
        output_file.write("   File: plots/fig_moves_vs_runtime_scatter_difficulty.png\n")
        output_file.write("   Purpose: Show the trade-off: does lower move count come at higher runtime? Identify outliers.\n")
        output_file.write("   Caption: 'Moves vs runtime by solver and difficulty. Lower & left is better. Different colors/markers indicate difficulty levels.'\n")
        output_file.write("   Interpretation: Downward sloping clusters → longer runtimes yield shorter solutions. Overlap shows efficiency trade-off.\n\n")
        
        output_file.write("3. SOLUTION LENGTH HISTOGRAM BY DIFFICULTY\n")
        output_file.write("   File: plots/fig_solution_length_histogram_difficulty.png\n")
        output_file.write("   Purpose: Compare how often each solver returns shorter solutions across difficulties\n")
        output_file.write("   Caption: 'Distribution of solution lengths by solver and difficulty — shows typical solution lengths.'\n")
        output_file.write("   Interpretation: Left-shifted distribution = shorter solutions. Overlap shows efficiency trade-off.\n\n")
        
        output_file.write("4. MEDIAN RUNTIME VS SCRAMBLE LENGTH\n")
        output_file.write("   File: plots/fig_median_runtime_vs_scramble_length.png\n")
        output_file.write("   Purpose: Show how runtime scales with scramble length (simple 'difficulty vs time' curve)\n")
        output_file.write("   Caption: 'Median runtime by scramble length for each solver; shows how difficulty affects time.'\n")
        output_file.write("   Interpretation: Steeper slope = more sensitive to difficulty. Parallel lines = similar scaling behavior.\n\n")
        
        output_file.write("5. MOVES BY SOLVER AND DIFFICULTY BOXPLOT\n")
        output_file.write("   File: plots/fig_moves_by_solver_difficulty_boxplot.png\n")
        output_file.write("   Purpose: Compare typical solution length (quality) across difficulty and solvers\n")
        output_file.write("   Caption: 'Solution length distribution by solver and difficulty (median & IQR).'\n")
        output_file.write("   Interpretation: Lower median = shorter solutions. Smaller IQR = more consistent solution quality.\n\n")
        
        output_file.write("6. FRACTION SOLVED CDF\n")
        output_file.write("   File: plots/fig_fraction_solved_cdf.png\n")
        output_file.write("   Purpose: Show what fraction of cases each solver finishes within a given time. Reveals heavy-tail behavior.\n")
        output_file.write("   Caption: 'Fraction of instances solved within time t per solver — useful to choose timeouts.'\n")
        output_file.write("   Interpretation: Steeper curve = more consistent performance. Right shift = slower overall. Flat sections near 1.0 show timeouts.\n\n")
        
        output_file.write("7. SUMMARY BARS WITH ERROR\n")
        output_file.write("   File: plots/fig_summary_bars_with_error.png\n")
        output_file.write("   Purpose: Compact summary plot showing mean runtime and mean moves with error bars per solver × difficulty\n")
        output_file.write("   Caption: 'Mean runtime and solution length by solver and difficulty with standard deviation error bars.'\n")
        output_file.write("   Interpretation: Lower bars = better performance. Error bars show variability. Left plot = speed, right plot = quality.\n\n")
        
        output_file.write("8. AVERAGED PLOTS (BAR CHART AND LINE PLOT)\n")
        output_file.write("   File: plots/fig_averaged_plots.png\n")
        output_file.write("   Purpose: Show average moves and computation time by solver across difficulty levels\n")
        output_file.write("   Caption: 'Average moves required and computation time by solver across difficulty levels.'\n")
        output_file.write("   Interpretation: Left plot shows solution efficiency (lower bars = shorter solutions). Right plot shows speed (lower lines = faster).\n\n")
        
        output_file.write("PLOT GENERATION SUMMARY:\n")
        output_file.write("- All plots saved as PNG files in the 'plots' folder\n")
        output_file.write("- Plots use log scales where appropriate for better visualization\n")
        output_file.write("- Color coding: Blue = Thistlethwaite, Red = Kociemba\n")
        output_file.write("- Professional styling with grids, transparency, and proper font sizes\n")
        output_file.write("- Each plot includes comprehensive captions and interpretation guidelines\n\n")
        
        output_file.write(f"\n{'='*100}\n")
        output_file.write("BENCHMARK COMPLETE\n")
        output_file.write(f"{'='*100}\n")

if __name__ == "__main__":
    print("="*80)
    print("CUBE SOLVER BENCHMARK WITH DIFFICULTY ANALYSIS")
    print("="*80)
    
    # Generate cube configurations using the difficulty categorizer
    print("Generating cube configurations with difficulty analysis...")
    categorizer = CubeDifficultyCategorizer()
    cubes = categorizer.generate_test_cubes(num_cubes=20)  # Increased for better statistics
    
    # Run benchmarks
    print("Running benchmarks on generated cubes...")
    results = []
    for i, cube_difficulty in enumerate(cubes, 1):
        print(f"Processing cube {i}/{len(cubes)}...")
        result = run_test_quiet(cube_difficulty)
        results.append(result)

    # Save per-instance CSV for reproducibility / paired analysis
    try:
        import pandas as pd
    except Exception:
        pd = None

    import csv
    import math

    def _safe_get(obj, attr_names):
        if obj is None:
            return None
        for name in attr_names:
            if hasattr(obj, name):
                try:
                    val = getattr(obj, name)
                    # call if callable and returns something
                    if callable(val):
                        try:
                            out = val()
                        except TypeError:
                            out = val  # leave the callable itself if it needs args
                    else:
                        out = val
                    if out is None:
                        continue
                    # convert numpy scalars to python
                    try:
                        if hasattr(out, 'item'):
                            out = out.item()
                    except Exception:
                        pass
                    return out
                except Exception:
                    continue
        # fallback to string conversion
        try:
            return str(obj)
        except Exception:
            return None

    def save_per_instance_csv(results, filename="results_per_instance.csv"):
        rows = []
        for i, r in enumerate(results, start=1):
            ci = r.get('cube_info', {}) or {}
            cube_obj = getattr(ci, 'cube', None) if ci is not None else None

            # Try to extract a compact cube representation / scramble if present
            cube_repr = _safe_get(ci, ['scramble', 'scramble_str', 'scramble_string'])
            if cube_repr is None:
                cube_repr = _safe_get(cube_obj, ['to_facelets', 'facelets', 'state', '__str__'])
            # Difficulty categories 
            manhattan_cat = getattr(ci, 'manhattan_category', None) or getattr(ci, 'manhattan', None)
            hamming_cat = getattr(ci, 'hamming_category', None) or getattr(ci, 'hamming', None)
            orientation_cat = getattr(ci, 'orientation_category', None) or getattr(ci, 'orientation', None)
            solution_cat = getattr(ci, 'solution_category', None) or getattr(ci, 'solution', None)

            th = r.get('thistlethwaite', {}) or {}
            koc = r.get('kociemba', {}) or {}

            # build row with solver-prefixed columns
            row = {
                "instance_id": i,
                "cube_repr": cube_repr,
                "manhattan_cat": manhattan_cat,
                "hamming_cat": hamming_cat,
                "orientation_cat": orientation_cat,
                "solution_cat": solution_cat,
                # Thistlethwaite
                "th_moves": th.get("moves", math.nan),
                "th_time": th.get("time", math.nan),
                "th_time_cpu": th.get("time_cpu", math.nan),
                "th_memory_kb": th.get("memory", math.nan),
                "th_nodes_expanded": th.get("nodes_expanded", math.nan),
                "th_table_lookups": th.get("table_lookups", math.nan),
                "th_pruned_nodes": th.get("pruned_nodes", math.nan),
                "th_success": th.get("success", None),
                "th_solution": th.get("solution", None),
                # Kociemba
                "koc_moves": koc.get("moves", math.nan),
                "koc_time": koc.get("time", math.nan),
                "koc_time_cpu": koc.get("time_cpu", math.nan),
                "koc_memory_kb": koc.get("memory", math.nan),
                "koc_nodes_expanded": koc.get("nodes_expanded", math.nan),
                "koc_table_lookups": koc.get("table_lookups", math.nan),
                "koc_pruned_nodes": koc.get("pruned_nodes", math.nan),
                "koc_success": koc.get("success", None),
                "koc_solution": koc.get("solution", None),
            }

            rows.append(row)

        # Try pandas first, otherwise fallback to csv module
        if pd is not None:
            try:
                df = pd.DataFrame(rows)                
                # df['th_solution'] = df['th_solution'].apply(lambda s: s if s is None else s[:200])
                # df['koc_solution'] = df['koc_solution'].apply(lambda s: s if s is None else s[:200])
                df.to_csv(filename, index=False, encoding='utf-8')
                print(f"Per-instance CSV saved to {filename} (pandas). Rows: {len(df)}")
                return
            except Exception as e:
                print("Pandas save failed, falling back to csv module:", e)

        # Fallback CSV writer
        if rows:
            keys = list(rows[0].keys())
        else:
            keys = []
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in rows:
                    # convert non-serializables
                    safe_row = {}
                    for k, v in r.items():
                        try:
                            # pandas/numpy scalars
                            if hasattr(v, 'item'):
                                v = v.item()
                            # ensure string-safe
                            if v is None:
                                safe_row[k] = ""
                            else:
                                safe_row[k] = v
                        except Exception:
                            safe_row[k] = str(v)
                    writer.writerow(safe_row)
            print(f"Per-instance CSV saved to {filename} (csv fallback). Rows: {len(rows)}")
        except Exception as e:
            print("Failed to write per-instance CSV:", e)

    # saves to project root
    save_per_instance_csv(results, filename="results_per_instance.csv")    
    # print(results)

    # Group results by difficulty and metric
    print("Grouping results by difficulty level and metric...")
    grouped_results = group_results_by_difficulty_and_metric(results)

    create_comprehensive_plots(results, output_filename="plots")
    
    # Save comprehensive results
    output_filename = "benchmark_results.txt"
    print(f"Writing comprehensive results to {output_filename}...")
    save_comprehensive_results(results, grouped_results, output_filename)
    
    
    print(f"Results saved to: {output_filename}")
    print(f"{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
