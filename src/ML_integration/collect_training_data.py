import time
import tracemalloc
import random
import pandas as pd
import sys
import os
from datetime import datetime
from src.cube_solver.cube.cube import Cube
from src.cube_solver.solver.thistlethwaite import Thistlethwaite
from src.cube_solver.solver.kociemba import Kociemba
from src.ML_integration.cube_difficulty_categorizer import CubeDifficultyCategorizer

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class TrainingDataCollector:
    """Collects training data by running solvers on random cubes."""
    
    def __init__(self, output_file = "ML_integration_data/training_data.csv"):
        self.output_file = output_file
        self.categorizer = CubeDifficultyCategorizer()
        self.solver_thistle = Thistlethwaite()
        self.solver_kociemba = Kociemba()
        
        # Initialize data collection
        self.data = []
        
    def generate_random_cubes(self, num_cubes = 100, max_scramble_length = 30):
        """Generate random cubes with varying difficulty."""
        cubes = []
        
        print(f"Generating {num_cubes} random cubes...")
        
        for i in range(num_cubes):
            # Vary scramble length to get different difficulty levels
            scramble_length = random.randint(5, max_scramble_length)
            scramble = self.categorizer.generate_scramble(scramble_length)
            
            # Create cube and analyze difficulty
            cube = Cube(scramble)
            cube_analysis = self.categorizer.analyze_cube(scramble)
            
            cubes.append({
                'cube_id': i + 1,
                'scramble': scramble,
                'scramble_length': scramble_length,
                'cube': cube,
                'analysis': cube_analysis
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_cubes} cubes...")
        
        return cubes
    
    def run_solver_with_metrics(self, solver, cube, solver_name):
        """Run a solver and collect comprehensive metrics."""
        metrics = {
            'solver': solver_name,
            'success': False,
            'solution': '',
            'moves': 0,
            'time_wall': 0.0,
            'time_cpu': 0.0,
            'memory_kb': 0.0,
            'nodes_expanded': 0,
            'table_lookups': 0,
            'pruned_nodes': 0,
            'error': None
        }
        
        try:
            # Reset solver statistics before run
            if hasattr(solver, 'nodes'):
                solver.nodes = [0] * 4 if hasattr(solver, 'num_phases') else 0
            if hasattr(solver, 'checks'):
                solver.checks = [0] * 4 if hasattr(solver, 'num_phases') else 0
            if hasattr(solver, 'prunes'):
                solver.prunes = [0] * 4 if hasattr(solver, 'num_phases') else 0
            
            # Start tracking
            tracemalloc.start()
            start_time_wall = time.perf_counter()
            start_time_cpu = time.process_time()
            
            # Run solver
            solution = solver.solve(cube, optimal=False)
            
            # Stop tracking
            end_time_wall = time.perf_counter()
            end_time_cpu = time.process_time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Extract moves
            if solution:
                moves = solution.split() if isinstance(solution, str) else solution
                move_count = len(moves)
            else:
                moves = []
                move_count = 0
            
            # Collect metrics
            metrics.update({
                'success': solution is not None and move_count > 0,
                'solution': solution if isinstance(solution, str) else ' '.join(solution),
                'moves': move_count,
                'time_wall': end_time_wall - start_time_wall,
                'time_cpu': end_time_cpu - start_time_cpu,
                'memory_kb': peak / 1024,  # Convert to KB
            })
            
            # Collect solver-specific metrics if available
            if hasattr(solver, 'nodes'):
                try:
                    nodes = solver.nodes
                    if isinstance(nodes, (list, tuple)):
                        metrics['nodes_expanded'] = sum(nodes)
                    else:
                        metrics['nodes_expanded'] = nodes
                except:
                    metrics['nodes_expanded'] = 0
                    
            if hasattr(solver, 'checks'):
                try:
                    checks = solver.checks
                    if isinstance(checks, (list, tuple)):
                        metrics['table_lookups'] = sum(checks)
                    else:
                        metrics['table_lookups'] = checks
                except:
                    metrics['table_lookups'] = 0
                    
            if hasattr(solver, 'prunes'):
                try:
                    prunes = solver.prunes
                    if isinstance(prunes, (list, tuple)):
                        metrics['pruned_nodes'] = sum(prunes)
                    else:
                        metrics['pruned_nodes'] = prunes
                except:
                    metrics['pruned_nodes'] = 0
                
        except Exception as e:
            metrics['error'] = str(e)
            tracemalloc.stop()  # Ensure memory tracking is stopped
        
        return metrics
    
    def collect_cube_metrics(self, cube_data):
        """Collect all metrics for a single cube."""
        cube = cube_data['cube']
        analysis = cube_data['analysis']
        
        # Get cube coordinates
        try:
            coords = cube.get_coords(partial_corner_perm=False, partial_edge_perm=False)
            partial_coords = cube.get_coords(partial_corner_perm=True, partial_edge_perm=True)
        except Exception as e:
            print(f"Error getting coordinates for cube {cube_data['cube_id']}: {e}")
            coords = (0, 0, 0, 0)
            partial_coords = (0, 0, (0, 0), (0, 0, 0))
        
        # Extract difficulty features
        difficulty_features = {
            'cube_id': cube_data['cube_id'],
            'scramble': cube_data['scramble'],
            'scramble_length': cube_data['scramble_length'],
            
            # Raw distances
            'manhattan_distance': analysis.manhattan_distance,
            'hamming_total': analysis.hamming_distance['total_hamming'],
            'hamming_corners': analysis.hamming_distance['corner_hamming'],
            'hamming_edges': analysis.hamming_distance['edge_hamming'],
            'orientation_total': analysis.orientation_distance['total_orientation'],
            'orientation_corners': analysis.orientation_distance['corner_orientation'],
            'orientation_edges': analysis.orientation_distance['edge_orientation'],
            
            # Cycle decomposition metrics
            'corner_cycle_count': analysis.cycle_metrics['corner_cycle_count'],
            'edge_cycle_count': analysis.cycle_metrics['edge_cycle_count'],
            'total_swap_cost': analysis.cycle_metrics['total_swap_cost'],
            'corner_swap_cost': analysis.cycle_metrics['corner_swap_cost'],
            'edge_swap_cost': analysis.cycle_metrics['edge_swap_cost'],
            'max_corner_cycle': analysis.cycle_metrics['max_corner_cycle'],
            'max_edge_cycle': analysis.cycle_metrics['max_edge_cycle'],
            'avg_corner_cycle': analysis.cycle_metrics['avg_corner_cycle'],
            'avg_edge_cycle': analysis.cycle_metrics['avg_edge_cycle'],
            
            # Thistlethwaite subgroup distances
            'dist_to_G1': analysis.thistlethwaite_distances['dist_to_G1'],
            'dist_to_G2': analysis.thistlethwaite_distances['dist_to_G2'],
            'dist_to_G3': analysis.thistlethwaite_distances['dist_to_G3'],
            'bad_edge_orientation': analysis.thistlethwaite_distances['bad_edge_orientation'],
            'bad_corner_orientation': analysis.thistlethwaite_distances['bad_corner_orientation'],
            'bad_edge_slice': analysis.thistlethwaite_distances['bad_edge_slice'],
            'bad_corner_tetrad': analysis.thistlethwaite_distances['bad_corner_tetrad'],
            'corner_parity': analysis.thistlethwaite_distances['corner_parity'],
            
            # Numerical scores
            'manhattan_score': analysis.manhattan_score,
            'hamming_score': analysis.hamming_score,
            'orientation_score': analysis.orientation_score,
            'solution_score': analysis.solution_score,
            'overall_score': analysis.overall_score,
            
            # Categories
            'manhattan_category': analysis.manhattan_category,
            'hamming_category': analysis.hamming_category,
            'orientation_category': analysis.orientation_category,
            'solution_category': analysis.solution_category,
            'overall_category': analysis.overall_category,
            
            # Cube coordinates
            'co': coords[0],
            'eo': coords[1],
            'cp': coords[2],
            'ep': coords[3],
            
            # Partial coordinates (for Thistlethwaite)
            'pcp_0': partial_coords[2][0] if isinstance(partial_coords[2], tuple) else partial_coords[2],
            'pcp_1': partial_coords[2][1] if isinstance(partial_coords[2], tuple) and len(partial_coords[2]) > 1 else 0,
            'pep_0': partial_coords[3][0] if isinstance(partial_coords[3], tuple) else partial_coords[3],
            'pep_1': partial_coords[3][1] if isinstance(partial_coords[3], tuple) and len(partial_coords[3]) > 1 else 0,
            'pep_2': partial_coords[3][2] if isinstance(partial_coords[3], tuple) and len(partial_coords[3]) > 2 else 0,
            
            # Additional metrics
            'estimated_solution_length': analysis.estimated_solution_length,
            'solution_variance': analysis.solution_variance,
        }
        
        # Add face uniformity metrics
        for face, uniformity in analysis.face_uniformity.items():
            difficulty_features[f'uniformity_{face}'] = uniformity
        
        # Add color clustering metrics
        for face, clustering in analysis.color_clustering.items():
            difficulty_features[f'clusters_{face}'] = clustering['num_clusters']
            difficulty_features[f'avg_cluster_{face}'] = clustering['avg_cluster_size']
            difficulty_features[f'max_cluster_{face}'] = clustering['max_cluster_size']
        
        return difficulty_features
    
    def run_experiment(self, num_cubes: int = 100):
        """Run the complete experiment collecting data from both solvers."""
        
        # Generate random cubes
        cubes = self.generate_random_cubes(num_cubes)
        
        print(f"\nRunning experiments on {num_cubes} cubes...")
        print("=" * 80)
        
        successful_runs = 0
        failed_runs = 0
        
        for i, cube_data in enumerate(cubes):
            cube_id = cube_data['cube_id']
            cube = cube_data['cube']
            
            print(f"Cube {cube_id}/{num_cubes}: {cube_data['scramble']}")
            
            # Collect cube difficulty metrics
            cube_metrics = self.collect_cube_metrics(cube_data)
            
            # Run Thistlethwaite solver
            # print(f"  Running Thistlethwaite...", end='')
            thistle_metrics = self.run_solver_with_metrics(self.solver_thistle, cube.copy(), "Thistlethwaite")
            
            if thistle_metrics['success']:
                # print(f"({thistle_metrics['moves']} moves, {thistle_metrics['time_wall']:.2f}s, {thistle_metrics['nodes_expanded']} nodes)")
                successful_runs += 1
            else:
                # print(f"(failed: {thistle_metrics.get('error', 'unknown')})")
                failed_runs += 1
            
            # Run Kociemba solver
            # print(f"Running Kociemba...", end='')
            kociemba_metrics = self.run_solver_with_metrics(self.solver_kociemba, cube.copy(), "Kociemba")
            
            if kociemba_metrics['success']:
                # print(f"({kociemba_metrics['moves']} moves, {kociemba_metrics['time_wall']:.2f}s, {kociemba_metrics['nodes_expanded']} nodes)")
                successful_runs += 1
            else:
                # print(f"(failed: {kociemba_metrics.get('error', 'unknown')})")
                failed_runs += 1
            
            # Combine metrics for this cube
            combined_record = cube_metrics.copy()
            
            # Add Thistlethwaite solver metrics
            for key, value in thistle_metrics.items():
                if key not in ['solver', 'solution']:  # Don't overwrite cube metrics
                    combined_record[f'th_{key}'] = value
            
            # Add Kociemba solver metrics
            for key, value in kociemba_metrics.items():
                if key not in ['solver', 'solution']:
                    combined_record[f'koc_{key}'] = value
            
            # Add solution strings (truncated for CSV)
            combined_record['th_solution'] = thistle_metrics.get('solution', '')[:200]
            combined_record['koc_solution'] = kociemba_metrics.get('solution', '')[:200]
            
            # Store the record
            self.data.append(combined_record)
        
        print(f"\nExperiment complete!")
        print(f"  Total runs: {len(cubes) * 2}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {failed_runs}")
        
        # Save final results all at once
        if self.data:
            self.save_to_csv(self.output_file)
        else:
            print("No data collected to save!")
        
        return self.data
    
    def save_to_csv(self, filename = None):
        """Save collected data to CSV file."""
        if filename is None:
            filename = self.output_file
        
        if not self.data:
            print("No data to save!")
            return
        
        df = pd.DataFrame(self.data)
        
        # Reorder columns for better readability
        preferred_order = [
            'cube_id', 'scramble', 'scramble_length',
            'manhattan_distance', 'hamming_total', 'orientation_total',
            'total_swap_cost', 'corner_swap_cost', 'edge_swap_cost',  # NEW
            'dist_to_G1', 'dist_to_G2', 'dist_to_G3',  # NEW
            'manhattan_score', 'hamming_score', 'orientation_score', 'overall_score',
            'overall_category',
            'th_success', 'th_moves', 'th_time_wall', 'th_nodes_expanded',
            'koc_success', 'koc_moves', 'koc_time_wall', 'koc_nodes_expanded',
        ]
        
        # Get existing columns and reorder
        existing_columns = df.columns.tolist()
        ordered_columns = [col for col in preferred_order if col in existing_columns]
        remaining_columns = [col for col in existing_columns if col not in ordered_columns]
        final_columns = ordered_columns + remaining_columns
        
        df = df[final_columns]
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename} ({len(df)} records)")
        
        # Also save a summary statistics file
        summary_file = filename.replace('.csv', '_summary.txt')
        self.save_summary_statistics(df, summary_file)
        
        # Save a smaller sample for quick inspection
        sample_file = filename.replace('.csv', '_sample.csv')
        sample_df = df.head(10)  # First 10 records
        sample_df.to_csv(sample_file, index=False)
        print(f"Sample saved to {sample_file} (10 records)")
    
    def save_summary_statistics(self, df, filename):
        """Save summary statistics to a text file."""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING DATA SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total cubes: {len(df)}\n")
            f.write(f"Date collected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Cube difficulty statistics
            f.write("CUBE DIFFICULTY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            for metric in ['manhattan_distance', 'hamming_total', 'orientation_total', 
                          'total_swap_cost', 'corner_swap_cost', 'edge_swap_cost',  # NEW
                          'dist_to_G1', 'dist_to_G2', 'dist_to_G3',  # NEW
                          'manhattan_score', 'hamming_score', 'orientation_score', 'overall_score']:
                if metric in df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Min: {df[metric].min():.2f}\n")
                    f.write(f"  Max: {df[metric].max():.2f}\n")
                    f.write(f"  Mean: {df[metric].mean():.2f}\n")
                    f.write(f"  Std: {df[metric].std():.2f}\n")
            
            # Category distribution
            if 'overall_category' in df.columns:
                f.write("\nDIFFICULTY CATEGORY DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                category_counts = df['overall_category'].value_counts()
                for category, count in category_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {category}: {count} cubes ({percentage:.1f}%)\n")
            
            # Solver performance statistics
            f.write("\nSOLVER PERFORMANCE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            solvers = ['th', 'koc']
            metrics = ['success', 'moves', 'time_wall', 'nodes_expanded']
            
            for solver in solvers:
                f.write(f"\n{solver.upper()} Solver:\n")
                
                success_rate = df[f'{solver}_success'].mean() * 100 if f'{solver}_success' in df.columns else 0
                f.write(f"Success rate: {success_rate:.1f}%\n")
                
                for metric in metrics:
                    col_name = f'{solver}_{metric}'
                    if col_name in df.columns:
                        if metric == 'success':
                            continue  # Already handled
                        f.write(f"  {metric}:\n")
                        f.write(f" - Min: {df[col_name].min():.2f}\n")
                        f.write(f" - Max: {df[col_name].max():.2f}\n")
                        f.write(f" - Mean: {df[col_name].mean():.2f}\n")
                        f.write(f" - Std: {df[col_name].std():.2f}\n")
            
            # Correlation analysis
            f.write("\nCORRELATION ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Correlate cycle metrics with Kociemba performance
            cycle_metrics = ['total_swap_cost', 'corner_swap_cost', 'edge_swap_cost',
                           'max_corner_cycle', 'max_edge_cycle']
            performance_metrics = ['th_moves', 'th_time_wall', 'th_nodes_expanded', 
                                  'koc_moves', 'koc_time_wall', 'koc_nodes_expanded']
            
            f.write("\nCycle Metrics vs Performance:\n")
            for cycle_metric in cycle_metrics:
                if cycle_metric in df.columns:
                    f.write(f"\n{cycle_metric} correlations:\n")
                    for perf_metric in performance_metrics:
                        if perf_metric in df.columns:
                            correlation = df[cycle_metric].corr(df[perf_metric])
                            f.write(f"  with {perf_metric}: {correlation:.3f}\n")
            
            # Correlate Thistlethwaite subgroup distances with performance
            subgroup_metrics = ['dist_to_G1', 'dist_to_G2', 'dist_to_G3']
            f.write("\nSubgroup Distances vs Performance:\n")
            for subgroup_metric in subgroup_metrics:
                if subgroup_metric in df.columns:
                    f.write(f"\n{subgroup_metric} correlations:\n")
                    for perf_metric in performance_metrics:
                        if perf_metric in df.columns:
                            correlation = df[subgroup_metric].corr(df[perf_metric])
                            f.write(f"  with {perf_metric}: {correlation:.3f}\n")
            
            # Additional useful statistics for ML
            f.write("\nADDITIONAL STATISTICS FOR ML TRAINING:\n")
            f.write("-" * 40 + "\n")
            
            if 'th_time_wall' in df.columns and 'koc_time_wall' in df.columns:
                faster_solver_counts = {
                    'Thistlethwaite faster': (df['th_time_wall'] < df['koc_time_wall']).sum(),
                    'Kociemba faster': (df['koc_time_wall'] < df['th_time_wall']).sum(),
                    'Equal': (df['th_time_wall'] == df['koc_time_wall']).sum()
                }
                
                f.write("Faster Solver Comparison:\n")
                for comparison, count in faster_solver_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {comparison}: {count} cubes ({percentage:.1f}%)\n")
            
            if 'th_moves' in df.columns and 'koc_moves' in df.columns:
                shorter_solution_counts = {
                    'Thistlethwaite shorter': (df['th_moves'] < df['koc_moves']).sum(),
                    'Kociemba shorter': (df['koc_moves'] < df['th_moves']).sum(),
                    'Equal length': (df['th_moves'] == df['koc_moves']).sum()
                }
                
                f.write("\nShorter Solution Comparison:\n")
                for comparison, count in shorter_solution_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {comparison}: {count} cubes ({percentage:.1f}%)\n")
            
            if 'th_nodes_expanded' in df.columns and 'koc_nodes_expanded' in df.columns:
                fewer_nodes_counts = {
                    'Thistlethwaite fewer nodes': (df['th_nodes_expanded'] < df['koc_nodes_expanded']).sum(),
                    'Kociemba fewer nodes': (df['koc_nodes_expanded'] < df['th_nodes_expanded']).sum(),
                    'Equal nodes': (df['th_nodes_expanded'] == df['koc_nodes_expanded']).sum()
                }
                
                f.write("\nFewer Nodes Expanded Comparison:\n")
                for comparison, count in fewer_nodes_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {comparison}: {count} cubes ({percentage:.1f}%)\n")
    
    def analyze_collected_data(self, data_file = None):
        """Analyze the collected data and generate insights."""
        if data_file is None:
            data_file = self.output_file
        
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found!")
            return
        
        df = pd.read_csv(data_file)
        
        print("\n" + "=" * 80)
        print("DATA COLLECTION ANALYSIS")
        print("=" * 80)
        
        print(f"\nTotal records: {len(df)}")
        
        # Check success rates
        print("\nSuccess Rates:")
        if 'th_success' in df.columns:
            th_success = df['th_success'].mean() * 100
            print(f"  Thistlethwaite: {th_success:.1f}%")
        
        if 'koc_success' in df.columns:
            koc_success = df['koc_success'].mean() * 100
            print(f"  Kociemba: {koc_success:.1f}%")
        
        # Compare solution lengths
        print("\nSolution Length Comparison:")
        if 'th_moves' in df.columns and 'koc_moves' in df.columns:
            avg_th_moves = df['th_moves'].mean()
            avg_koc_moves = df['koc_moves'].mean()
            print(f" - Thistlethwaite avg moves: {avg_th_moves:.1f}")
            print(f" - Kociemba avg moves: {avg_koc_moves:.1f}")
            print(f" - Difference: {avg_th_moves - avg_koc_moves:.1f} moves")
        
        # Runtime comparison
        print("\nRuntime Comparison (seconds):")
        if 'th_time_wall' in df.columns and 'koc_time_wall' in df.columns:
            avg_th_time = df['th_time_wall'].mean()
            avg_koc_time = df['koc_time_wall'].mean()
            print(f"  Thistlethwaite avg time: {avg_th_time:.3f}s")
            print(f"  Kociemba avg time: {avg_koc_time:.3f}s")
            print(f"  Speed ratio: {avg_th_time / avg_koc_time if avg_koc_time > 0 else 'N/A'}:1")
        
        # Nodes expanded comparison
        print("\nNodes Expanded Comparison:")
        if 'th_nodes_expanded' in df.columns and 'koc_nodes_expanded' in df.columns:
            avg_th_nodes = df['th_nodes_expanded'].mean()
            avg_koc_nodes = df['koc_nodes_expanded'].mean()
            print(f" - Thistlethwaite avg nodes: {avg_th_nodes:.0f}")
            print(f" - Kociemba avg nodes: {avg_koc_nodes:.0f}")
            print(f" - Node ratio: {avg_koc_nodes / avg_th_nodes if avg_th_nodes > 0 else 'N/A'}:1")
        
        # Cycle decomposition statistics
        if 'total_swap_cost' in df.columns:
            print("\nCycle Decomposition Statistics:")
            print(f"  Average swap cost: {df['total_swap_cost'].mean():.2f}")
            print(f"  Max swap cost: {df['total_swap_cost'].max():.2f}")
            print(f"  Min swap cost: {df['total_swap_cost'].min():.2f}")
        
        # Subgroup distances statistics
        if 'dist_to_G1' in df.columns:
            print("\nThistlethwaite Subgroup Distances:")
            print(f"  Avg dist to G1: {df['dist_to_G1'].mean():.2f}")
            print(f"  Avg dist to G2: {df['dist_to_G2'].mean():.2f}")
            print(f"  Avg dist to G3: {df['dist_to_G3'].mean():.2f}")
        
        # Correlation analysis for new metrics
        print("\nCorrelation of New Metrics with Kociemba Time:")
        if 'total_swap_cost' in df.columns and 'koc_time_wall' in df.columns:
            corr = df['total_swap_cost'].corr(df['koc_time_wall'])
            print(f"  Swap cost vs Kociemba time: {corr:.3f}")
        
        if 'max_corner_cycle' in df.columns and 'koc_time_wall' in df.columns:
            corr = df['max_corner_cycle'].corr(df['koc_time_wall'])
            print(f"  Max corner cycle vs Kociemba time: {corr:.3f}")
        
        if 'dist_to_G1' in df.columns and 'th_time_wall' in df.columns:
            corr = df['dist_to_G1'].corr(df['th_time_wall'])
            print(f"  Dist to G1 vs Thistlethwaite time: {corr:.3f}")
        
        # Difficulty distribution
        if 'overall_category' in df.columns:
            print("\nDifficulty Distribution:")
            category_counts = df['overall_category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {category}: {count} cubes ({percentage:.1f}%)")
        
        # Which solver is faster?
        if 'th_time_wall' in df.columns and 'koc_time_wall' in df.columns:
            print("\nWhich solver is faster?")
            th_faster = (df['th_time_wall'] < df['koc_time_wall']).sum()
            koc_faster = (df['koc_time_wall'] < df['th_time_wall']).sum()
            equal = (df['th_time_wall'] == df['koc_time_wall']).sum()
            
            total = len(df)
            print(f"  Thistlethwaite faster: {th_faster} cubes ({th_faster/total*100:.1f}%)")
            print(f"  Kociemba faster: {koc_faster} cubes ({koc_faster/total*100:.1f}%)")
            print(f"  Equal speed: {equal} cubes ({equal/total*100:.1f}%)")
        
        # Which solver uses fewer nodes?
        if 'th_nodes_expanded' in df.columns and 'koc_nodes_expanded' in df.columns:
            print("\nWhich solver expands fewer nodes?")
            th_fewer_nodes = (df['th_nodes_expanded'] < df['koc_nodes_expanded']).sum()
            koc_fewer_nodes = (df['koc_nodes_expanded'] < df['th_nodes_expanded']).sum()
            equal_nodes = (df['th_nodes_expanded'] == df['koc_nodes_expanded']).sum()
            
            total = len(df)
            print(f"  Thistlethwaite fewer nodes: {th_fewer_nodes} cubes ({th_fewer_nodes/total*100:.1f}%)")
            print(f"  Kociemba fewer nodes: {koc_fewer_nodes} cubes ({koc_fewer_nodes/total*100:.1f}%)")
            print(f"  Equal nodes: {equal_nodes} cubes ({equal_nodes/total*100:.1f}%)")


def main():
    """Main function to run the data collection."""
    print("=" * 80)
    print("RUBIK'S CUBE SOLVER TRAINING DATA COLLECTOR")
    print("=" * 80)
    
    # Configuration
    NUM_CUBES = 5000
    os.makedirs("ML_integration_data", exist_ok=True)
    OUTPUT_FILE = "ML_integration_data/training_data.csv"
    
    # Create collector
    collector = TrainingDataCollector(output_file=OUTPUT_FILE)
    
    # Update baselines if previous data exists
    if os.path.exists("ML_integration_data/training_data.csv"):
        collector.categorizer.update_baselines_from_data("ML_integration_data/training_data.csv")
    
    # Run experiment
    try:
        data = collector.run_experiment(num_cubes=NUM_CUBES)
        
        # Analyze collected data
        collector.analyze_collected_data(OUTPUT_FILE)
        
        print(f"\nData collection complete!")
        print(f"  Data saved to: {OUTPUT_FILE}")
        print(f"  Summary saved to: {OUTPUT_FILE.replace('.csv', '_summary.txt')}")
        print(f"  Sample saved to: {OUTPUT_FILE.replace('.csv', '_sample.csv')}")
        
    except KeyboardInterrupt:
        print("\nData collection interrupted by user!")
        if collector.data:
            print("  Saving collected data so far...")
            collector.save_to_csv("ML_integration_data/training_data_interrupted.csv")
    except Exception as e:
        print(f"\nError during data collection: {e}")
        import traceback
        traceback.print_exc()
        
        if collector.data:
            print("Saving collected data so far...")
            collector.save_to_csv("ML_integration_data/training_data_error.csv")


if __name__ == "__main__":
    main()