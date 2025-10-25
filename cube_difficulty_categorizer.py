import random
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cube_solver.cube.cube import Cube
from cube_solver.cube.enums import Face, Color, Cubie, Orbit
from cube_solver.solver.thistlethwaite import Thistlethwaite
from cube_solver.solver.kociemba import Kociemba


@dataclass
class CubeDifficulty:
    """Represents a cube configuration with its difficulty metrics."""
    scramble: str
    cube: Cube
    
    # Distance-based metrics
    manhattan_distance: int
    hamming_distance: Dict[str, int]
    orientation_distance: Dict[str, int]
    
    # Solver-based metrics
    phase_distances: Dict[str, Any]
    estimated_solution_length: int
    solution_variance: float
    
    # Human-perceived metrics
    face_uniformity: Dict[str, float]
    color_clustering: Dict[str, Any]
    matched_pairs: Dict[str, int]
    
    # Difficulty categories for each metric
    manhattan_category: str
    hamming_category: str
    orientation_category: str
    solution_category: str
    overall_category: str


class CubeDifficultyCategorizer:
    # Categorizes cube configurations by difficulty metrics.
    
    def __init__(self):
        self.solver_thistlethwaite = Thistlethwaite()
        self.solver_kociemba = Kociemba()
    
    def generate_scramble(self, length: int = None) -> str:
        # Generate a random scramble.
        moves = ["U", "D", "L", "R", "F", "B"]
        suffixes = ["", "'", "2"]
        
        if length is None:
            length = random.randint(5, 25)
        
        scramble = []
        last_face = None
        
        for _ in range(length):
            # Avoid repeating the same face
            available_moves = [move for move in moves if move != last_face]
            if not available_moves:
                available_moves = moves
            
            move = random.choice(available_moves)
            suffix = random.choice(suffixes)
            scramble.append(move + suffix)
            last_face = move
        
        return " ".join(scramble)
    
    def calculate_manhattan_distance(self, cube: Cube) -> int:
        # Calculate Manhattan distance of cubies from their solved positions.
        total_distance = 0
        
        # For corners (indices 0-7)
        for i in range(8):
            current_pos = cube.permutation[i]
            solved_pos = i
            if current_pos != solved_pos:
                total_distance += abs(current_pos - solved_pos)
        
        # For edges (indices 8-19)
        for i in range(8, 20):
            current_pos = cube.permutation[i]
            solved_pos = i
            if current_pos != solved_pos:
                total_distance += abs(current_pos - solved_pos)
        
        return total_distance
    
    def calculate_hamming_distance(self, cube: Cube) -> Dict[str, int]:
        # Calculate Hamming distance (number of pieces in wrong positions).
        corner_hamming = sum(1 for i in range(8) if cube.permutation[i] != i)
        edge_hamming = sum(1 for i in range(8, 20) if cube.permutation[i] != i)
        total_hamming = corner_hamming + edge_hamming
        
        return {
            'corner_hamming': corner_hamming,
            'edge_hamming': edge_hamming,
            'total_hamming': total_hamming
        }
    
    def calculate_orientation_distance(self, cube: Cube) -> Dict[str, int]:
        # Calculate orientation distance (number of misoriented pieces).
        corner_orientation = sum(1 for i in range(8) if cube.orientation[i] != 0)
        edge_orientation = sum(1 for i in range(8, 20) if cube.orientation[i] != 0)
        total_orientation = corner_orientation + edge_orientation
        
        return {
            'corner_orientation': corner_orientation,
            'edge_orientation': edge_orientation,
            'total_orientation': total_orientation
        }
    
    def calculate_phase_distances(self, cube: Cube) -> Dict[str, Any]:
        # Calculate phase distances for Thistlethwaite and Kociemba solvers.
        try:
            # Get phase coordinates for Thistlethwaite
            coords = cube.get_coords(partial_corner_perm=True, partial_edge_perm=True)
            thistlethwaite_phases = []
            
            for phase in range(4):
                phase_coords = self.solver_thistlethwaite.phase_coords(coords, phase)
                thistlethwaite_phases.append(phase_coords)
            
            # Get phase coordinates for Kociemba
            kociemba_phases = []
            for phase in range(2):
                phase_coords = self.solver_kociemba.phase_coords(coords, phase)
                kociemba_phases.append(phase_coords)
            
            return {
                'thistlethwaite_phases': thistlethwaite_phases,
                'kociemba_phases': kociemba_phases
            }
        except Exception as e:
            return {
                'thistlethwaite_phases': None,
                'kociemba_phases': None,
                'error': str(e)
            }
    
    def estimate_solution_length(self, cube: Cube) -> Tuple[int, float]:
        # Estimate solution length and calculate variance.
        solutions = []
        
        try:
            # Try Thistlethwaite solver
            solution = self.solver_thistlethwaite.solve(cube, optimal=False)
            if solution:
                moves = solution.split() if isinstance(solution, str) else solution
                solutions.append(len(moves))
        except:
            pass
        
        try:
            # Try Kociemba solver
            solution = self.solver_kociemba.solve(cube, optimal=False)
            if solution:
                moves = solution.split() if isinstance(solution, str) else solution
                solutions.append(len(moves))
        except:
            pass
        
        if solutions:
            avg_length = np.mean(solutions)
            variance = np.var(solutions) if len(solutions) > 1 else 0.0
            return int(avg_length), variance
        else:
            return 0, 0.0
    
    def calculate_face_color_uniformity(self, cube: Cube) -> Dict[str, float]:
        # Calculate face color uniformity (how many stickers are correct color).
        face_uniformity = {}
        cube_repr = str(cube)
        
        # Expected colors for each face (solved state)
        expected_colors = {
            'U': 'W',  # White
            'F': 'G',  # Green
            'R': 'R',  # Red
            'D': 'Y',  # Yellow
            'B': 'B',  # Blue
            'L': 'O'   # Orange
        }
        
        # Face positions in the string representation
        face_positions = {
            'U': (0, 9),       # Up face: positions 0-8
            'F': (9, 18),      # Front face: positions 9-17
            'R': (18, 27),     # Right face: positions 18-26
            'D': (27, 36),     # Down face: positions 27-35
            'B': (36, 45),     # Back face: positions 36-44
            'L': (45, 54)      # Left face: positions 45-53
        }
        
        for face, (start, end) in face_positions.items():
            face_stickers = cube_repr[start:end]
            expected_color = expected_colors[face]
            correct_stickers = sum(1 for sticker in face_stickers if sticker == expected_color)
            uniformity = correct_stickers / 9.0  # 9 stickers per face
            face_uniformity[face] = uniformity
        
        return face_uniformity
    
    def calculate_color_clustering(self, cube: Cube) -> Dict[str, Any]:
        # Calculate color clustering (size of contiguous same-colored blocks).
        cube_repr = str(cube)
        clustering_metrics = {}
        
        # Face positions in the string representation
        face_positions = {
            'U': (0, 9),       # Up face: positions 0-8
            'F': (9, 18),      # Front face: positions 9-17
            'R': (18, 27),     # Right face: positions 18-26
            'D': (27, 36),     # Down face: positions 27-35
            'B': (36, 45),     # Back face: positions 36-44
            'L': (45, 54)      # Left face: positions 45-53
        }
        
        for face, (start, end) in face_positions.items():
            face_stickers = cube_repr[start:end]
            
            # Find contiguous blocks of same color
            clusters = []
            current_cluster = [face_stickers[0]]
            
            for i in range(1, len(face_stickers)):
                if face_stickers[i] == face_stickers[i-1]:
                    current_cluster.append(face_stickers[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [face_stickers[i]]
            clusters.append(current_cluster)
            
            # Calculate clustering metrics
            cluster_sizes = [len(cluster) for cluster in clusters]
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
            num_clusters = len(clusters)
            
            clustering_metrics[face] = {
                'max_cluster_size': max_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'num_clusters': num_clusters,
                'cluster_sizes': cluster_sizes
            }
        
        return clustering_metrics
    
    def calculate_matched_pairs(self, cube: Cube) -> Dict[str, int]:
        # Calculate number of matched edge/corner color pairs.
        
        matched_corners = 0
        matched_edges = 0
        
        # Check for basic cross (4 edges around center)
        cross_edges = ['UB', 'UF', 'UL', 'UR']  # Edges that should be in cross
        
        return {
            'matched_corners': matched_corners,
            'matched_edges': matched_edges,
            'total_matched': matched_corners + matched_edges
        }
    
    def categorize_by_manhattan_distance(self, manhattan_distance: int) -> str:
        # Categorize cube by Manhattan distance.
        if manhattan_distance <= 20:
            return "Very Easy"
        elif manhattan_distance <= 40:
            return "Easy"
        elif manhattan_distance <= 60:
            return "Medium"
        elif manhattan_distance <= 80:
            return "Hard"
        else:
            return "Expert"
    
    def categorize_by_hamming_distance(self, hamming_distance: int) -> str:
        # Categorize cube by Hamming distance.
        if hamming_distance <= 5:
            return "Very Easy"
        elif hamming_distance <= 10:
            return "Easy"
        elif hamming_distance <= 15:
            return "Medium"
        elif hamming_distance <= 18:
            return "Hard"
        else:
            return "Expert"
    
    def categorize_by_orientation_distance(self, orientation_distance: int) -> str:
        # Categorize cube by orientation distance.
        if orientation_distance <= 5:
            return "Very Easy"
        elif orientation_distance <= 10:
            return "Easy"
        elif orientation_distance <= 15:
            return "Medium"
        elif orientation_distance <= 20:
            return "Hard"
        else:
            return "Expert"
    
    def categorize_by_solution_length(self, solution_length: int) -> str:
        # Categorize cube by estimated solution length.
        if solution_length <= 5:
            return "Very Easy"
        elif solution_length <= 15:
            return "Easy"
        elif solution_length <= 25:
            return "Medium"
        elif solution_length <= 35:
            return "Hard"
        else:
            return "Expert"
    
    def analyze_cube(self, scramble: str) -> CubeDifficulty:
        # Analyze a cube and calculate all difficulty metrics.
        cube = Cube(scramble)
        
        # Calculate all metrics
        manhattan_distance = self.calculate_manhattan_distance(cube)
        hamming_distance = self.calculate_hamming_distance(cube)
        orientation_distance = self.calculate_orientation_distance(cube)
        phase_distances = self.calculate_phase_distances(cube)
        solution_length, solution_variance = self.estimate_solution_length(cube)
        face_uniformity = self.calculate_face_color_uniformity(cube)
        color_clustering = self.calculate_color_clustering(cube)
        matched_pairs = self.calculate_matched_pairs(cube)
        
        # Categorize by each metric
        manhattan_category = self.categorize_by_manhattan_distance(manhattan_distance)
        hamming_category = self.categorize_by_hamming_distance(hamming_distance['total_hamming'])
        orientation_category = self.categorize_by_orientation_distance(orientation_distance['total_orientation'])
        solution_category = self.categorize_by_solution_length(solution_length)
        
        # Determine overall difficulty category (use the most restrictive)
        categories = [manhattan_category, hamming_category, orientation_category, solution_category]
        overall_category = max(categories, key=lambda x: ['Very Easy', 'Easy', 'Medium', 'Hard', 'Expert'].index(x))
        
        return CubeDifficulty(
            scramble=scramble,
            cube=cube,
            manhattan_distance=manhattan_distance,
            hamming_distance=hamming_distance,
            orientation_distance=orientation_distance,
            phase_distances=phase_distances,
            estimated_solution_length=solution_length,
            solution_variance=solution_variance,
            face_uniformity=face_uniformity,
            color_clustering=color_clustering,
            matched_pairs=matched_pairs,
            manhattan_category=manhattan_category,
            hamming_category=hamming_category,
            orientation_category=orientation_category,
            solution_category=solution_category,
            overall_category=overall_category
        )
    
    def generate_test_cubes(self, num_cubes: int = 10) -> List[CubeDifficulty]:
        # Generate a set of test cubes with different difficulty levels.
        cubes = []
        
        # Generate cubes with different scramble lengths
        scramble_lengths = [3, 5, 8, 12, 15, 20, 25, 30, 35, 40]
        
        for i in range(num_cubes):
            length = scramble_lengths[i % len(scramble_lengths)]
            scramble = self.generate_scramble(length)
            cube_difficulty = self.analyze_cube(scramble)
            cubes.append(cube_difficulty)
        
        return cubes
    
    def group_by_metric(self, cubes: List[CubeDifficulty], metric_name: str) -> Dict[str, List[CubeDifficulty]]:
        # Group cubes by a specific metric.
        groups = defaultdict(list)
        
        for cube in cubes:
            if metric_name == 'manhattan_distance':
                category = cube.manhattan_category
            elif metric_name == 'hamming_distance':
                category = cube.hamming_category
            elif metric_name == 'orientation_distance':
                category = cube.orientation_category
            elif metric_name == 'solution_length':
                category = cube.solution_category
            else:
                category = "Unknown"
            
            groups[category].append(cube)
        
        return dict(groups)
    
    def print_analysis_report(self, cubes: List[CubeDifficulty]):
        # Print a comprehensive analysis report.
        print("=" * 80)
        print("CUBE DIFFICULTY CATEGORIZATION REPORT")
        print("=" * 80)
        
        print(f"\nTotal cubes analyzed: {len(cubes)}")
        
        # Overall difficulty distribution
        difficulty_counts = defaultdict(int)
        for cube in cubes:
            difficulty_counts[cube.overall_category] += 1
        
        print("\nOverall Difficulty Distribution:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count}")
        
        # Metric analysis
        print("\n" + "=" * 50)
        print("METRIC ANALYSIS")
        print("=" * 50)
        
        # Manhattan distance analysis
        manhattan_values = [cube.manhattan_distance for cube in cubes]
        print(f"\nManhattan Distance:")
        print(f"  Min: {min(manhattan_values)}")
        print(f"  Max: {max(manhattan_values)}")
        print(f"  Avg: {np.mean(manhattan_values):.2f}")
        print(f"  Std: {np.std(manhattan_values):.2f}")
        
        # Hamming distance analysis
        hamming_values = [cube.hamming_distance['total_hamming'] for cube in cubes]
        print(f"\nHamming Distance:")
        print(f"  Min: {min(hamming_values)}")
        print(f"  Max: {max(hamming_values)}")
        print(f"  Avg: {np.mean(hamming_values):.2f}")
        print(f"  Std: {np.std(hamming_values):.2f}")
        
        # Orientation distance analysis
        orientation_values = [cube.orientation_distance['total_orientation'] for cube in cubes]
        print(f"\nOrientation Distance:")
        print(f"  Min: {min(orientation_values)}")
        print(f"  Max: {max(orientation_values)}")
        print(f"  Avg: {np.mean(orientation_values):.2f}")
        print(f"  Std: {np.std(orientation_values):.2f}")
        
        # Solution length analysis
        solution_values = [cube.estimated_solution_length for cube in cubes]
        print(f"\nEstimated Solution Length:")
        print(f"  Min: {min(solution_values)}")
        print(f"  Max: {max(solution_values)}")
        print(f"  Avg: {np.mean(solution_values):.2f}")
        print(f"  Std: {np.std(solution_values):.2f}")
        
        # Face uniformity analysis
        print(f"\nFace Uniformity Analysis:")
        for face in ['U', 'F', 'R', 'D', 'B', 'L']:
            uniformity_values = [cube.face_uniformity[face] for cube in cubes]
            print(f"  {face} face: {np.mean(uniformity_values):.3f} ± {np.std(uniformity_values):.3f}")
        
        # Group by different metrics
        print("\n" + "=" * 50)
        print("GROUPING BY METRICS")
        print("=" * 50)
        
        # Group by Manhattan distance
        manhattan_groups = self.group_by_metric(cubes, 'manhattan_distance')
        print(f"\nGrouped by Manhattan Distance:")
        for category, group_cubes in manhattan_groups.items():
            print(f"  {category}: {len(group_cubes)} cubes")
        
        # Group by Hamming distance
        hamming_groups = self.group_by_metric(cubes, 'hamming_distance')
        print(f"\nGrouped by Hamming Distance:")
        for category, group_cubes in hamming_groups.items():
            print(f"  {category}: {len(group_cubes)} cubes")
        
        # Group by solution length
        solution_groups = self.group_by_metric(cubes, 'solution_length')
        print(f"\nGrouped by Solution Length:")
        for category, group_cubes in solution_groups.items():
            print(f"  {category}: {len(group_cubes)} cubes")
        
        # Group by orientation distance
        orientation_groups = self.group_by_metric(cubes, 'orientation_distance')
        print(f"\nGrouped by Orientation Distance:")
        for category, group_cubes in orientation_groups.items():
            print(f"  {category}: {len(group_cubes)} cubes")
        
        print("\n" + "=" * 50)
        print("DETAILED CUBE CONFIGURATIONS")
        print("=" * 50)
        
        for i, cube in enumerate(cubes):
            print(f"\nCube {i+1}:")
            print(f"  Scramble: {cube.scramble}")
            print(f"  Overall Category: {cube.overall_category}")
            print(f"  Manhattan Distance: {cube.manhattan_distance} ({cube.manhattan_category})")
            print(f"  Hamming Distance: {cube.hamming_distance['total_hamming']} ({cube.hamming_category})")
            print(f"  Orientation Distance: {cube.orientation_distance['total_orientation']} ({cube.orientation_category})")
            print(f"  Estimated Solution Length: {cube.estimated_solution_length} ({cube.solution_category})")
            print(f"  Solution Variance: {cube.solution_variance:.2f}")
            
            # Face uniformity
            avg_uniformity = np.mean(list(cube.face_uniformity.values()))
            print(f"  Average Face Uniformity: {avg_uniformity:.3f}")
            
            # Color clustering
            avg_clustering = np.mean([cube.color_clustering[face]['avg_cluster_size'] for face in cube.color_clustering])
            print(f"  Average Color Clustering: {avg_clustering:.2f}")


def main():
    # Main function to run the cube difficulty categorization.
    categorizer = CubeDifficultyCategorizer()
    
    print("Generating cube configurations...")
    cubes = categorizer.generate_test_cubes(num_cubes=10)
    
    print("Analyzing configurations...")
    categorizer.print_analysis_report(cubes)
    
    # Return cubes for further use
    return cubes


if __name__ == "__main__":
    cubes = main()
    
    # Example of how to access specific metrics for algorithm testing
    print("\n" + "=" * 50)
    print("EXAMPLE: Accessing specific metrics for algorithm testing")
    print("=" * 50)
    
    # Group cubes by different metrics for algorithm testing
    categorizer = CubeDifficultyCategorizer()
    
    # Group by Manhattan distance
    manhattan_groups = categorizer.group_by_metric(cubes, 'manhattan_distance')
    print(f"\nCubes grouped by Manhattan Distance:")
    for category, group_cubes in manhattan_groups.items():
        print(f"  {category}: {len(group_cubes)} cubes")
        for cube in group_cubes:
            print(f"    - {cube.scramble} (distance: {cube.manhattan_distance})")
    
    # Group by Hamming distance
    hamming_groups = categorizer.group_by_metric(cubes, 'hamming_distance')
    print(f"\nCubes grouped by Hamming Distance:")
    for category, group_cubes in hamming_groups.items():
        print(f"  {category}: {len(group_cubes)} cubes")
        for cube in group_cubes:
            print(f"    - {cube.scramble} (distance: {cube.hamming_distance['total_hamming']})")
    
    # Group by solution length
    solution_groups = categorizer.group_by_metric(cubes, 'solution_length')
    print(f"\nCubes grouped by Solution Length:")
    for category, group_cubes in solution_groups.items():
        print(f"  {category}: {len(group_cubes)} cubes")
        for cube in group_cubes:
            print(f"    - {cube.scramble} (length: {cube.estimated_solution_length})")
    
    # Group by orientation distance
    orientation_groups = categorizer.group_by_metric(cubes, 'orientation_distance')
    print(f"\nCubes grouped by Orientation Distance:")
    for category, group_cubes in orientation_groups.items():
        print(f"  {category}: {len(group_cubes)} cubes")
        for cube in group_cubes:
            print(f"    - {cube.scramble} (distance: {cube.orientation_distance['total_orientation']})")
    
    # Return the cubes for further use
    print(f"\nReturning {len(cubes)} cube configurations for algorithm testing.")
