import random
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from src.cube_solver.cube.cube import Cube
from src.cube_solver.solver.thistlethwaite import Thistlethwaite
from src.cube_solver.solver.kociemba import Kociemba

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class CubeDifficulty:
    """Represents a cube configuration with its difficulty metrics."""
    scramble: str
    cube: Cube
    
    # Distance-based metrics
    manhattan_distance: int
    hamming_distance: Dict[str, int]
    orientation_distance: Dict[str, int]
    
    # Cycle decomposition metrics
    cycle_metrics: Dict[str, Any]
    
    # Thistlethwaite subgroup distances
    thistlethwaite_distances: Dict[str, int]
    
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
    
    # Numerical difficulty scores , 0=easy 1=hard
    manhattan_score: float  
    hamming_score: float   
    orientation_score: float 
    solution_score: float 
    overall_score: float  
    
    # Cube coordinates for ML features
    coords: Tuple[int, ...]  # (co, eo, cp, ep) coordinates
    partial_coords: Tuple[Any, ...]  # (co, eo, pcp, pep) coordinates


class CubeDifficultyCategorizer:
    """Categorizes cube configurations by difficulty metrics."""
    
    # Theoretical maximums for normalization
    MAX_MANHATTAN = 120  # Theoretical maximum Manhattan distance
    MAX_HAMMING = 20     # 20 pieces total
    MAX_ORIENTATION = 20  # 20 pieces total
    MAX_SOLUTION_LENGTH = 60  # Conservative upper bound
    MAX_CYCLE_SWAP = 16  # Maximum swap cost (8+8 corners/edges)
    
    # Reference ranges for each difficulty level (0-1)
    DIFFICULTY_RANGES = {
        'very_easy': (0.0, 0.2),
        'easy': (0.2, 0.4),
        'medium': (0.4, 0.6),
        'hard': (0.6, 0.8),
        'expert': (0.8, 1.0)
    }
    
    def __init__(self):
        self.solver_thistlethwaite = Thistlethwaite()
        self.solver_kociemba = Kociemba()
        
        # Statistical baselines (to be updated from benchmark data)
        self.baseline_manhattan = None
        self.baseline_hamming = None
        self.baseline_orientation = None
        self.baseline_solution = None

    def calculate_search_complexity_metrics(self, cube):
        """Calculate metrics specifically for predicting IDA* search complexity."""
        metrics = {}
        
        # Get existing metrics
        cycle_metrics = self.calculate_cycle_decomposition(cube)
        thistle_metrics = self.calculate_thistlethwaite_subgroup_distances(cube)
        
        # 1. Parity-based metrics
        corner_perm = cube.permutation[:8]
        edge_perm = cube.permutation[8:20]
        
        # Calculate corner permutation parity
        corner_parity = self._calculate_parity(corner_perm)
        edge_parity = self._calculate_parity([p-8 for p in edge_perm])  
        
        metrics['corner_parity'] = corner_parity
        metrics['edge_parity'] = edge_parity
        metrics['total_parity'] = (corner_parity + edge_parity) % 2
        
        # 2. Cycle structure complexity
        # Long cycles are much harder for IDA*
        max_cycle_length = max(cycle_metrics['max_corner_cycle'], cycle_metrics['max_edge_cycle'])
        metrics['max_cycle_length'] = max_cycle_length
        
        # 3. "Depth" metrics - estimate how deep IDA* needs to go
        # For Kociemba phase 1: distance to H subgroup
        metrics['phase1_depth_estimate'] = self._estimate_phase1_depth(cube)
        
        # 4. Branching factor estimate
        metrics['branching_factor_estimate'] = self._estimate_branching_factor(cube)
        
        # 5. Symmetry metrics (symmetric states are often easier)
        metrics['symmetry_score'] = self._calculate_symmetry_score(cube)
        
        # 6. Combined difficulty score for IDA*
        # Weight factors that matter most for IDA*
        ida_difficulty = (
            0.3 * (cycle_metrics['total_swap_cost'] / 16) +  # 0-1 normalized
            0.4 * (max_cycle_length / 8) +  # Normalized by max possible
            0.2 * (thistle_metrics['dist_to_G1'] / 12) +  # Normalized
            0.1 * metrics['phase1_depth_estimate']
        )
        metrics['ida_difficulty_score'] = ida_difficulty
        
        return metrics

    def _calculate_parity(self, permutation):
        """Calculate parity of a permutation (0=even, 1=odd)."""
        visited = [False] * len(permutation)
        transpositions = 0
        
        for i in range(len(permutation)):
            if not visited[i]:
                cycle_length = 0
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = permutation[j]
                    cycle_length += 1
                if cycle_length > 1:
                    transpositions += (cycle_length - 1)
        
        return transpositions % 2

    def _estimate_phase1_depth(self, cube):
        """Estimate depth needed for Kociemba phase 1."""
        # Count edge orientations + edges not in E-slice
        # This is a heuristic for phase 1 distance
        bad_edges = 0
        for i in range(8, 20):
            if cube.orientation[i] != 0:
                bad_edges += 1
        
        # E-slice edges in solved state (adjust based on your mapping)
        e_slice_positions = [8, 9, 10, 11]  # UF, UR, UB, UL
        for i in range(8, 20):
            if (i in e_slice_positions and cube.permutation[i] not in e_slice_positions) or \
            (i not in e_slice_positions and cube.permutation[i] in e_slice_positions):
                bad_edges += 1
        
        return bad_edges / 2  # Rough estimate

    def _estimate_branching_factor(self, cube: Cube) -> float:
        """Estimate branching factor for IDA* search."""
        # States with many oriented pieces have lower branching
        oriented_corners = sum(1 for i in range(8) if cube.orientation[i] == 0)
        oriented_edges = sum(1 for i in range(8, 20) if cube.orientation[i] == 0)
        
        # More oriented pieces = fewer moves affect orientation = lower branching
        orientation_factor = (oriented_corners + oriented_edges) / 20
        
        # Estimated branching factor (1-18 moves, average ~13)
        return 18 - (orientation_factor * 10)
    
    def generate_scramble(self, length: int = None) -> str:
        """Generate a random scramble."""
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
    
    def calculate_manhattan_distance(self, cube):
        """Calculate Manhattan distance of cubies from their solved positions."""
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
    
    def calculate_hamming_distance(self, cube):
        """Calculate Hamming distance (number of pieces in wrong positions)."""
        corner_hamming = sum(1 for i in range(8) if cube.permutation[i] != i)
        edge_hamming = sum(1 for i in range(8, 20) if cube.permutation[i] != i)
        total_hamming = corner_hamming + edge_hamming
        
        return {
            'corner_hamming': corner_hamming,
            'edge_hamming': edge_hamming,
            'total_hamming': total_hamming
        }
    
    def calculate_orientation_distance(self, cube):
        """Calculate orientation distance (number of misoriented pieces)."""
        corner_orientation = sum(1 for i in range(8) if cube.orientation[i] != 0)
        edge_orientation = sum(1 for i in range(8, 20) if cube.orientation[i] != 0)
        total_orientation = corner_orientation + edge_orientation
        
        return {
            'corner_orientation': corner_orientation,
            'edge_orientation': edge_orientation,
            'total_orientation': total_orientation
        }
    
    def calculate_cycle_decomposition(self, cube) :
        """Calculate cycle decomposition metrics for permutation groups."""
        # Corner cycles (indices 0-7 in permutation)
        corner_perm = cube.permutation[:8]
        corner_visited = [False] * 8
        corner_cycles = []
        
        for i in range(8):
            if not corner_visited[i]:
                current = i
                cycle = []
                while not corner_visited[current]:
                    corner_visited[current] = True
                    cycle.append(current)
                    current = corner_perm[current]
                corner_cycles.append(cycle)
        
        # Edge cycles (indices 8-19 in permutation)
        # We'll work with 0-11 for edges internally
        edge_perm_full = cube.permutation[8:20]
        edge_perm = [p - 8 for p in edge_perm_full]  # Convert to 0-11 range
        
        edge_visited = [False] * 12
        edge_cycles = []
        
        for i in range(12):
            if not edge_visited[i]:
                current = i
                cycle = []
                while not edge_visited[current]:
                    edge_visited[current] = True
                    cycle.append(current)
                    current = edge_perm[current]
                edge_cycles.append(cycle)
        
        # Calculate metrics
        corner_cycle_lengths = [len(cycle) for cycle in corner_cycles]
        edge_cycle_lengths = [len(cycle) for cycle in edge_cycles]
        
        # Count non-trivial cycles (length > 1)
        corner_nontrivial_cycles = sum(1 for length in corner_cycle_lengths if length > 1)
        edge_nontrivial_cycles = sum(1 for length in edge_cycle_lengths if length > 1)
        
        # Calculate swap cost: (N - C) where C is number of cycles
        corner_swap_cost = 8 - len(corner_cycles)
        edge_swap_cost = 12 - len(edge_cycles)
        total_swap_cost = corner_swap_cost + edge_swap_cost
        
        # Handle edge cases for empty lists
        max_corner_cycle = max(corner_cycle_lengths) if corner_cycle_lengths else 1
        max_edge_cycle = max(edge_cycle_lengths) if edge_cycle_lengths else 1
        avg_corner_cycle = np.mean(corner_cycle_lengths) if corner_cycle_lengths else 1
        avg_edge_cycle = np.mean(edge_cycle_lengths) if edge_cycle_lengths else 1
        
        return {
            'corner_cycles': corner_cycles,
            'corner_cycle_count': len(corner_cycles),
            'corner_cycle_lengths': corner_cycle_lengths,
            'edge_cycles': edge_cycles,
            'edge_cycle_count': len(edge_cycles),
            'edge_cycle_lengths': edge_cycle_lengths,
            'corner_swap_cost': max(0, corner_swap_cost),
            'edge_swap_cost': max(0, edge_swap_cost),
            'total_swap_cost': max(0, total_swap_cost),
            'max_corner_cycle': max_corner_cycle,
            'max_edge_cycle': max_edge_cycle,
            'avg_corner_cycle': avg_corner_cycle,
            'avg_edge_cycle': avg_edge_cycle,
            'corner_nontrivial_cycles': corner_nontrivial_cycles,
            'edge_nontrivial_cycles': edge_nontrivial_cycles
        }
    
    def calculate_thistlethwaite_subgroup_distances(self, cube):
        """Calculate distances to Thistlethwaite subgroups."""
        # Get cube representation
        try:
            # G0 -> G1: Edge orientations only
            # In G1, all edge orientations must be 0
            # Count bad edge orientations (edges not in their G1 allowed orientation)
            bad_edge_orientation = 0
            for i in range(8, 20):
                if cube.orientation[i] != 0:
                    bad_edge_orientation += 1
            
            # G1 -> G2: Corner orientations and E-slice edges
            # In G2, corner orientations must be 0, and E-slice edges must be in E-slice
            
            # E-slice edges are positions 9, 11, 12, 14 (UB, UF, UL, UR in some mappings)
            e_slice_positions = [1, 3, 4, 6]  # Adjusted indices (0-based from edges)
            bad_corner_orientation = sum(1 for i in range(8) if cube.orientation[i] != 0)
            
            # Count edges not in E-slice that should be, and edges in E-slice that shouldn't be
            bad_edge_slice = 0
            # E-slice edges in solved state: positions 8, 9, 10, 11 (UF, UR, UB, UL)
            e_slice_solved = [8, 9, 10, 11]
            for i in range(8, 20):
                # If an edge that should be in E-slice is not in E-slice positions
                if (i in e_slice_solved and cube.permutation[i] not in e_slice_solved) or \
                   (i not in e_slice_solved and cube.permutation[i] in e_slice_solved):
                    bad_edge_slice += 1
            
            dist_to_G2 = bad_corner_orientation + bad_edge_slice
            
            # G2 -> G3: Corner tetrads
            # Corners must be in their correct tetrad (even/odd permutation of corners)
            # Count corners not in their natural orbit
            bad_corner_tetrad = 0
            # Check corner parity - if permutation has odd parity, all corners are in wrong tetrad
            corner_perm = cube.permutation[:8]
            visited = [False] * 8
            parity = 0
            
            for i in range(8):
                if not visited[i]:
                    cycle_length = 0
                    j = i
                    while not visited[j]:
                        visited[j] = True
                        j = corner_perm[j]
                        cycle_length += 1
                    if cycle_length > 1:
                        parity += (cycle_length - 1)
            
            # If parity is odd, cube is not in G2->G3 subgroup
            if parity % 2 == 1:
                bad_corner_tetrad = 8  # All corners wrong
            
            dist_to_G3 = bad_corner_tetrad
            
            return {
                'dist_to_G1': bad_edge_orientation,
                'dist_to_G2': dist_to_G2,
                'dist_to_G3': dist_to_G3,
                'bad_edge_orientation': bad_edge_orientation,
                'bad_corner_orientation': bad_corner_orientation,
                'bad_edge_slice': bad_edge_slice,
                'bad_corner_tetrad': bad_corner_tetrad,
                'corner_parity': parity % 2
            }
            
        except Exception as e:
            # Fallback if calculation fails
            return {
                'dist_to_G1': 0,
                'dist_to_G2': 0,
                'dist_to_G3': 0,
                'bad_edge_orientation': 0,
                'bad_corner_orientation': 0,
                'bad_edge_slice': 0,
                'bad_corner_tetrad': 0,
                'corner_parity': 0,
                'error': str(e)
            }
    
    def calculate_phase_distances(self, cube):
        """Calculate phase distances for Thistlethwaite and Kociemba solvers."""
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
        """Estimate solution length and calculate variance."""
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
    
    def calculate_face_color_uniformity(self, cube):
        """Calculate face color uniformity (how many stickers are correct color)."""
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
    
    def calculate_color_clustering(self, cube):
        """Calculate color clustering (size of contiguous same-colored blocks)."""
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
    
    def calculate_matched_pairs(self, cube):
        """Calculate number of matched edge/corner color pairs."""
        matched_corners = 0
        matched_edges = 0
        
        # Check for basic cross (4 edges around center)
        cross_edges = ['UB', 'UF', 'UL', 'UR']  # Edges that should be in cross
        
        return {
            'matched_corners': matched_corners,
            'matched_edges': matched_edges,
            'total_matched': matched_corners + matched_edges
        }
    
    # Calculate numerical difficulty scores (0-1)
    def calculate_manhattan_score(self, manhattan_distance: int) -> float:
        """Calculate normalized Manhattan distance score (0-1)."""
        # Clip to avoid values > 1.0
        return min(1.0, manhattan_distance / self.MAX_MANHATTAN)
    
    def calculate_hamming_score(self, hamming_distance):
        """Calculate normalized Hamming distance score (0-1)."""
        return min(1.0, hamming_distance / self.MAX_HAMMING)
    
    def calculate_orientation_score(self, orientation_distance):
        """Calculate normalized orientation distance score (0-1)."""
        return min(1.0, orientation_distance / self.MAX_ORIENTATION)
    
    def calculate_solution_score(self, solution_length):
        """Calculate normalized solution length score (0-1)."""
        return min(1.0, solution_length / self.MAX_SOLUTION_LENGTH)
    
    def calculate_cycle_score(self, swap_cost):
        """Calculate normalized cycle swap cost score (0-1)."""
        return min(1.0, swap_cost / self.MAX_CYCLE_SWAP)
    
    def calculate_overall_score(self, scores):
        """Calculate overall difficulty score using weighted average."""
        # Weights can be adjusted based on importance
        weights = {
            'manhattan': 0.20,
            'hamming': 0.20,
            'orientation': 0.20,
            'solution': 0.20,
            'cycle': 0.20  
        }
        
        overall = (scores['manhattan'] * weights['manhattan'] +
                   scores['hamming'] * weights['hamming'] +
                   scores['orientation'] * weights['orientation'] +
                   scores['solution'] * weights['solution'] +
                   scores['cycle'] * weights['cycle'])
        
        return min(1.0, overall)
    
    # Convert numerical score to category
    def score_to_category(self, score: float) -> str:
        """Convert numerical score (0-1) to difficulty category."""
        for category, (low, high) in self.DIFFICULTY_RANGES.items():
            if low <= score < high:
                return category.replace('_', ' ').title()
        return "Expert"  # Default for scores >= 1.0
    
    def categorize_by_manhattan_distance(self, manhattan_distance):
        """Categorize cube by Manhattan distance."""
        score = self.calculate_manhattan_score(manhattan_distance)
        return self.score_to_category(score)
    
    def categorize_by_hamming_distance(self, hamming_distance):
        """Categorize cube by Hamming distance."""
        score = self.calculate_hamming_score(hamming_distance)
        return self.score_to_category(score)
    
    def categorize_by_orientation_distance(self, orientation_distance):
        """Categorize cube by orientation distance."""
        score = self.calculate_orientation_score(orientation_distance)
        return self.score_to_category(score)
    
    def categorize_by_solution_length(self, solution_length):
        """Categorize cube by estimated solution length."""
        score = self.calculate_solution_score(solution_length)
        return self.score_to_category(score)
    
    def analyze_cube(self, scramble):
        """Analyze a cube and calculate all difficulty metrics."""
        cube = Cube(scramble)
        
        # Get cube coordinates for ML features
        coords = cube.get_coords(partial_corner_perm=False, partial_edge_perm=False)
        partial_coords = cube.get_coords(partial_corner_perm=True, partial_edge_perm=True)
        
        # Calculate all metrics
        manhattan_distance = self.calculate_manhattan_distance(cube)
        hamming_distance = self.calculate_hamming_distance(cube)
        orientation_distance = self.calculate_orientation_distance(cube)
        cycle_metrics = self.calculate_cycle_decomposition(cube)  
        thistlethwaite_distances = self.calculate_thistlethwaite_subgroup_distances(cube)  
        phase_distances = self.calculate_phase_distances(cube)
        solution_length, solution_variance = self.estimate_solution_length(cube)
        face_uniformity = self.calculate_face_color_uniformity(cube)
        color_clustering = self.calculate_color_clustering(cube)
        matched_pairs = self.calculate_matched_pairs(cube)
        
        # Calculate numerical difficulty scores
        manhattan_score = self.calculate_manhattan_score(manhattan_distance)
        hamming_score = self.calculate_hamming_score(hamming_distance['total_hamming'])
        orientation_score = self.calculate_orientation_score(orientation_distance['total_orientation'])
        solution_score = self.calculate_solution_score(solution_length)
        cycle_score = self.calculate_cycle_score(cycle_metrics['total_swap_cost'])  # NEW
        
        # Calculate overall score
        scores = {
            'manhattan': manhattan_score,
            'hamming': hamming_score,
            'orientation': orientation_score,
            'solution': solution_score,
            'cycle': cycle_score  # NEW
        }
        overall_score = self.calculate_overall_score(scores)
        
        # Categorize by each metric 
        manhattan_category = self.categorize_by_manhattan_distance(manhattan_distance)
        hamming_category = self.categorize_by_hamming_distance(hamming_distance['total_hamming'])
        orientation_category = self.categorize_by_orientation_distance(orientation_distance['total_orientation'])
        solution_category = self.categorize_by_solution_length(solution_length)
        overall_category = self.score_to_category(overall_score)
        
        return CubeDifficulty(
            scramble=scramble,
            cube=cube,
            manhattan_distance=manhattan_distance,
            hamming_distance=hamming_distance,
            orientation_distance=orientation_distance,
            cycle_metrics=cycle_metrics, 
            thistlethwaite_distances=thistlethwaite_distances,  
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
            overall_category=overall_category,
            manhattan_score=manhattan_score,
            hamming_score=hamming_score,
            orientation_score=orientation_score,
            solution_score=solution_score,
            overall_score=overall_score,
            coords=coords,
            partial_coords=partial_coords
        )
    
    def update_baselines_from_data(self, data_file = "results_per_instance.csv"):
        """Update normalization baselines from benchmark data."""
        try:
            import pandas as pd
            
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                
                # Update maximums based on empirical data
                if 'co' in df.columns:
                    self.MAX_MANHATTAN = max(df['manhattan_distance'].max() * 1.5, self.MAX_MANHATTAN)
                    self.MAX_HAMMING = max(df['hamming_distance'].max() * 1.5, self.MAX_HAMMING)
                    self.MAX_ORIENTATION = max(df['orientation_distance'].max() * 1.5, self.MAX_ORIENTATION)
                    self.MAX_SOLUTION_LENGTH = max(df['solution_length'].max() * 1.5, self.MAX_SOLUTION_LENGTH)
                    
                    print(f"Updated baselines from {data_file}:")
                    print(f" - Manhattan max: {self.MAX_MANHATTAN}")
                    print(f" - Hamming max: {self.MAX_HAMMING}")
                    print(f" - Orientation max: {self.MAX_ORIENTATION}")
                    print(f" - Solution max: {self.MAX_SOLUTION_LENGTH}")
        except Exception as e:
            print(f"Could not update baselines from {data_file}: {e}")
    
    def get_ml_features(self, cube_difficulty):
        """Extract features suitable for ML training."""
        features = {
            # Numerical difficulty scores
            'manhattan_score': cube_difficulty.manhattan_score,
            'hamming_score': cube_difficulty.hamming_score,
            'orientation_score': cube_difficulty.orientation_score,
            'solution_score': cube_difficulty.solution_score,
            'overall_score': cube_difficulty.overall_score,
            
            # Raw distances
            'manhattan_distance': cube_difficulty.manhattan_distance,
            'hamming_total': cube_difficulty.hamming_distance['total_hamming'],
            'orientation_total': cube_difficulty.orientation_distance['total_orientation'],
            'estimated_solution_length': cube_difficulty.estimated_solution_length,
            
            # Cycle decomposition features
            'corner_cycle_count': cube_difficulty.cycle_metrics['corner_cycle_count'],
            'edge_cycle_count': cube_difficulty.cycle_metrics['edge_cycle_count'],
            'total_swap_cost': cube_difficulty.cycle_metrics['total_swap_cost'],
            'corner_swap_cost': cube_difficulty.cycle_metrics['corner_swap_cost'],
            'edge_swap_cost': cube_difficulty.cycle_metrics['edge_swap_cost'],
            'max_corner_cycle': cube_difficulty.cycle_metrics['max_corner_cycle'],
            'max_edge_cycle': cube_difficulty.cycle_metrics['max_edge_cycle'],
            'avg_corner_cycle': cube_difficulty.cycle_metrics['avg_corner_cycle'],
            'avg_edge_cycle': cube_difficulty.cycle_metrics['avg_edge_cycle'],
            
            # Thistlethwaite subgroup distances
            'dist_to_G1': cube_difficulty.thistlethwaite_distances['dist_to_G1'],
            'dist_to_G2': cube_difficulty.thistlethwaite_distances['dist_to_G2'],
            'dist_to_G3': cube_difficulty.thistlethwaite_distances['dist_to_G3'],
            'bad_edge_orientation': cube_difficulty.thistlethwaite_distances['bad_edge_orientation'],
            'bad_corner_orientation': cube_difficulty.thistlethwaite_distances['bad_corner_orientation'],
            'bad_edge_slice': cube_difficulty.thistlethwaite_distances['bad_edge_slice'],
            'bad_corner_tetrad': cube_difficulty.thistlethwaite_distances['bad_corner_tetrad'],
            'corner_parity': cube_difficulty.thistlethwaite_distances['corner_parity'],
            
            # Cube coordinates
            'co': cube_difficulty.coords[0],
            'eo': cube_difficulty.coords[1],
            'cp': cube_difficulty.coords[2],
            'ep': cube_difficulty.coords[3],
            
            # Phase distances
            'has_phase_data': cube_difficulty.phase_distances.get('thistlethwaite_phases') is not None,
            
            # Face uniformity features
            'avg_face_uniformity': np.mean(list(cube_difficulty.face_uniformity.values())),
            'min_face_uniformity': np.min(list(cube_difficulty.face_uniformity.values())),
            'max_face_uniformity': np.max(list(cube_difficulty.face_uniformity.values())),
            
            # Color clustering features
            'avg_cluster_size': np.mean([cube_difficulty.color_clustering[face]['avg_cluster_size'] 
                                         for face in cube_difficulty.color_clustering]),
            'max_cluster_size': np.max([cube_difficulty.color_clustering[face]['max_cluster_size'] 
                                        for face in cube_difficulty.color_clustering]),
            
            # Categorical labels 
            'manhattan_category': cube_difficulty.manhattan_category,
            'hamming_category': cube_difficulty.hamming_category,
            'orientation_category': cube_difficulty.orientation_category,
            'solution_category': cube_difficulty.solution_category,
            'overall_category': cube_difficulty.overall_category
        }
        
        return features
    
    def generate_test_cubes(self, num_cubes = 10):
        """Generate a set of test cubes with different difficulty levels."""
        cubes = []
        
        # Generate cubes with different scramble lengths
        scramble_lengths = [3, 5, 8, 12, 15, 20, 25, 30, 35, 40]
        
        for i in range(num_cubes):
            length = scramble_lengths[i % len(scramble_lengths)]
            scramble = self.generate_scramble(length)
            cube_difficulty = self.analyze_cube(scramble)
            cubes.append(cube_difficulty)
        
        return cubes
    
    def group_by_metric(self, cubes, metric_name):
        """Group cubes by a specific metric."""
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
            elif metric_name == 'overall_score':
                # Group by score ranges
                if cube.overall_score < 0.2:
                    category = "Very Easy"
                elif cube.overall_score < 0.4:
                    category = "Easy"
                elif cube.overall_score < 0.6:
                    category = "Medium"
                elif cube.overall_score < 0.8:
                    category = "Hard"
                else:
                    category = "Expert"
            else:
                category = "Unknown"
            
            groups[category].append(cube)
        
        return dict(groups)
    
    def print_analysis_report(self, cubes):
        """Print a comprehensive analysis report."""
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
        manhattan_scores = [cube.manhattan_score for cube in cubes]
        print(f"\nManhattan Distance:")
        print(f"  Min: {min(manhattan_values)} (score: {min(manhattan_scores):.3f})")
        print(f"  Max: {max(manhattan_values)} (score: {max(manhattan_scores):.3f})")
        print(f"  Avg: {np.mean(manhattan_values):.2f} (score: {np.mean(manhattan_scores):.3f})")
        print(f"  Std: {np.std(manhattan_values):.2f}")
        
        # Hamming distance analysis
        hamming_values = [cube.hamming_distance['total_hamming'] for cube in cubes]
        hamming_scores = [cube.hamming_score for cube in cubes]
        print(f"\nHamming Distance:")
        print(f"  Min: {min(hamming_values)} (score: {min(hamming_scores):.3f})")
        print(f"  Max: {max(hamming_values)} (score: {max(hamming_scores):.3f})")
        print(f"  Avg: {np.mean(hamming_values):.2f} (score: {np.mean(hamming_scores):.3f})")
        print(f"  Std: {np.std(hamming_values):.2f}")
        
        # Orientation distance analysis
        orientation_values = [cube.orientation_distance['total_orientation'] for cube in cubes]
        orientation_scores = [cube.orientation_score for cube in cubes]
        print(f"\nOrientation Distance:")
        print(f"  Min: {min(orientation_values)} (score: {min(orientation_scores):.3f})")
        print(f"  Max: {max(orientation_values)} (score: {max(orientation_scores):.3f})")
        print(f"  Avg: {np.mean(orientation_values):.2f} (score: {np.mean(orientation_scores):.3f})")
        print(f"  Std: {np.std(orientation_values):.2f}")
        
        # Cycle decomposition analysis 
        cycle_swap_values = [cube.cycle_metrics['total_swap_cost'] for cube in cubes]
        max_corner_cycle = [cube.cycle_metrics['max_corner_cycle'] for cube in cubes]
        max_edge_cycle = [cube.cycle_metrics['max_edge_cycle'] for cube in cubes]
        print(f"\nCycle Decomposition:")
        print(f"  Swap Cost - Min: {min(cycle_swap_values)}, Max: {max(cycle_swap_values)}, Avg: {np.mean(cycle_swap_values):.2f}")
        print(f"  Max Corner Cycle - Min: {min(max_corner_cycle)}, Max: {max(max_corner_cycle)}, Avg: {np.mean(max_corner_cycle):.2f}")
        print(f"  Max Edge Cycle - Min: {min(max_edge_cycle)}, Max: {max(max_edge_cycle)}, Avg: {np.mean(max_edge_cycle):.2f}")
        
        # Thistlethwaite subgroup distances 
        dist_G1_values = [cube.thistlethwaite_distances['dist_to_G1'] for cube in cubes]
        dist_G2_values = [cube.thistlethwaite_distances['dist_to_G2'] for cube in cubes]
        dist_G3_values = [cube.thistlethwaite_distances['dist_to_G3'] for cube in cubes]
        print(f"\nThistlethwaite Subgroup Distances:")
        print(f"  Dist to G1 - Min: {min(dist_G1_values)}, Max: {max(dist_G1_values)}, Avg: {np.mean(dist_G1_values):.2f}")
        print(f"  Dist to G2 - Min: {min(dist_G2_values)}, Max: {max(dist_G2_values)}, Avg: {np.mean(dist_G2_values):.2f}")
        print(f"  Dist to G3 - Min: {min(dist_G3_values)}, Max: {max(dist_G3_values)}, Avg: {np.mean(dist_G3_values):.2f}")
        
        # Solution length analysis
        solution_values = [cube.estimated_solution_length for cube in cubes]
        solution_scores = [cube.solution_score for cube in cubes]
        print(f"\nEstimated Solution Length:")
        print(f"  Min: {min(solution_values)} (score: {min(solution_scores):.3f})")
        print(f"  Max: {max(solution_values)} (score: {max(solution_scores):.3f})")
        print(f"  Avg: {np.mean(solution_values):.2f} (score: {np.mean(solution_scores):.3f})")
        print(f"  Std: {np.std(solution_values):.2f}")
        
        # Overall score analysis
        overall_scores = [cube.overall_score for cube in cubes]
        print(f"\nOverall Difficulty Score:")
        print(f"  Min: {min(overall_scores):.3f}")
        print(f"  Max: {max(overall_scores):.3f}")
        print(f"  Avg: {np.mean(overall_scores):.3f}")
        print(f"  Std: {np.std(overall_scores):.3f}")
        
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
            print(f"  Overall Category: {cube.overall_category} (score: {cube.overall_score:.3f})")
            print(f"  Manhattan Distance: {cube.manhattan_distance} ({cube.manhattan_category}, score: {cube.manhattan_score:.3f})")
            print(f"  Hamming Distance: {cube.hamming_distance['total_hamming']} ({cube.hamming_category}, score: {cube.hamming_score:.3f})")
            print(f"  Orientation Distance: {cube.orientation_distance['total_orientation']} ({cube.orientation_category}, score: {cube.orientation_score:.3f})")
            print(f"  Estimated Solution Length: {cube.estimated_solution_length} ({cube.solution_category}, score: {cube.solution_score:.3f})")
            print(f"  Solution Variance: {cube.solution_variance:.2f}")
            
            # New metrics
            print(f"  Cycle Swap Cost: {cube.cycle_metrics['total_swap_cost']} (corners: {cube.cycle_metrics['corner_swap_cost']}, edges: {cube.cycle_metrics['edge_swap_cost']})")
            print(f"  Dist to G1: {cube.thistlethwaite_distances['dist_to_G1']}, G2: {cube.thistlethwaite_distances['dist_to_G2']}, G3: {cube.thistlethwaite_distances['dist_to_G3']}")
            
            # Cube coordinates
            print(f"  Cube Coordinates: co={cube.coords[0]}, eo={cube.coords[1]}, cp={cube.coords[2]}, ep={cube.coords[3]}")
            
            # Face uniformity
            avg_uniformity = np.mean(list(cube.face_uniformity.values()))
            print(f"  Average Face Uniformity: {avg_uniformity:.3f}")
            
            # Color clustering
            avg_clustering = np.mean([cube.color_clustering[face]['avg_cluster_size'] for face in cube.color_clustering])
            print(f"  Average Color Clustering: {avg_clustering:.2f}")


def main():
    """Main function to run the cube difficulty categorization."""
    categorizer = CubeDifficultyCategorizer()
    
    # Update baselines from benchmark data if available
    categorizer.update_baselines_from_data("results_per_instance.csv")
    
    print("Generating cube configurations...")
    cubes = categorizer.generate_test_cubes(num_cubes=10)
    
    print("Analyzing configurations...")
    categorizer.print_analysis_report(cubes)
    
    # Example of extracting ML features
    print("\n" + "=" * 50)
    print("EXAMPLE ML FEATURES EXTRACTION")
    print("=" * 50)
    
    for i, cube in enumerate(cubes[:3]):
        ml_features = categorizer.get_ml_features(cube)
        print(f"\nCube {i+1} ML Features:")
        for key, value in list(ml_features.items())[:15]: 
            print(f"  {key}: {value}")
    
    # Return cubes for further use
    return cubes


if __name__ == "__main__":
    cubes = main()