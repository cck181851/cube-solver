import warnings
warnings.filterwarnings('ignore')    

import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from src.ML_integration.cube_difficulty_categorizer import CubeDifficultyCategorizer
from src.cube_solver.cube.cube import Cube

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class CubeMLPredictor:
    """ML model to predict solver runtime, solution length, and nodes expanded for Rubik's cubes."""
    
    def __init__(self, data_file = "ML_training_data/training_data.csv"):
        self.data_file = data_file
        self.data = None
        self.X = None
        self.X_scaled_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Enhanced models dictionary
        self.models = {
            # Regression models
            'th_time': None,
            'th_moves': None,
            'th_nodes': None,
            'koc_time': None,
            'koc_moves': None,
            'koc_nodes': None,
            
            # Traditional classification
            'faster_solver': None,
            'shorter_solution': None,
            'less_nodes': None,
            
            # Kociemba difficulty classification
            'koc_nodes_class': None,
            'koc_time_class': None,
            'avoid_kociemba': None,
        }
        
        # Feature importance storage
        self.feature_importances = {}
        
        # Target variables
        self.targets = {}
        
        # Dataset size thresholds
        self.size_thresholds = {
            'very_small': 100,
            'small': 500,
            'medium': 2000,
            'large': 5000
        }
        
    def load_data(self, data_file = None):
        """Load and preprocess the training data."""
        if data_file is None:
            data_file = self.data_file
        
        print(f"Loading data from {data_file}...")
        self.data = pd.read_csv(data_file)
        print(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
        
        return self.data
    
    def get_dataset_size_category(self):
        """Determine the size category of the dataset for adaptive training."""
        if self.data is None:
            return "unknown"
        
        n = len(self.data)
        if n < self.size_thresholds['very_small']:
            return "very_small"
        elif n < self.size_thresholds['small']:
            return "small"
        elif n < self.size_thresholds['medium']:
            return "medium"
        else:
            return "large"
    
    def explore_data(self):
        """Explore the dataset and generate insights."""
        if self.data is None:
            print("No data loaded!")
            return
        
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        # Basic info
        dataset_size = len(self.data)
        size_category = self.get_dataset_size_category()
        print(f"\nDataset shape: {self.data.shape} (Size category: {size_category})")
        
        # Check for missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        print(f"\nMissing values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
        
        # Summary statistics for key columns
        key_columns = [
            'manhattan_score', 'hamming_score', 'orientation_score', 'overall_score',
            'th_moves', 'th_time_wall', 'th_nodes_expanded',
            'koc_moves', 'koc_time_wall', 'koc_nodes_expanded',
            'total_swap_cost', 'dist_to_G1', 'dist_to_G2', 'dist_to_G3'
        ]
        
        print(f"\nSummary statistics for key columns:")
        for col in key_columns:
            if col in self.data.columns:
                print(f"  {col}:")
                print(f"    Min: {self.data[col].min():.3f}")
                print(f"    Max: {self.data[col].max():.3f}")
                print(f"    Mean: {self.data[col].mean():.3f}")
                print(f"    Std: {self.data[col].std():.3f}")
        
        # Correlation analysis - adaptive based on dataset size
        size_category = self.get_dataset_size_category()
        
        if size_category in ["medium", "large"]:
            print(f"\nCorrelation of difficulty scores with performance:")
            difficulty_metrics = ['manhattan_score', 'hamming_score', 'orientation_score', 'overall_score']
            performance_metrics = ['th_moves', 'th_time_wall', 'th_nodes_expanded', 
                                  'koc_moves', 'koc_time_wall', 'koc_nodes_expanded']
            
            for diff_metric in difficulty_metrics:
                if diff_metric in self.data.columns:
                    print(f"\n  {diff_metric}:")
                    for perf_metric in performance_metrics:
                        if perf_metric in self.data.columns:
                            corr = self.data[diff_metric].corr(self.data[perf_metric])
                            print(f"    with {perf_metric}: {corr:.3f}")
        
        return self.data
    
    def prepare_features(self, feature_set = 'adaptive'):
        """Prepare features for ML training with adaptive feature selection."""
        if self.data is None:
            print("No data loaded!")
            return None
        
        dataset_size = len(self.data)
        size_category = self.get_dataset_size_category()
        
        print(f"\nPreparing features (Dataset: {dataset_size} records, Category: {size_category})...")
        
        # Adaptive feature selection based on dataset size
        if feature_set == 'adaptive':
            if size_category == "very_small":
                # For very small datasets, use only the most important features
                features = [
                    'overall_score', 'orientation_score', 'manhattan_score',
                    'total_swap_cost', 'dist_to_G1', 'dist_to_G2',
                    'estimated_solution_length', 'co', 'eo'
                ]
            elif size_category == "small":
                # For small datasets, use a balanced set
                features = [
                    'manhattan_score', 'hamming_score', 'orientation_score', 'overall_score',
                    'manhattan_distance', 'hamming_total', 'orientation_total',
                    'total_swap_cost', 'corner_swap_cost', 'edge_swap_cost',
                    'dist_to_G1', 'dist_to_G2', 'dist_to_G3',
                    'co', 'eo', 'cp', 'ep',
                    'estimated_solution_length'
                ]
            else:
                # For medium/large datasets, use all available features
                exclude_cols = [
                    'cube_id', 'scramble', 'scramble_length',
                    'th_success', 'th_moves', 'th_time_wall', 'th_nodes_expanded',
                    'koc_success', 'koc_moves', 'koc_time_wall', 'koc_nodes_expanded',
                    'th_time_cpu', 'th_memory_kb', 'th_table_lookups', 'th_pruned_nodes', 'th_error',
                    'koc_time_cpu', 'koc_memory_kb', 'koc_table_lookups', 'koc_pruned_nodes', 'koc_error',
                    'th_solution', 'koc_solution',
                    'is_th_faster', 'is_koc_shorter', 'is_th_less_nodes',
                    'log_th_time', 'log_koc_time', 'log_koc_nodes', 'log_th_nodes'
                ]
                features = [col for col in self.data.columns if col not in exclude_cols]
        else:
            # Use specified feature set
            if feature_set == 'basic':
                features = ['co', 'eo', 'cp', 'ep']
            elif feature_set == 'difficulty':
                features = ['manhattan_score', 'hamming_score', 'orientation_score', 'overall_score']
            elif feature_set == 'search':
                features = ['total_swap_cost', 'dist_to_G1', 'dist_to_G2', 'dist_to_G3']
            elif feature_set == 'all':
                exclude_cols = [
                    'cube_id', 'scramble', 'scramble_length',
                    'th_success', 'th_moves', 'th_time_wall', 'th_nodes_expanded',
                    'koc_success', 'koc_moves', 'koc_time_wall', 'koc_nodes_expanded',
                    'th_time_cpu', 'th_memory_kb', 'th_table_lookups', 'th_pruned_nodes', 'th_error',
                    'koc_time_cpu', 'koc_memory_kb', 'koc_table_lookups', 'koc_pruned_nodes', 'koc_error',
                    'th_solution', 'koc_solution',
                    'is_th_faster', 'is_koc_shorter', 'is_th_less_nodes',
                    'log_th_time', 'log_koc_time', 'log_koc_nodes', 'log_th_nodes'
                ]
                features = [col for col in self.data.columns if col not in exclude_cols]
            else:
                raise ValueError(f"Unknown feature_set: {feature_set}")
        
        # Filter out features that don't exist in the data
        features = [f for f in features if f in self.data.columns]
        
        print(f"Selected {len(features)} features")
        if size_category == "very_small" and len(features) < 20:
            print(f"Features: {features}")
        
        # Create X matrix
        self.X = self.data[features].copy()
        
        # Add interaction features for better prediction
        print("Adding interaction features...")
        
        # Basic interaction features that work well across all sizes
        if 'orientation_total' in self.X.columns and 'total_swap_cost' in self.X.columns:
            self.X['orientation_x_swap'] = self.X['orientation_total'] * self.X['total_swap_cost']
        
        if 'dist_to_G1' in self.X.columns and 'dist_to_G2' in self.X.columns:
            self.X['G1_x_G2'] = self.X['dist_to_G1'] * self.X['dist_to_G2']
        
        # For larger datasets, add more complex interactions
        if size_category in ["medium", "large"]:
            if 'manhattan_distance' in self.X.columns and 'estimated_solution_length' in self.X.columns:
                self.X['manhattan_x_solution'] = self.X['manhattan_distance'] * self.X['estimated_solution_length']
            
            if 'total_swap_cost' in self.X.columns and 'max_corner_cycle' in self.X.columns:
                self.X['swap_x_max_corner'] = self.X['total_swap_cost'] * self.X['max_corner_cycle']
        
        # Handle categorical features if any
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding {len(categorical_cols)} categorical features...")
            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                self.label_encoders[col] = le
        
        # Scale features
        print("Scaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        
        return self.X_scaled_df
    
    def prepare_targets(self, target_type = 'all'):
        """Prepare target variables for ML training."""
        if self.data is None:
            print("No data loaded!")
            return None
        
        print(f"\nPreparing targets...")
        
        # Create binary classification targets
        self.data['is_th_faster'] = (self.data['th_time_wall'] < self.data['koc_time_wall']).astype(int)
        self.data['is_koc_shorter'] = (self.data['koc_moves'] < self.data['th_moves']).astype(int)
        self.data['is_th_less_nodes'] = (self.data['th_nodes_expanded'] < self.data['koc_nodes_expanded']).astype(int)
        
        # Apply log transformation to skewed targets
        print("Applying log transformation to skewed targets...")
        self.data['log_th_time'] = np.log1p(self.data['th_time_wall'])
        self.data['log_koc_time'] = np.log1p(self.data['koc_time_wall'])
        self.data['log_koc_nodes'] = np.log1p(self.data['koc_nodes_expanded'])
        self.data['log_th_nodes'] = np.log1p(self.data['th_nodes_expanded'])
        
        # Define targets for different prediction tasks
        self.targets = {}
        
        if target_type in ['regression', 'all']:
            self.targets['th_time'] = self.data['log_th_time']
            self.targets['th_moves'] = self.data['th_moves']
            self.targets['th_nodes'] = self.data['log_th_nodes']
            self.targets['koc_time'] = self.data['log_koc_time']
            self.targets['koc_moves'] = self.data['koc_moves']
            self.targets['koc_nodes'] = self.data['log_koc_nodes']
        
        if target_type in ['classification', 'all']:
            self.targets['faster_solver'] = self.data['is_th_faster']
            self.targets['shorter_solution'] = self.data['is_koc_shorter']
            self.targets['less_nodes'] = self.data['is_th_less_nodes']
        
        print(f"Prepared {len(self.targets)} target variables")
        return self.targets
    
    def get_cv_folds(self):
        """Get appropriate number of CV folds based on dataset size."""
        dataset_size = len(self.data) if self.data is not None else 0
        if dataset_size < 50:
            return 3  # Minimum folds
        elif dataset_size < 200:
            return 5
        elif dataset_size < 1000:
            return 5
        else:
            return 5  # Standard 5-fold CV for larger datasets
    
    def train_regression_model_adaptive(self, X, y, target_name):
        """Train regression model with adaptive parameters based on dataset size."""
        size_category = self.get_dataset_size_category()
        
        print(f"Training regression model for {target_name} (Dataset: {size_category})...")
        
        # Split data with adaptive test size
        test_size = 0.3 if size_category == "very_small" else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Adaptive model parameters based on dataset size
        if size_category == "very_small":
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
        elif size_category == "small":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif target_name in ['koc_time', 'koc_nodes'] and size_category in ["medium", "large"]:
            # Use XGBoost for difficult targets with sufficient data
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R^2: {r2:.4f}")
        
        # Adaptive cross-validation
        cv_folds = self.get_cv_folds()
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        print(f"CV R^2 ({cv_folds}-fold): {cv_scores.mean():.4f} (+-{cv_scores.std():.4f})")
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_importances[target_name] = {
                'importances': importances,
                'features': feature_names
            }
        
        return model, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    def train_classification_model_adaptive(self, X, y, target_name):
        """Train classification model with adaptive parameters based on dataset size."""
        size_category = self.get_dataset_size_category()
        
        print(f"Training classification model for {target_name} (Dataset: {size_category})...")
        
        # Split data with adaptive test size and stratification
        test_size = 0.3 if size_category == "very_small" else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Adaptive model parameters
        if size_category == "very_small":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
        elif size_category == "small":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif size_category in ["medium", "large"]:
            # Use XGBoost for larger datasets
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Adaptive cross-validation
        cv_folds = self.get_cv_folds()
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        print(f"CV Accuracy ({cv_folds}-fold): {cv_scores.mean():.4f} (+-{cv_scores.std():.4f})")
        
        return model, {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std()
        }
    
    def train_all_models_enhanced(self):
        """Enhanced training with Kociemba difficulty classification."""
        if self.X is None:
            print("Features not prepared! Call prepare_features() first.")
            return
        
        # Use enhanced target preparation
        print("\nUsing enhanced target preparation with difficulty classification...")
        self.prepare_targets_enhanced(target_type='all')
        
        dataset_size = len(self.data) if self.data is not None else 0
        size_category = self.get_dataset_size_category()
        
        print("\n" + "="*80)
        print(f"ENHANCED MODEL TRAINING (Dataset: {dataset_size} records)")
        print("="*80)
        
        results = {}
        
        # Train regression models (with clipped + log-transformed targets)
        print("\n1. REGRESSION MODELS (with clipped outliers):")
        
        regression_targets = ['th_time', 'th_moves', 'th_nodes', 'koc_time', 'koc_moves', 'koc_nodes']
        for target in regression_targets:
            print(f"\n  {target.replace('_', ' ').title()}:")
            model, metrics = self.train_regression_model_adaptive(
                self.X_scaled_df, self.targets[target], target
            )
            self.models[target] = model
            results[target] = metrics
        
        # Train traditional classification models
        print("\n2. TRADITIONAL CLASSIFICATION MODELS:")
        
        traditional_class_targets = ['faster_solver', 'shorter_solution', 'less_nodes']
        for target in traditional_class_targets:
            print(f"\n  {target.replace('_', ' ').title()}:")
            model, metrics = self.train_classification_model_adaptive(
                self.X_scaled_df, self.targets[target], target
            )
            self.models[target] = model
            results[target] = metrics
        
        # Train Kociemba difficulty classification models
        print("\n3. KOCIEMBA DIFFICULTY CLASSIFICATION MODELS:")
        kociemba_class_results = self.train_kociemba_difficulty_models()
        results.update(kociemba_class_results)
        
        # Print enhanced summary
        print("\n" + "="*80)
        print("ENHANCED TRAINING SUMMARY")
        print("="*80)
        
        print(f"\nDataset Size: {dataset_size} records ({size_category})")
        
        # Kociemba difficulty distribution
        if 'avoid_kociemba' in self.targets:
            avoid_rate = self.targets['avoid_kociemba'].mean()
            print(f"\nKociemba Difficulty Analysis:")
            print(f"Cubes to avoid Kociemba: {avoid_rate:.1%}")
            print(f"Cubes where Kociemba is good: {(1-avoid_rate):.1%}")
        
        # Compare regression vs classification for Kociemba
        print("\nKociemba Prediction Strategy:")
        print(" - Use classification for: Should we use Kociemba?")
        print(" - Use regression for: If using Kociemba, how many nodes/time?")
        print(" - Classification accuracy > 85% is sufficient for solver selection")
        
        return results
    
    def evaluate_performance_adaptive(self):
        """Comprehensive evaluation of all models with adaptive metrics."""
        if self.X is None or len(self.targets) == 0:
            print("Features or targets not prepared!")
            return
        
        dataset_size = len(self.data) if self.data is not None else 0
        size_category = self.get_dataset_size_category()
        
        print("\n" + "="*80)
        print(f"COMPREHENSIVE MODEL EVALUATION (Dataset: {size_category})")
        print("="*80)
        
        results = {}
        
        regression_targets = ['th_time', 'th_moves', 'th_nodes', 'koc_time', 'koc_moves', 'koc_nodes']
        
        for target_name in regression_targets:
            if target_name in self.targets and self.models.get(target_name) is not None:
                print(f"\nEvaluating {target_name}...")
                
                # Adaptive test size
                test_size = 0.3 if size_category == "very_small" else 0.2
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X_scaled_df, self.targets[target_name], test_size=test_size, random_state=42
                )
                
                model = self.models[target_name]
                y_pred = model.predict(X_test)
                
                # Convert log-transformed predictions back to original scale
                if target_name in ['th_time', 'koc_time', 'koc_nodes', 'th_nodes']:
                    y_test_original = np.expm1(y_test)
                    y_pred_original = np.expm1(y_pred)
                else:
                    y_test_original = y_test
                    y_pred_original = y_pred
                
                mae = mean_absolute_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
                r2 = r2_score(y_test_original, y_pred_original)
                
                results[target_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R^2': r2,
                    'avg_actual': y_test_original.mean(),
                    'avg_predicted': y_pred_original.mean()
                }
                
                print(f"MAE: {mae:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R^2: {r2:.4f}")
        
        # Print summary table
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Target':<15} {'MAE':<10} {'RMSE':<10} {'R^2':<10} {'Avg Actual':<12} {'Avg Predicted':<12}")
        print("-" * 80)
        
        for target_name, metrics in results.items():
            print(f"{target_name:<15} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} "
                  f"{metrics['R^2']:<10.4f} {metrics['avg_actual']:<12.4f} {metrics['avg_predicted']:<12.4f}")
        
        # Performance assessment by dataset size
        print("\n" + "="*80)
        print("PERFORMANCE ASSESSMENT")
        print("="*80)
        
        if size_category == "very_small":
            print(" - With very small dataset (< 100):")
            print(" - Focus on classification accuracy and th_moves/koc_moves")
            print(" - R^2 < 0.3 for difficult targets is expected")
            print(" - Collect more data for better results")
        elif size_category == "small":
            print("With small dataset (< 500):")
            print(" - Good results for moves prediction expected")
            print(" - Moderate results for time/nodes prediction")
            print(" - R^2 0.3-0.6 for difficult targets is good")
        elif size_category == "medium":
            print("With medium dataset (500-2000):")
            print(" - Good results for most targets expected")
            print(" - R^2 > 0.5 for difficult targets is good")
        else:
            print("With large dataset (> 2000):")
            print(" - Excellent results for all targets expected")
            print(" - R^2 > 0.7 for difficult targets is achievable")
        
        return results
    
    def extract_features_from_cube(self, cube, scramble):
        """Extract features directly from a Cube object for prediction."""
        try:
            # Initialize categorizer
            categorizer = CubeDifficultyCategorizer()
            
            # Analyze the cube using the scramble
            cube_analysis = categorizer.analyze_cube(scramble)
            
            # Build a comprehensive feature dictionary
            features = {}
            
            # Essential features (always included)
            features['overall_score'] = cube_analysis.overall_score
            features['orientation_score'] = cube_analysis.orientation_score
            features['manhattan_score'] = cube_analysis.manhattan_score
            
            # Search complexity features
            features['total_swap_cost'] = cube_analysis.cycle_metrics['total_swap_cost']
            features['dist_to_G1'] = cube_analysis.thistlethwaite_distances['dist_to_G1']
            features['dist_to_G2'] = cube_analysis.thistlethwaite_distances['dist_to_G2']
            
            # Additional features if available
            if hasattr(cube_analysis, 'hamming_score'):
                features['hamming_score'] = cube_analysis.hamming_score
            
            if hasattr(cube_analysis, 'estimated_solution_length'):
                features['estimated_solution_length'] = cube_analysis.estimated_solution_length
            
            # Cube coordinates
            if hasattr(cube_analysis, 'coords'):
                features['co'] = cube_analysis.coords[0]
                features['eo'] = cube_analysis.coords[1]
            
            # Add interaction features
            if 'total_swap_cost' in features and 'dist_to_G1' in features:
                features['swap_x_G1'] = features['total_swap_cost'] * features['dist_to_G1']
            
            if 'orientation_score' in features and 'total_swap_cost' in features:
                features['orientation_x_swap'] = features['orientation_score'] * features['total_swap_cost']
            
            return pd.DataFrame([features])
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return minimal feature set
            return pd.DataFrame([{
                'overall_score': 0.5,
                'orientation_score': 0.5,
                'manhattan_score': 0.5,
                'total_swap_cost': 8,
                'dist_to_G1': 6,
                'dist_to_G2': 10,
                'co': 0,
                'eo': 0
            }])
    
    def predict_for_cube(self, cube, scramble):
        """Make predictions for a new cube with its scramble."""
        if any(model is None for model in self.models.values()):
            print("Models not trained! Train models first.")
            return self.get_fallback_predictions()
        
        print(f"\nMaking predictions for scramble: {scramble}")
        
        # Extract features from the cube
        try:
            features_df = self.extract_features_from_cube(cube, scramble)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return self.get_fallback_predictions()
        
        # Ensure all training columns are present
        if self.X is None:
            print("Training features not available!")
            return self.get_fallback_predictions()
        
        # Add missing columns with default values
        for col in self.X.columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Keep only columns that exist in training data
        features_df = features_df[self.X.columns]
        
        # Handle categorical features if any
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                try:
                    features_df[col] = self.label_encoders[col].transform(features_df[col])
                except:
                    features_df[col] = 0
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.X.columns)
        except Exception as e:
            print(f"Error scaling features: {e}")
            return self.get_fallback_predictions()
        
        # Make predictions
        predictions = {}
        
        # Regression predictions
        try:
            for target in ['th_time', 'th_moves', 'th_nodes', 'koc_time', 'koc_moves', 'koc_nodes']:
                if self.models[target] is not None:
                    pred = self.models[target].predict(features_scaled_df)[0]
                    
                    # Convert from log scale back to original if needed
                    if target in ['th_time', 'koc_time', 'koc_nodes', 'th_nodes']:
                        pred = np.expm1(pred)
                    
                    # Set reasonable bounds
                    if 'time' in target:
                        predictions[f'{target}_pred'] = max(0.001, min(10.0, pred))
                    elif 'nodes' in target:
                        predictions[f'{target}_pred'] = max(1, min(100000, int(pred)))
                    else:  # moves
                        predictions[f'{target}_pred'] = max(1, min(100, int(pred)))
        except Exception as e:
            print(f"Error making regression predictions: {e}")
        
        # Classification predictions
        try:
            for target in ['faster_solver', 'shorter_solution', 'less_nodes']:
                if self.models[target] is not None:
                    pred = self.models[target].predict(features_scaled_df)[0]
                    prob = self.models[target].predict_proba(features_scaled_df)[0]
                    
                    if target == 'faster_solver':
                        predictions[target] = 'Thistlethwaite' if pred == 1 else 'Kociemba'
                        predictions['faster_confidence'] = float(max(prob))
                    elif target == 'shorter_solution':
                        predictions[target] = 'Kociemba' if pred == 1 else 'Thistlethwaite'
                        predictions['shorter_confidence'] = float(max(prob))
                    else:  
                        predictions[target] = 'Thistlethwaite' if pred == 1 else 'Kociemba'
                        predictions['nodes_confidence'] = float(max(prob))
        except Exception as e:
            print(f"Error making classification predictions: {e}")
        
        return predictions
    
    def get_fallback_predictions(self):
        """Return fallback predictions based on training data averages."""
        if self.data is None:
            # Default averages
            return {
                'th_time_pred': 0.018,
                'th_moves_pred': 28,
                'th_nodes_pred': 144,
                'koc_time_pred': 0.207,
                'koc_moves_pred': 19,
                'koc_nodes_pred': 4386,
                'faster_solver': 'Thistlethwaite',
                'faster_confidence': 0.86,
                'shorter_solution': 'Kociemba',
                'shorter_confidence': 0.96,
                'less_nodes': 'Thistlethwaite',
                'nodes_confidence': 0.80
            }
        
        # Calculate averages from training data
        return {
            'th_time_pred': float(self.data['th_time_wall'].mean()),
            'th_moves_pred': int(self.data['th_moves'].mean()),
            'th_nodes_pred': int(self.data['th_nodes_expanded'].mean()),
            'koc_time_pred': float(self.data['koc_time_wall'].mean()),
            'koc_moves_pred': int(self.data['koc_moves'].mean()),
            'koc_nodes_pred': int(self.data['koc_nodes_expanded'].mean()),
            'faster_solver': 'Thistlethwaite',
            'faster_confidence': float((self.data['th_time_wall'] < self.data['koc_time_wall']).mean()),
            'shorter_solution': 'Kociemba',
            'shorter_confidence': float((self.data['koc_moves'] < self.data['th_moves']).mean()),
            'less_nodes': 'Thistlethwaite',
            'nodes_confidence': float((self.data['th_nodes_expanded'] < self.data['koc_nodes_expanded']).mean())
        }
    
    def save_models(self, directory = "ML_integration_data/saved_models"):
        """Save trained models to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            if model is not None:
                filename = os.path.join(directory, f"{model_name}_{timestamp}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Saved {model_name} to {filename}")
        
        # Save scaler and feature info
        scaler_file = os.path.join(directory, f"scaler_{timestamp}.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        feature_file = os.path.join(directory, f"features_{timestamp}.pkl")
        with open(feature_file, 'wb') as f:
            pickle.dump({
                'feature_names': list(self.X.columns) if self.X is not None else [],
                'label_encoders': self.label_encoders
            }, f)
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory = "ML_integration_data/saved_models"):
        """Load trained models from disk."""
        import glob
        
        model_files = glob.glob(os.path.join(directory, "*.pkl"))
        
        for filename in model_files:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                
            # Extract model name from filename
            basename = os.path.basename(filename)
            for model_name in self.models.keys():
                if basename.startswith(f"{model_name}_"):
                    self.models[model_name] = model
                    break
                elif basename.startswith('scaler_'):
                    self.scaler = model
                elif basename.startswith('features_'):
                    feature_info = model
                    if 'feature_names' in feature_info:
                        self.X = pd.DataFrame(columns=feature_info['feature_names'])
                    if 'label_encoders' in feature_info:
                        self.label_encoders = feature_info['label_encoders']
        
        print(f"Loaded models from {directory}")
    
    def plot_feature_importance(self, target_name = 'koc_nodes'):
        """Plot feature importance for a specific target model."""
        if target_name not in self.models or self.models[target_name] is None:
            print(f"No model trained for {target_name}")
            return
        
        model = self.models[target_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model for {target_name} doesn't have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'feature_{i}' for i in range(self.X.shape[1])]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot top features (adaptive based on dataset size)
        dataset_size = len(self.data) if self.data is not None else 0
        top_n = min(15, len(feature_names)) if dataset_size < 500 else min(20, len(feature_names))
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances for {target_name} (Dataset: {dataset_size} records)")
        plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        plt.savefig(f'ML_integration_data/plots/feature_importance_{target_name}.png', dpi=150)
        plt.show()
        
        # Print top features
        print(f"\nTop {top_n} features for {target_name}:")
        for i in range(top_n):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def plot_predictions_vs_actual(self, target_name = 'th_moves'):
        """Plot predicted vs actual values for a target."""
        if target_name not in self.targets or self.models.get(target_name) is None:
            print(f"Model for {target_name} not available!")
            return
        
        dataset_size = len(self.data) if self.data is not None else 0
        
        # Get predictions for all data
        y_pred = self.models[target_name].predict(self.X_scaled_df)
        y_actual = self.targets[target_name].values
        
        # Convert from log scale if needed
        if target_name in ['th_time', 'koc_time', 'koc_nodes', 'th_nodes']:
            y_pred = np.expm1(y_pred)
            y_actual = np.expm1(y_actual)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted for {target_name} (Dataset: {dataset_size} records)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics
        r2 = r2_score(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        plt.text(0.05, 0.95, f'R^2 = {r2:.3f}\nMAE = {mae:.3f}\nN = {dataset_size}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        if not os.path.exists('ML_integration_data/plots'):
            os.makedirs('ML_integration_data/plots')
        
        plt.savefig(f'ML_integration_data/plots/pred_vs_actual_{target_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def prepare_targets_enhanced(self, target_type = 'all'):
        """Enhanced target preparation with Kociemba difficulty classification."""
        if self.data is None:
            print("No data loaded!")
            return None
        
        print(f"\nPreparing enhanced targets for Kociemba prediction...")
        
        # Create binary classification targets
        self.data['is_th_faster'] = (self.data['th_time_wall'] < self.data['koc_time_wall']).astype(int)
        self.data['is_koc_shorter'] = (self.data['koc_moves'] < self.data['th_moves']).astype(int)
        self.data['is_th_less_nodes'] = (self.data['th_nodes_expanded'] < self.data['koc_nodes_expanded']).astype(int)
        
        # Create Kociemba difficulty classes
        print("  Creating Kociemba difficulty classes...")
        
        # For Kociemba nodes: create 4 difficulty classes
        koc_nodes = self.data['koc_nodes_expanded']
        
        # Use quantile-based bins for balanced classes
        nodes_bins = [
            0,
            koc_nodes.quantile(0.25),      # Q1
            koc_nodes.quantile(0.50),      # Median
            koc_nodes.quantile(0.75),      # Q3
            koc_nodes.max() + 1           # Max
        ]
        
        # Clip extreme outliers for regression
        NODES_CLIP_MAX = 20000
        self.data['koc_nodes_clipped'] = koc_nodes.clip(upper=NODES_CLIP_MAX)
        
        # Create node difficulty classes
        self.data['koc_nodes_class'] = pd.cut(
            koc_nodes,
            bins=nodes_bins,
            labels=['very_easy', 'easy', 'medium', 'hard'],
            include_lowest=True
        )
        
        # For Kociemba time: create 4 difficulty classes
        koc_time = self.data['koc_time_wall']
        
        time_bins = [
            0,
            koc_time.quantile(0.25),      # Q1
            koc_time.quantile(0.50),      # Median
            koc_time.quantile(0.75),      # Q3
            koc_time.max() + 0.001       # Max
        ]
        
        # Clip extreme outliers for regression
        TIME_CLIP_MAX = 1.0
        self.data['koc_time_clipped'] = koc_time.clip(upper=TIME_CLIP_MAX)
        
        # Create time difficulty classes
        self.data['koc_time_class'] = pd.cut(
            koc_time,
            bins=time_bins,
            labels=['very_fast', 'fast', 'medium', 'slow'],
            include_lowest=True
        )
        
        # Create a combined difficulty indicator
        # This is a binary classification: "should we use Kociemba or avoid it?"
        # Avoid Kociemba if either nodes are hard or time is slow
        self.data['avoid_kociemba'] = (
            (self.data['koc_nodes_class'].isin(['hard', 'medium'])) |
            (self.data['koc_time_class'].isin(['slow', 'medium']))
        ).astype(int)
        
        print(f"Kociemba difficulty distribution:")
        print(f"Nodes classes: {self.data['koc_nodes_class'].value_counts().to_dict()}")
        print(f"Time classes: {self.data['koc_time_class'].value_counts().to_dict()}")
        print(f"Avoid Kociemba: {self.data['avoid_kociemba'].mean():.1%} of cubes")
        
        # Apply log1p transformation
        print("Applying log transformation to clipped targets...")
        
        # For regression: use clipped + log-transformed values
        self.data['log_koc_nodes'] = np.log1p(self.data['koc_nodes_clipped'])
        self.data['log_koc_time'] = np.log1p(self.data['koc_time_clipped'])
        
        # Also log-transform Thistlethwaite for consistency
        self.data['log_th_time'] = np.log1p(self.data['th_time_wall'])
        self.data['log_th_nodes'] = np.log1p(self.data['th_nodes_expanded'])
        
        # Define targets for different prediction tasks
        self.targets = {}
        
        if target_type in ['regression', 'all']:
            # For regression: use clipped + log-transformed values
            self.targets['th_time'] = self.data['log_th_time']
            self.targets['th_moves'] = self.data['th_moves']
            self.targets['th_nodes'] = self.data['log_th_nodes']
            self.targets['koc_time'] = self.data['log_koc_time']
            self.targets['koc_moves'] = self.data['koc_moves']
            self.targets['koc_nodes'] = self.data['log_koc_nodes']
        
        if target_type in ['classification', 'all']:
            # Enhanced classification targets
            self.targets['faster_solver'] = self.data['is_th_faster']
            self.targets['shorter_solution'] = self.data['is_koc_shorter']
            self.targets['less_nodes'] = self.data['is_th_less_nodes']
            # Kociemba difficulty classification targets
            self.targets['koc_nodes_class'] = self.data['koc_nodes_class']
            self.targets['koc_time_class'] = self.data['koc_time_class']
            self.targets['avoid_kociemba'] = self.data['avoid_kociemba']
        
        print(f"Prepared {len(self.targets)} target variables")
        print(f"New classification targets: koc_nodes_class, koc_time_class, avoid_kociemba")
        
        return self.targets


    def train_kociemba_difficulty_models(self):
        """Specialized training for Kociemba difficulty classification."""
        if self.X is None or 'koc_nodes_class' not in self.targets:
            print("Features or Kociemba targets not prepared!")
            return
        
        print("\n" + "="*80)
        print("TRAINING KOCIEMBA DIFFICULTY CLASSIFICATION MODELS")
        print("="*80)
        
        results = {}
        
        # Important: For classification, we need to handle class imbalance
        kociemba_class_targets = ['koc_nodes_class', 'koc_time_class', 'avoid_kociemba']
        
        for target_name in kociemba_class_targets:
            print(f"\nTraining {target_name} classifier...")
            
            y = self.targets[target_name]
            
            # Check if we need label encoding for categorical targets
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                class_names = le.classes_
            else:
                y_encoded = y
                class_names = None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # For binary classification (avoid_kociemba), use XGBoost
            if target_name == 'avoid_kociemba':
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    random_state=42,
                    verbosity=0
                )
            else:
                # For multi-class, use RandomForest with class weighting
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42
                )
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            # Store model
            self.models[target_name] = model
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            
            if class_names is not None:
                print(f"Classes: {class_names.tolist()}")
                cm = confusion_matrix(y_test, y_pred)
                print(f"Confusion matrix:\n{cm}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_scaled_df, y_encoded, cv=5, scoring='accuracy')
            print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            results[target_name] = {
                'accuracy': accuracy,
                'cv_accuracy': cv_scores.mean(),
                'model': model
            }
        
        return results

    def predict_kociemba_difficulty(self, cube, scramble):
        """Predict Kociemba difficulty for a specific cube."""
        if any(model is None for model in [self.models.get('koc_nodes_class'), 
                                        self.models.get('koc_time_class'),
                                        self.models.get('avoid_kociemba')]):
            print("Kociemba difficulty models not trained!")
            return self.get_fallback_kociemba_difficulty()
        
        print(f"\nPredicting Kociemba difficulty for scramble: {scramble[:50]}...")
        
        # Extract features
        try:
            features_df = self.extract_features_from_cube(cube, scramble)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return self.get_fallback_kociemba_difficulty()
        
        # Prepare features as in training
        for col in self.X.columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        features_df = features_df[self.X.columns]
        
        # Handle categorical features
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                try:
                    features_df[col] = self.label_encoders[col].transform(features_df[col])
                except:
                    features_df[col] = 0
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.X.columns)
        except Exception as e:
            print(f"Error scaling features: {e}")
            return self.get_fallback_kociemba_difficulty()
        
        # Make predictions
        predictions = {}
        
        # Predict difficulty classes
        try:
            # Nodes difficulty class
            if self.models['koc_nodes_class'] is not None:
                nodes_class_pred = self.models['koc_nodes_class'].predict(features_scaled_df)[0]
                nodes_class_prob = self.models['koc_nodes_class'].predict_proba(features_scaled_df)[0]
                
                # Map back to original class names if needed
                predictions['koc_nodes_difficulty'] = nodes_class_pred
                predictions['koc_nodes_confidence'] = float(max(nodes_class_prob))
                
                # Interpret the class
                if nodes_class_pred in ['hard', 'medium']:
                    predictions['koc_nodes_advice'] = 'Avoid Kociemba - high node count expected'
                else:
                    predictions['koc_nodes_advice'] = 'Kociemba should be efficient'
            
            # Time difficulty class
            if self.models['koc_time_class'] is not None:
                time_class_pred = self.models['koc_time_class'].predict(features_scaled_df)[0]
                time_class_prob = self.models['koc_time_class'].predict_proba(features_scaled_df)[0]
                
                predictions['koc_time_difficulty'] = time_class_pred
                predictions['koc_time_confidence'] = float(max(time_class_prob))
                
                if time_class_pred in ['slow', 'medium']:
                    predictions['koc_time_advice'] = 'Avoid Kociemba - slow solve expected'
                else:
                    predictions['koc_time_advice'] = 'Kociemba should be fast'
            
            # Combined avoidance recommendation
            if self.models['avoid_kociemba'] is not None:
                avoid_pred = self.models['avoid_kociemba'].predict(features_scaled_df)[0]
                avoid_prob = self.models['avoid_kociemba'].predict_proba(features_scaled_df)[0]
                
                predictions['avoid_kociemba'] = bool(avoid_pred)
                predictions['avoid_confidence'] = float(max(avoid_prob))
                
                if avoid_pred == 1:
                    predictions['solver_recommendation'] = 'Use Thistlethwaite instead of Kociemba'
                else:
                    predictions['solver_recommendation'] = 'Kociemba should work well'
            
        except Exception as e:
            print(f"Error making Kociemba difficulty predictions: {e}")
        
        return predictions


    def get_fallback_kociemba_difficulty(self):
        """Return fallback Kociemba difficulty predictions."""
        if self.data is None:
            return {
                'koc_nodes_difficulty': 'medium',
                'koc_nodes_confidence': 0.5,
                'koc_nodes_advice': 'Insufficient data for prediction',
                'koc_time_difficulty': 'medium',
                'koc_time_confidence': 0.5,
                'koc_time_advice': 'Insufficient data for prediction',
                'avoid_kociemba': False,
                'avoid_confidence': 0.5,
                'solver_recommendation': 'Use default solver'
            }
        
        # Calculate class distribution from training data
        if 'koc_nodes_class' in self.data.columns:
            nodes_class_dist = self.data['koc_nodes_class'].value_counts(normalize=True)
            most_common_nodes = nodes_class_dist.idxmax()
            nodes_confidence = nodes_class_dist.max()
        else:
            most_common_nodes = 'medium'
            nodes_confidence = 0.5
        
        if 'koc_time_class' in self.data.columns:
            time_class_dist = self.data['koc_time_class'].value_counts(normalize=True)
            most_common_time = time_class_dist.idxmax()
            time_confidence = time_class_dist.max()
        else:
            most_common_time = 'medium'
            time_confidence = 0.5
        
        avoid_rate = self.data['avoid_kociemba'].mean() if 'avoid_kociemba' in self.data.columns else 0.5
        
        return {
            'koc_nodes_difficulty': most_common_nodes,
            'koc_nodes_confidence': float(nodes_confidence),
            'koc_nodes_advice': f'Based on training data: {most_common_nodes} difficulty',
            'koc_time_difficulty': most_common_time,
            'koc_time_confidence': float(time_confidence),
            'koc_time_advice': f'Based on training data: {most_common_time} difficulty',
            'avoid_kociemba': avoid_rate > 0.5,
            'avoid_confidence': float(max(avoid_rate, 1-avoid_rate)),
            'solver_recommendation': 'Use Thistlethwaite for hard cubes, Kociemba for easy ones'
        }


    def evaluate_kociemba_difficulty_models(self):
        """Evaluate the Kociemba difficulty classification models."""
        if self.X is None:
            print("Features not prepared!")
            return
        
        print("\n" + "="*80)
        print("KOCIEMBA DIFFICULTY CLASSIFICATION EVALUATION")
        print("="*80)
        
        results = {}
        
        # Evaluate classification models
        kociemba_class_targets = ['koc_nodes_class', 'koc_time_class', 'avoid_kociemba']
        
        for target_name in kociemba_class_targets:
            if target_name not in self.models or self.models[target_name] is None:
                continue
            
            print(f"\nEvaluating {target_name}...")
            
            model = self.models[target_name]
            y_true = self.targets[target_name]
            
            # Encode if categorical
            if y_true.dtype == 'object' or y_true.dtype.name == 'category':
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                class_names = le.classes_
            else:
                y_true_encoded = y_true
                class_names = ['False', 'True'] if target_name == 'avoid_kociemba' else None
            
            # Get predictions for all data
            y_pred = model.predict(self.X_scaled_df)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_encoded, y_pred)
            
            if class_names is not None:
                cm = confusion_matrix(y_true_encoded, y_pred)
                cr = classification_report(y_true_encoded, y_pred, target_names=class_names)
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Confusion Matrix:\n{cm}")
                print(f"Classification Report:\n{cr}")
            else:
                print(f"Accuracy: {accuracy:.4f}")
            
            # Store results
            results[target_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm if 'cm' in locals() else None
            }
        
        # Compare with regression performance
        print("\n" + "="*80)
        print("COMPARISON: CLASSIFICATION vs REGRESSION FOR KOCIEMBA")
        print("="*80)
        
        print("\nRegression Performance (R² scores):")
        print(f"  koc_nodes (regression): {self.evaluate_single_model('koc_nodes').get('r2', 0):.4f}")
        print(f"  koc_time (regression): {self.evaluate_single_model('koc_time').get('r2', 0):.4f}")
        
        print("\nClassification Performance (Accuracy):")
        for target_name in ['koc_nodes_class', 'koc_time_class', 'avoid_kociemba']:
            if target_name in results:
                print(f"  {target_name}: {results[target_name]['accuracy']:.4f}")
        
        print("\nRecommendation:")
        print("Use classification models for solver selection decisions")
        print("Use regression models only for estimating exact values when classification says 'easy'")
        
        return results


    def evaluate_single_model(self, target_name):
        """Helper to evaluate a single regression model."""
        if target_name not in self.models or self.models[target_name] is None:
            return {'r2': 0, 'mae': 0}
        
        model = self.models[target_name]
        y_true = self.targets[target_name]
        
        # Predict
        y_pred = model.predict(self.X_scaled_df)
        
        # Convert from log scale if needed
        if target_name in ['th_time', 'koc_time', 'koc_nodes', 'th_nodes']:
            y_true_original = np.expm1(y_true)
            y_pred_original = np.expm1(y_pred)
        else:
            y_true_original = y_true
            y_pred_original = y_pred
        
        # Calculate metrics
        r2 = r2_score(y_true_original, y_pred_original)
        mae = mean_absolute_error(y_true_original, y_pred_original)
        
        return {'r2': r2, 'mae': mae}
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for classification models."""
        print("Generating confusion matrix plots...")
        
        classification_targets = ['koc_nodes_class', 'koc_time_class', 'avoid_kociemba']
        
        for target_name in classification_targets:
            if target_name not in self.models or self.models[target_name] is None:
                continue
            
            print(f"Plotting confusion matrix for {target_name}...")
            
            model = self.models[target_name]
            y_true = self.targets[target_name]
            
            # Encode if categorical
            if y_true.dtype == 'object' or y_true.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                class_names = le.classes_
            else:
                y_true_encoded = y_true
                class_names = ['False', 'True'] if target_name == 'avoid_kociemba' else ['Class 0', 'Class 1']
            
            # Get predictions
            y_pred = model.predict(self.X_scaled_df)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_encoded, y_pred)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            if target_name == 'avoid_kociemba':
                # Binary classification
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Use Kociemba', 'Avoid Kociemba'],
                        yticklabels=['Use Kociemba', 'Avoid Kociemba'])
                plt.title(f'Confusion Matrix: When to Avoid Kociemba\nAccuracy: {accuracy_score(y_true_encoded, y_pred):.3f}')
            else:
                # Multi-class classification
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
                plt.title(f'Confusion Matrix: {target_name}\nAccuracy: {accuracy_score(y_true_encoded, y_pred):.3f}')
            
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save plot
            if not os.path.exists('ML_integration_data/plots'):
                os.makedirs('ML_integration_data/plots')
            
            plt.savefig(f'ML_integration_data/plots/confusion_matrix_{target_name}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Print classification report
            from sklearn.metrics import classification_report
            print(f"\nClassification Report for {target_name}:")
            if target_name == 'avoid_kociemba':
                print(classification_report(y_true_encoded, y_pred, 
                                        target_names=['Use Kociemba', 'Avoid Kociemba']))
            else:
                print(classification_report(y_true_encoded, y_pred, target_names=class_names))    


def main():
    """Main function with enhanced Kociemba prediction."""
    print("=" * 80)
    print("ENHANCED RUBIK'S CUBE SOLVER ML PREDICTOR")
    print("=" * 80)

    os.makedirs("ML_integration_data/plots", exist_ok=True)
    os.makedirs("ML_integration_data/saved_models", exist_ok=True)
    
    # Initialize predictor
    predictor = CubeMLPredictor(data_file="ML_integration_data/training_data.csv")
    
    # Load data
    data = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Prepare adaptive features
    predictor.prepare_features(feature_set='adaptive')
    
    # Train enhanced models
    results = predictor.train_all_models_enhanced()
    
    # Evaluate performance
    predictor.evaluate_performance_adaptive()
    
    # Evaluate Kociemba difficulty models
    predictor.evaluate_kociemba_difficulty_models()
    
    # Save models
    predictor.save_models()
    
    # Plot feature importance for key targets
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE PLOTS")
    print("="*80)
    
    for target in ['koc_nodes', 'koc_time', 'th_moves']:
        try:
            predictor.plot_feature_importance(target)
        except Exception as e:
            print(f"Could not plot feature importance for {target}: {e}")
    
    # Plot predictions vs actual
    print("\n" + "="*80)
    print("GENERATING PREDICTION VS ACTUAL PLOTS")
    print("="*80)
    
    for target in ['th_moves', 'koc_moves', 'th_nodes', 'koc_nodes']:
        try:
            predictor.plot_predictions_vs_actual(target)
        except Exception as e:
            print(f"Could not plot predictions for {target}: {e}")
    
    # Plot confusion matrices for classification models
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX PLOTS")
    print("="*80)
    
    predictor.plot_confusion_matrices()
    
    # Test with a new cube - show both old and new predictions
    print("\n" + "="*80)
    print("TESTING ENHANCED PREDICTIONS")
    print("="*80)
    
    # Create a test cube
    categorizer = CubeDifficultyCategorizer()
    test_scramble = categorizer.generate_scramble(20)
    test_cube = Cube(test_scramble)
    
    # Make traditional predictions
    print(f"\nTraditional predictions for scramble: {test_scramble}")
    traditional_predictions = predictor.predict_for_cube(test_cube, test_scramble)
    
    for key, value in traditional_predictions.items():
        if 'confidence' in key:
            print(f"  {key}: {value:.2%}")
        elif 'pred' in key:
            if 'time' in key:
                print(f"  {key}: {value:.4f}s")
            elif 'nodes' in key:
                print(f"  {key}: {value:,} nodes")
            else:
                print(f"  {key}: {value} moves")
        else:
            print(f"  {key}: {value}")
    
    # Make enhanced Kociemba difficulty predictions
    print(f"\nEnhanced Kociemba difficulty predictions:")
    kociemba_difficulty = predictor.predict_kociemba_difficulty(test_cube, test_scramble)
    
    for key, value in kociemba_difficulty.items():
        if 'confidence' in key:
            print(f"  {key}: {value:.2%}")
        elif isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Decision making based on enhanced predictions
    print(f"\nSolver Selection Decision:")
    
    if 'avoid_kociemba' in kociemba_difficulty:
        if kociemba_difficulty['avoid_kociemba']:
            print(f"RECOMMENDATION: Use Thistlethwaite solver")
            print(f"Reason: {kociemba_difficulty.get('solver_recommendation', 'High difficulty predicted')}")
        else:
            print(f"RECOMMENDATION: Use Kociemba solver")
            print(f"Reason: {kociemba_difficulty.get('solver_recommendation', 'Low difficulty predicted')}")
    
    print("\n" + "="*80)
    print("ENHANCED ML TRAINING COMPLETE!")
    print("="*80)
    
    return predictor


if __name__ == "__main__":
    try:
        predictor = main()
    except Exception as e:
        print(f"\n Error during enhanced ML training: {e}")
        import traceback
        traceback.print_exc()