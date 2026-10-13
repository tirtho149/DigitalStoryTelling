import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
from math import sqrt
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_descriptive_statistics(data, metric_name):
    """Calculate comprehensive descriptive statistics for a given metric"""
    statistics = {
        "Count": len(data),
        "Mean": np.mean(data),
        "Standard Deviation": np.std(data, ddof=1),
        "Minimum": np.min(data),
        "First Quartile (Q1)": np.percentile(data, 25),
        "Second Quartile (Median)": np.percentile(data, 50),
        "Third Quartile (Q3)": np.percentile(data, 75),
        "Maximum": np.max(data),
        "Interquartile Range (IQR)": np.percentile(data, 75) - np.percentile(data, 25),
        "Skewness": skew(data),
        "Kurtosis": kurtosis(data),
        "5th Percentile": np.percentile(data, 5),
        "95th Percentile": np.percentile(data, 95),
        "Variance": np.var(data, ddof=1),
        "Coefficient of Variation": (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else 0
    }
    return pd.DataFrame(statistics.items(), columns=["Statistic", metric_name])

def calculate_feature_importance_statistics(all_feature_importances, feature_cols):
    """Calculate comprehensive statistics for feature importances across all iterations"""
    # Convert to numpy array for easier manipulation
    fi_array = np.array(all_feature_importances)
    
    # Calculate statistics for each feature across all iterations
    fi_stats = []
    for i, feature in enumerate(feature_cols):
        feature_values = fi_array[:, i]
        stats = {
            "Feature": feature,
            "Mean_Importance": np.mean(feature_values),
            "Std_Importance": np.std(feature_values, ddof=1),
            "Min_Importance": np.min(feature_values),
            "Q1_Importance": np.percentile(feature_values, 25),
            "Median_Importance": np.percentile(feature_values, 50),
            "Q3_Importance": np.percentile(feature_values, 75),
            "Max_Importance": np.max(feature_values),
            "IQR_Importance": np.percentile(feature_values, 75) - np.percentile(feature_values, 25),
            "CV_Importance": (np.std(feature_values, ddof=1) / np.mean(feature_values)) * 100 if np.mean(feature_values) != 0 else 0,
            "Skewness": skew(feature_values),
            "Kurtosis": kurtosis(feature_values),
            "Rank_by_Mean": 0  # Will be filled later
        }
        fi_stats.append(stats)
    
    # Create DataFrame and add ranking
    fi_df = pd.DataFrame(fi_stats)
    fi_df = fi_df.sort_values('Mean_Importance', ascending=False).reset_index(drop=True)
    fi_df['Rank_by_Mean'] = range(1, len(fi_df) + 1)
    
    return fi_df

def create_comprehensive_tables(iteration, r2_scores, rmse_scores, mae_scores, 
                               all_feature_importances, feature_cols, save_dir="../results/random_forest_run", study_name=""):
    """Create comprehensive statistical tables with study-specific naming"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Add study name prefix to files if provided
    prefix = f"{study_name}_" if study_name else ""
    
    # 1. R² Score Statistics
    r2_stats = calculate_descriptive_statistics(r2_scores, "R2_Score")
    r2_stats.to_csv(f"{save_dir}/{prefix}r2_statistics_iter{iteration}.csv", index=False)
    
    # 2. RMSE Statistics
    rmse_stats = calculate_descriptive_statistics(rmse_scores, "RMSE")
    rmse_stats.to_csv(f"{save_dir}/{prefix}rmse_statistics_iter{iteration}.csv", index=False)
    
    # 3. MAE Statistics
    mae_stats = calculate_descriptive_statistics(mae_scores, "MAE")
    mae_stats.to_csv(f"{save_dir}/{prefix}mae_statistics_iter{iteration}.csv", index=False)
    
    # 4. Combined Performance Metrics Summary
    combined_stats = pd.DataFrame({
        'Statistic': r2_stats['Statistic'],
        'R2_Score': r2_stats['R2_Score'],
        'RMSE': rmse_stats['RMSE'],
        'MAE': mae_stats['MAE']
    })
    combined_stats.to_csv(f"{save_dir}/{prefix}combined_performance_statistics_iter{iteration}.csv", index=False)
    
    # 5. Feature Importance Statistics
    if all_feature_importances:
        fi_stats = calculate_feature_importance_statistics(all_feature_importances, feature_cols)
        fi_stats.to_csv(f"{save_dir}/{prefix}feature_importance_statistics_iter{iteration}.csv", index=False)
        
        # 6. Feature Importance Rankings and Tiers
        fi_summary = pd.DataFrame({
            'Rank': fi_stats['Rank_by_Mean'],
            'Feature': fi_stats['Feature'],
            'Mean_Importance': fi_stats['Mean_Importance'],
            'Std_Importance': fi_stats['Std_Importance'],
            'Coefficient_of_Variation': fi_stats['CV_Importance'],
            'Tier': pd.cut(fi_stats['Mean_Importance'], 
                          bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        })
        fi_summary.to_csv(f"{save_dir}/{prefix}feature_importance_rankings_iter{iteration}.csv", index=False)
        
        # 7. Top and Bottom Features Analysis
        top_features = fi_stats.head(10)[['Feature', 'Mean_Importance', 'Std_Importance', 'CV_Importance']]
        bottom_features = fi_stats.tail(10)[['Feature', 'Mean_Importance', 'Std_Importance', 'CV_Importance']]
        
        top_features.to_csv(f"{save_dir}/{prefix}top_features_iter{iteration}.csv", index=False)
        bottom_features.to_csv(f"{save_dir}/{prefix}bottom_features_iter{iteration}.csv", index=False)
        
        return combined_stats, fi_stats, fi_summary
    
    return combined_stats, None, None

def print_enhanced_report(iteration, r2_scores, rmse_scores, mae_scores, 
                         all_feature_importances, feature_cols, study_name=""):
    """Print comprehensive statistical report with study identification"""
    study_header = f" - {study_name.upper()}" if study_name else ""
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE STATISTICAL REPORT - ITERATION {iteration}{study_header}")
    print(f"{'='*100}")
    
    # Performance Metrics Analysis
    metrics_data = {
        'R² Score': np.array(r2_scores),
        'RMSE': np.array(rmse_scores),
        'MAE': np.array(mae_scores)
    }
    
    for metric_name, data in metrics_data.items():
        print(f"\n{metric_name.upper()} ANALYSIS ({len(data)} iterations):")
        print(f"  Count:                   {len(data)}")
        print(f"  Mean:                    {np.mean(data):.6f}")
        print(f"  Standard Deviation:      {np.std(data, ddof=1):.6f}")
        print(f"  Minimum:                 {np.min(data):.6f}")
        print(f"  Q1 (25th percentile):    {np.percentile(data, 25):.6f}")
        print(f"  Median (50th percentile): {np.percentile(data, 50):.6f}")
        print(f"  Q3 (75th percentile):    {np.percentile(data, 75):.6f}")
        print(f"  Maximum:                 {np.max(data):.6f}")
        print(f"  IQR:                     {np.percentile(data, 75) - np.percentile(data, 25):.6f}")
        print(f"  Skewness:                {skew(data):.6f}")
        print(f"  Kurtosis:                {kurtosis(data):.6f}")
        print(f"  Coefficient of Variation: {(np.std(data, ddof=1) / np.mean(data)) * 100:.2f}%")
    
    # Feature Importance Analysis
    if all_feature_importances:
        fi_stats = calculate_feature_importance_statistics(all_feature_importances, feature_cols)
        
        print(f"\n{'='*50}")
        print("FEATURE IMPORTANCE COMPREHENSIVE ANALYSIS")
        print(f"{'='*50}")
        
        print(f"\nTOP 10 FEATURES BY MEAN IMPORTANCE:")
        print(f"{'Rank':<4} {'Feature':<30} {'Mean':<10} {'Std':<10} {'CV%':<8}")
        print("-" * 70)
        for i, (_, row) in enumerate(fi_stats.head(10).iterrows()):
            print(f"{i+1:<4} {row['Feature']:<30} {row['Mean_Importance']:<10.6f} "
                  f"{row['Std_Importance']:<10.6f} {row['CV_Importance']:<8.2f}")
        
        print(f"\nBOTTOM 10 FEATURES BY MEAN IMPORTANCE:")
        print(f"{'Rank':<4} {'Feature':<30} {'Mean':<10} {'Std':<10} {'CV%':<8}")
        print("-" * 70)
        bottom_10 = fi_stats.tail(10)
        for i, (_, row) in enumerate(bottom_10.iterrows()):
            rank = len(fi_stats) - len(bottom_10) + i + 1
            print(f"{rank:<4} {row['Feature']:<30} {row['Mean_Importance']:<10.6f} "
                  f"{row['Std_Importance']:<10.6f} {row['CV_Importance']:<8.2f}")
        
        # Feature Importance Distribution Analysis
        mean_importances = fi_stats['Mean_Importance'].values
        print(f"\nFEATURE IMPORTANCE DISTRIBUTION ANALYSIS:")
        print(f"  Total Features:          {len(mean_importances)}")
        print(f"  Mean Importance:         {np.mean(mean_importances):.6f}")
        print(f"  Std Dev Importance:      {np.std(mean_importances, ddof=1):.6f}")
        print(f"  Min Importance:          {np.min(mean_importances):.6f}")
        print(f"  Max Importance:          {np.max(mean_importances):.6f}")
        print(f"  Importance Range:        {np.max(mean_importances) - np.min(mean_importances):.6f}")
        
        # Tier Analysis
        high_importance = fi_stats[fi_stats['Mean_Importance'] > np.percentile(mean_importances, 75)]
        medium_importance = fi_stats[(fi_stats['Mean_Importance'] > np.percentile(mean_importances, 25)) & 
                                   (fi_stats['Mean_Importance'] <= np.percentile(mean_importances, 75))]
        low_importance = fi_stats[fi_stats['Mean_Importance'] <= np.percentile(mean_importances, 25)]
        
        print(f"\nFEATURE IMPORTANCE TIERS:")
        print(f"  High Importance (>75th percentile): {len(high_importance)} features")
        print(f"  Medium Importance (25-75th percentile): {len(medium_importance)} features")
        print(f"  Low Importance (≤25th percentile): {len(low_importance)} features")

def run_single_experiment(data_path, study_name, n_iter=1000, random_seed=42):
    """Run a single experiment for one dataset"""
    print(f"\n{'='*120}")
    print(f"STARTING STUDY: {study_name.upper()}")
    print(f"Data Path: {data_path}")
    print(f"{'='*120}")
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    target_col = 'Funding Round Duration (days)'

    # Select exactly 15 feature columns excluding target and common ID-like fields
    candidate_feature_cols = [col for col in df.columns if col != target_col]
    id_like_columns = {'id', 'ID', 'loan_id', 'Loan ID', 'index', 'Microlender_ID', 'Microlender ID'}
    candidate_feature_cols = [c for c in candidate_feature_cols if c not in id_like_columns]
    if len(candidate_feature_cols) < 15:
        raise ValueError(f"Expected at least 15 feature columns (excluding target), found {len(candidate_feature_cols)}. Columns: {candidate_feature_cols}")
    feature_cols = candidate_feature_cols[:15]

    X = df[feature_cols]
    y = df[target_col]

    # Create study-specific directory (save under 'study1')
    study_dir = "../results/random_forest_run"
    os.makedirs(study_dir, exist_ok=True)

    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rng = np.random.default_rng(random_seed)
    all_metrics = []
    r2_scores, rmse_scores, mae_scores = [], [], []
    all_feature_importances = []
    
    print(f"\nStarting {n_iter} iterations for {study_name}...")
    print(f"{'='*80}")
    
    for i in tqdm(range(n_iter), desc=f"Running {study_name} iterations"):
        rs = rng.integers(0, 1_000_000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
        rf = RandomForestRegressor(random_state=rs)
        search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20,
                                    scoring='neg_mean_squared_error', cv=3, random_state=rs, n_jobs=-1)
        search.fit(X_train, y_train)
        best_rf = search.best_estimator_

        y_pred_test = best_rf.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        r2_scores.append(test_r2)
        rmse_scores.append(test_rmse)
        mae_scores.append(test_mae)
        all_feature_importances.append(best_rf.feature_importances_)

        tqdm.write(f"{study_name} - Iter {i+1:4d}: R² = {test_r2:.6f} | RMSE = {test_rmse:.4f} | MAE = {test_mae:.4f}")

        # Generate comprehensive tables at intervals
        if (i + 1) % 100 == 0:
            print_enhanced_report(i + 1, r2_scores, rmse_scores, mae_scores,
                                all_feature_importances, feature_cols, study_name)
            create_comprehensive_tables(i + 1, r2_scores, rmse_scores, mae_scores,
                                      all_feature_importances, feature_cols, study_dir, study_name)

        all_metrics.append({
            'iteration': i + 1, 
            'test_r2': test_r2, 
            'test_rmse': test_rmse, 
            'test_mae': test_mae,
            'study': study_name
        })

    # Final comprehensive analysis
    print(f"\n{'='*120}")
    print(f"FINAL COMPREHENSIVE ANALYSIS - {study_name.upper()}")
    print(f"{'='*120}")
    
    print_enhanced_report("FINAL", r2_scores, rmse_scores, mae_scores,
                         all_feature_importances, feature_cols, study_name)
    
    # Generate final comprehensive tables
    combined_stats, fi_stats, fi_summary = create_comprehensive_tables(
        "FINAL", r2_scores, rmse_scores, mae_scores,
        all_feature_importances, feature_cols, study_dir, study_name
    )
    # Overall visualizations: Feature Importance and R² distribution
    try:
        if fi_stats is not None:
            ordered = fi_stats.sort_values('Mean_Importance', ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Mean_Importance', y='Feature', data=ordered, palette='viridis')
            plt.title(f"Overall Feature Importance - {study_name}")
            plt.xlabel("Mean Importance (across iterations)")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(study_dir, f"{study_name}_feature_importance_overall.png"), dpi=300)
            plt.close()

        if len(r2_scores) > 0:
            plt.figure(figsize=(8, 5))
            sns.histplot(r2_scores, bins=30, kde=True, color="#2a9d8f")
            plt.title(f"R² Distribution - {study_name}")
            plt.xlabel("R² Score")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(study_dir, f"{study_name}_r2_distribution.png"), dpi=300)
            plt.close()
    except Exception as viz_err:
        print(f"Warning: Failed to create visualizations for {study_name}: {viz_err}")
    
    # Save detailed results for this study
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{study_dir}/{study_name}_detailed_experiment_results.csv", index=False)
    
    # Save iteration-by-iteration data for further analysis
    iterations_summary = pd.DataFrame({
        'iteration': range(1, n_iter + 1),
        'r2_score': r2_scores,
        'rmse': rmse_scores,
        'mae': mae_scores,
        'study': [study_name] * n_iter
    })
    iterations_summary.to_csv(f"{study_dir}/{study_name}_iterations_summary.csv", index=False)
    
    # Calculate and save correlation matrix between performance metrics
    correlation_matrix = iterations_summary[['r2_score', 'rmse', 'mae']].corr()
    correlation_matrix.to_csv(f"{study_dir}/{study_name}_performance_metrics_correlation.csv")
    
    print(f"\n{study_name} study completed!")
    print(f"All tables saved to: {study_dir}/")
    
    return metrics_df, combined_stats, fi_stats

def create_comparative_analysis(groundtruth_metrics, llama_metrics):
    """Create comparative analysis between the two studies"""
    print(f"\n{'='*120}")
    print("COMPARATIVE ANALYSIS: GROUND TRUTH vs PREDICTION LLAMA")
    print(f"{'='*120}")
    
    # Combine datasets for comparison
    combined_metrics = pd.concat([groundtruth_metrics, llama_metrics], ignore_index=True)
    
    # Create comparative statistics
    comparison_stats = []
    metrics = ['test_r2', 'test_rmse', 'test_mae']
    
    for metric in metrics:
        gt_data = groundtruth_metrics[metric]
        llama_data = llama_metrics[metric]
        
        comparison = {
            'Metric': metric.upper(),
            'GroundTruth_Mean': np.mean(gt_data),
            'GroundTruth_Std': np.std(gt_data, ddof=1),
            'PredictionLlama_Mean': np.mean(llama_data),
            'PredictionLlama_Std': np.std(llama_data, ddof=1),
            'Mean_Difference': np.mean(gt_data) - np.mean(llama_data),
            'Std_Difference': np.std(gt_data, ddof=1) - np.std(llama_data, ddof=1)
        }
        comparison_stats.append(comparison)
        
        print(f"\n{metric.upper()} COMPARISON:")
        print(f"  Ground Truth    - Mean: {np.mean(gt_data):.6f}, Std: {np.std(gt_data, ddof=1):.6f}")
        print(f"  Prediction Llama - Mean: {np.mean(llama_data):.6f}, Std: {np.std(llama_data, ddof=1):.6f}")
        print(f"  Difference (GT - Llama) - Mean: {comparison['Mean_Difference']:.6f}, Std: {comparison['Std_Difference']:.6f}")
    
    # Save comparative analysis
    comparison_df = pd.DataFrame(comparison_stats)
    os.makedirs("../results/random_forest_run", exist_ok=True)
    comparison_df.to_csv("../results/random_forest_run/comparative_analysis.csv", index=False)
    
    # Save combined detailed results
    combined_metrics.to_csv("../results/random_forest_run/combined_detailed_results.csv", index=False)
    
    print(f"\nComparative analysis saved to: study1/comparative_analysis.csv")
    print(f"Combined results saved to: study1/combined_detailed_results.csv")
    
    return comparison_df, combined_metrics

def run_dual_study_experiment(n_iter=10, random_seed=42):
    """Run experiments on both datasets and create comparative analysis"""
    
    # Define the two studies
    studies = {
        "GroundTruth": "../data/random_forest/Result - GroundTruth_numeric.csv",
        "PredictionLlama": "../data/random_forest/Result - predictionLLama_numeric.csv"
    }
    
    results = {}
    
    # Run experiments for both studies
    for study_name, data_path in studies.items():
        print(f"\n🚀 Starting {study_name} Study...")
        
        try:
            metrics_df, combined_stats, fi_stats = run_single_experiment(
                data_path, study_name, n_iter, random_seed
            )
            results[study_name] = {
                'metrics': metrics_df,
                'combined_stats': combined_stats,
                'feature_importance': fi_stats
            }
            print(f"✅ {study_name} study completed successfully!")
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find file {data_path}")
            print(f"Please ensure the file exists and the path is correct.")
            continue
        except Exception as e:
            print(f"❌ Error in {study_name} study: {str(e)}")
            continue
    
    # Create comparative analysis if both studies completed
    if len(results) == 2:
        print(f"\n🔍 Creating comparative analysis...")
        try:
            comparison_df, combined_metrics = create_comparative_analysis(
                results['GroundTruth']['metrics'],
                results['PredictionLlama']['metrics']
            )
            print(f"✅ Comparative analysis completed!")
            
            return results, comparison_df, combined_metrics
            
        except Exception as e:
            print(f"❌ Error in comparative analysis: {str(e)}")
            return results, None, None
    else:
        print(f"⚠️  Comparative analysis skipped - not all studies completed successfully")
        return results, None, None

if __name__ == "__main__":
    print("🎯 DUAL STUDY ML EXPERIMENT")
    print("=" * 120)
    print("This script will run comprehensive machine learning experiments on:")
    print("1. Ground Truth dataset (../data/random_forest/Result - GroundTruth_numeric.csv)")
    print("2. Prediction Llama dataset (../data/random_forest/Result - predictionLLama_numeric.csv)")
    print("=" * 120)
    
    # Run the dual study experiment
    results, comparison_df, combined_metrics = run_dual_study_experiment(n_iter=1000, random_seed=42)
    
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print("=" * 120)
    print("📁 Check the following directories for results:")
    print("   • study1/ - All CSV tables and visualizations for both studies")
    print("=" * 120)