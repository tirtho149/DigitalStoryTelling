import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# -------------------------------
# 0. Setup output directory
# -------------------------------
output_dir = '../results/country_ranking_run'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# -------------------------------
# 1. Load your actual data
# -------------------------------
csv_file_path = '../data/country_ranking_survey/study2.csv'

try:
    df_raw = pd.read_csv(csv_file_path)
    print("Data loaded successfully from:", csv_file_path)
except FileNotFoundError:
    print(f"CSV file not found at: {csv_file_path}")
    print("Please check the file path and ensure the file exists.")
    raise

# Save initial data overview
data_overview_path = os.path.join(output_dir, 'data_overview.txt')
with open(data_overview_path, 'w') as f:
    f.write("DATA OVERVIEW\n")
    f.write("="*50 + "\n\n")
    f.write(f"Data shape: {df_raw.shape}\n\n")
    f.write("First 5 rows:\n")
    f.write(df_raw.head().to_string())
    f.write(f"\n\nCountries in dataset: {list(df_raw['Country'].unique())}\n\n")
    f.write("Number of students per country:\n")
    f.write(df_raw['Country'].value_counts().to_string())
    f.write("\n")

print(f"Data overview saved to: {data_overview_path}")

# -------------------------------
# 2. Reshape data from wide to long format for analysis
# -------------------------------
# Extract feature columns (exclude Student_ID and Country)
feature_columns = [col for col in df_raw.columns if col not in ['Student_ID', 'Country']]

# Reshape to long format
df_long = df_raw.melt(
    id_vars=['Student_ID', 'Country'], 
    value_vars=feature_columns,
    var_name='Feature', 
    value_name='Rank'
)

# -------------------------------
# 3. Normalize rank to importance (1 = most important, 0 = least important)
# -------------------------------
# Assuming rankings are 1-5 where 1 = most important
max_rank = df_long['Rank'].max()
min_rank = df_long['Rank'].min()

df_long['Importance'] = 1 - (df_long['Rank'] - min_rank) / (max_rank - min_rank)

# -------------------------------
# 4. Calculate average importance per Country × Feature
# -------------------------------
avg_country_feature = df_long.groupby(['Country', 'Feature'])['Importance'].mean().reset_index()
avg_pivot = avg_country_feature.pivot(index='Country', columns='Feature', values='Importance').round(3)

# Calculate overall feature importance (averaged across countries)
overall_feature_importance = df_long.groupby('Feature')['Importance'].mean().sort_values(ascending=False)

# Save feature analysis
feature_analysis_path = os.path.join(output_dir, 'feature_analysis.txt')
with open(feature_analysis_path, 'w') as f:
    f.write("FEATURE ANALYSIS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Feature columns identified: {feature_columns}\n\n")
    f.write(f"Long format data shape: {df_long.shape}\n\n")
    f.write(f"Rank range: {min_rank} to {max_rank}\n\n")
    f.write("Overall Feature Importance (highest to lowest):\n")
    for feat, imp in overall_feature_importance.items():
        f.write(f"{feat}: {imp:.3f}\n")

print(f"Feature analysis saved to: {feature_analysis_path}")

# -------------------------------
# 5. OLS Regression with interactions
# -------------------------------
# Ensure categorical variables are properly encoded
countries = df_long['Country'].unique()
features = df_long['Feature'].unique()

df_long['Country'] = pd.Categorical(df_long['Country'], categories=countries, ordered=True)
df_long['Feature'] = pd.Categorical(df_long['Feature'], categories=features, ordered=True)

# Fit OLS model
ols_model = smf.ols('Importance ~ Country + Feature + Country:Feature', data=df_long).fit()
df_long['Importance_OLS_pred'] = ols_model.predict(df_long)

# Get OLS predictions for each country-feature combination
avg_ols_pred = df_long.groupby(['Country', 'Feature'])['Importance_OLS_pred'].mean().reset_index()
ols_pivot = avg_ols_pred.pivot(index='Country', columns='Feature', values='Importance_OLS_pred').round(3)

# Save OLS model results
ols_results_path = os.path.join(output_dir, 'ols_model_results.txt')
with open(ols_results_path, 'w') as f:
    f.write("OLS MODEL RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"R-squared: {ols_model.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {ols_model.rsquared_adj:.4f}\n")
    f.write(f"F-statistic: {ols_model.fvalue:.4f} (p = {ols_model.f_pvalue:.6f})\n\n")
    f.write("Full Model Summary:\n")
    f.write(str(ols_model.summary()))

print(f"OLS model results saved to: {ols_results_path}")

# -------------------------------
# 6. Fixed Ordered Logistic Regression
# -------------------------------
# Reset index to ensure proper alignment
df_long_olr = df_long.reset_index(drop=True).copy()

# Create  variables for OLR - with proper reference categories
df_dummies = pd.get_dummies(df_long_olr[['Country', 'Feature']], 
                           columns=['Country', 'Feature'], 
                           drop_first=True)

# Add interaction terms manually if needed (optional - can be computationally expensive)
country_dummies = [col for col in df_dummies.columns if col.startswith('Country_')]
feature_dummies = [col for col in df_dummies.columns if col.startswith('Feature_')]

# Create a subset of key interactions to avoid overfitting
interaction_terms = pd.DataFrame()
if len(country_dummies) > 0 and len(feature_dummies) > 0:
    # Only create interactions for the first few most important features/countries
    n_interactions = min(3, len(country_dummies), len(feature_dummies))
    for i in range(n_interactions):
        for j in range(n_interactions):
            if i < len(country_dummies) and j < len(feature_dummies):
                interaction_name = f"Int_{country_dummies[i]}_{feature_dummies[j]}"
                interaction_terms[interaction_name] = (df_dummies[country_dummies[i]] * 
                                                     df_dummies[feature_dummies[j]])

# Combine all predictors
if not interaction_terms.empty:
    X_olr = pd.concat([df_dummies, interaction_terms], axis=1)
else:
    X_olr = df_dummies

print(f"OLR predictor matrix shape: {X_olr.shape}")

try:
    # Fit OLR model with more robust settings
    olr_model = OrderedModel(
        df_long_olr['Rank'].astype(int),
        X_olr,
        distr='logit'
    )
    
    # Try different optimization methods
    olr_fitted = None
    for method in ['bfgs', 'newton', 'lbfgs']:
        try:
            print(f"Trying OLR optimization method: {method}")
            olr_fitted = olr_model.fit(method=method, disp=False, maxiter=1000)
            print(f"OLR converged successfully with method: {method}")
            break
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")
            continue
    
    if olr_fitted is not None:
        # Get predictions - fixed alignment issue
        pred_probs = olr_fitted.predict(X_olr)
        
        # Calculate expected ranks
        rank_values = np.arange(min_rank, max_rank + 1)
        if pred_probs.shape[1] != len(rank_values):
            print(f"Warning: Prediction shape mismatch. Got {pred_probs.shape[1]} categories, expected {len(rank_values)}")
            # Adjust rank_values to match prediction dimensions
            rank_values = np.arange(1, pred_probs.shape[1] + 1)
        
        expected_rank = (pred_probs * rank_values).sum(axis=1)
        
        # Convert back to importance score
        df_long_olr['Importance_OLR'] = 1 - (expected_rank - rank_values.min()) / (rank_values.max() - rank_values.min())
        
        # Merge back to main dataframe
        df_long['Importance_OLR'] = df_long_olr['Importance_OLR']
        
        # Get OLR predictions for each country-feature combination
        avg_olr = df_long.groupby(['Country', 'Feature'])['Importance_OLR'].mean().reset_index()
        olr_pivot = avg_olr.pivot(index='Country', columns='Feature', values='Importance_OLR').round(3)
        
        # Save OLR model results
        olr_results_path = os.path.join(output_dir, 'olr_model_results.txt')
        with open(olr_results_path, 'w') as f:
            f.write("ORDERED LOGISTIC REGRESSION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Pseudo R-squared: {olr_fitted.prsquared:.4f}\n")
            f.write(f"Log-likelihood: {olr_fitted.llf:.4f}\n")
            f.write(f"AIC: {olr_fitted.aic:.4f}\n")
            f.write(f"BIC: {olr_fitted.bic:.4f}\n\n")
            f.write("Model Convergence: Successful\n")
            f.write(f"Number of observations: {olr_fitted.nobs}\n")
            f.write(f"Number of parameters: {len(olr_fitted.params)}\n\n")
            f.write("Full Model Summary:\n")
            f.write(str(olr_fitted.summary()))
        
        print(f"OLR model results saved to: {olr_results_path}")
        olr_success = True
        
    else:
        raise Exception("All optimization methods failed")
        
except Exception as e:
    error_log_path = os.path.join(output_dir, 'olr_error_log.txt')
    with open(error_log_path, 'w') as f:
        f.write("ORDERED LOGISTIC REGRESSION ERROR LOG\n")
        f.write("="*50 + "\n\n")
        f.write(f"OLR model failed with error: {str(e)}\n")
        f.write("Proceeding with OLS results only...\n")
        f.write(f"\nDebugging information:\n")
        f.write(f"df_long shape: {df_long.shape}\n")
        f.write(f"X_olr shape: {X_olr.shape}\n")
        f.write(f"Rank values: {df_long['Rank'].unique()}\n")
        f.write(f"Rank value counts: {df_long['Rank'].value_counts().to_dict()}\n")
    
    print(f"OLR error logged to: {error_log_path}")
    
    # Use OLS predictions as fallback
    df_long['Importance_OLR'] = df_long['Importance_OLS_pred']
    olr_pivot = ols_pivot.copy()
    olr_success = False

# -------------------------------
# 7. Enhanced Statistical significance testing
# -------------------------------
def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '.'
    else:
        return ''

# Extract p-values from OLS model
ols_pvalues = ols_model.pvalues
significance_matrix = pd.DataFrame(index=countries, columns=features, dtype=str)

# Fill significance matrix based on OLS results
reference_country = countries[0]
reference_feature = features[0]

for country in countries:
    for feature in features:
        if country == reference_country and feature == reference_feature:
            # Reference category - test intercept
            significance_matrix.loc[country, feature] = get_significance_stars(ols_pvalues.get('Intercept', 1.0))
        else:
            p_val = None
            
            # Check interaction terms first (most specific)
            if country != reference_country and feature != reference_feature:
                interaction_param = f'Country[T.{country}]:Feature[T.{feature}]'
                if interaction_param in ols_pvalues:
                    p_val = ols_pvalues[interaction_param]
            
            # If no interaction, check main effects
            if p_val is None:
                main_effects_p = []
                if country != reference_country:
                    country_param = f'Country[T.{country}]'
                    if country_param in ols_pvalues:
                        main_effects_p.append(ols_pvalues[country_param])
                
                if feature != reference_feature:
                    feature_param = f'Feature[T.{feature}]'
                    if feature_param in ols_pvalues:
                        main_effects_p.append(ols_pvalues[feature_param])
                
                # Use the minimum p-value (most significant)
                if main_effects_p:
                    p_val = min(main_effects_p)
            
            # Default to intercept if no specific parameter found
            if p_val is None:
                p_val = ols_pvalues.get('Intercept', 1.0)
            
            significance_matrix.loc[country, feature] = get_significance_stars(p_val)

# -------------------------------
# 8. Enhanced visualization
# -------------------------------
# -------------------------------
# 8. Enhanced visualization with Times New Roman and Mathematical Formulas
# -------------------------------

# Set Times New Roman font globally
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

fig = plt.figure(figsize=(32, 14))

# Common colormap settings
vmin = min(avg_pivot.min().min(), ols_pivot.min().min(), olr_pivot.min().min()) - 0.01
vmax = max(avg_pivot.max().max(), ols_pivot.max().max(), olr_pivot.max().max()) + 0.01

# Truncate feature names for better display
feature_labels = [feat[:25] + "..." if len(feat) > 25 else feat for feat in features]
avg_pivot_display = avg_pivot.copy()
avg_pivot_display.columns = feature_labels
ols_pivot_display = ols_pivot.copy()
ols_pivot_display.columns = feature_labels
olr_pivot_display = olr_pivot.copy()
olr_pivot_display.columns = feature_labels

# Create subplots with more space for formulas
gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.4, wspace=0.3)

# Heatmap 1: Average Importance
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(avg_pivot_display, annot=True, fmt=".3f", cmap="YlGnBu", 
            vmin=vmin, vmax=vmax, cbar_kws={'label': 'Normalized Importance'},
            square=True, linewidths=0.5, ax=ax1,
            annot_kws={'fontweight': 'bold', 'fontsize': 10})
ax1.set_title("Average Normalized Importance\n(Raw Rankings)", 
              fontsize=16, fontweight='bold', pad=20, family='Times New Roman')
ax1.set_ylabel("Country", fontsize=14, fontweight='bold', family='Times New Roman')
ax1.set_xlabel("Feature", fontsize=14, fontweight='bold', family='Times New Roman')
ax1.tick_params(axis='x', rotation=45, labelsize=11, which='major')
ax1.tick_params(axis='y', rotation=0, labelsize=11, which='major')

# Add formula below heatmap 1
ax1_formula = fig.add_subplot(gs[1, 0])
ax1_formula.text(0.5, 0.7, r'$I_{cf} = \frac{1}{n} \sum_{i=1}^{n} \left(1 - \frac{R_{cfi} - R_{min}}{R_{max} - R_{min}}\right)$', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, fontweight='bold', family='Times New Roman',
                 transform=ax1_formula.transAxes)
ax1_formula.text(0.5, 0.3, 'where $I_{cf}$ = Importance for country c, feature f\n$R_{cfi}$ = Rank for individual i\n$n$ = Sample size per country', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=11, fontweight='normal', family='Times New Roman',
                 transform=ax1_formula.transAxes)
ax1_formula.axis('off')

# Heatmap 2: OLS Predicted
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(ols_pivot_display, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=vmin, vmax=vmax, cbar_kws={'label': 'Predicted Importance'},
            square=True, linewidths=0.5, ax=ax2,
            annot_kws={'fontweight': 'bold', 'fontsize': 10})
ax2.set_title("OLS Predicted Importance\n(Country + Feature + Interactions)", 
              fontsize=16, fontweight='bold', pad=20, family='Times New Roman')
ax2.set_ylabel("Country", fontsize=14, fontweight='bold', family='Times New Roman')
ax2.set_xlabel("Feature", fontsize=14, fontweight='bold', family='Times New Roman')
ax2.tick_params(axis='x', rotation=45, labelsize=11, which='major')
ax2.tick_params(axis='y', rotation=0, labelsize=11, which='major')

# Add formula below heatmap 2
ax2_formula = fig.add_subplot(gs[1, 1])
ax2_formula.text(0.5, 0.7, r'$\hat{I}_{cf} = \beta_0 + \sum_{c} \beta_c D_c + \sum_{f} \beta_f D_f + \sum_{c,f} \beta_{cf} D_c D_f + \epsilon$', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, fontweight='bold', family='Times New Roman',
                 transform=ax2_formula.transAxes)
ax2_formula.text(0.5, 0.3, r'where $D_c$ = Country , $D_f$ = Feature ' + '\n' + r'$\beta_{cf}$ = Interaction coefficients', 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=11, fontweight='normal', family='Times New Roman',
                 transform=ax2_formula.transAxes)
ax2_formula.axis('off')

# Heatmap 3: OLR Predicted
ax3 = fig.add_subplot(gs[0, 2])
if olr_success:
    title_suffix = "\n(Ordered Logistic Regression)"
    cmap = "PuBuGn"
    formula_text = r'$P(R \leq j|X) = \frac{\exp(\alpha_j - X\beta)}{1 + \exp(\alpha_j - X\beta)}$'
    formula_desc = r'where $\alpha_j$ = Cutpoint for rank j' + '\n' + r'$X$ = Predictor matrix, $\beta$ = Coefficients'
else:
    title_suffix = "\n(OLS - OLR failed)"
    cmap = "Greys"
    formula_text = "OLR Model Failed to Converge"
    formula_desc = "Using OLS predictions as fallback"
    
sns.heatmap(olr_pivot_display, annot=True, fmt=".3f", cmap=cmap,
            vmin=vmin, vmax=vmax, cbar_kws={'label': 'Predicted Importance'},
            square=True, linewidths=0.5, ax=ax3,
            annot_kws={'fontweight': 'bold', 'fontsize': 10})
ax3.set_title(f"OLR Predicted Importance{title_suffix}", 
              fontsize=16, fontweight='bold', pad=20, family='Times New Roman')
ax3.set_ylabel("Country", fontsize=14, fontweight='bold', family='Times New Roman')
ax3.set_xlabel("Feature", fontsize=14, fontweight='bold', family='Times New Roman')
ax3.tick_params(axis='x', rotation=45, labelsize=11, which='major')
ax3.tick_params(axis='y', rotation=0, labelsize=11, which='major')

# Add formula below heatmap 3
ax3_formula = fig.add_subplot(gs[1, 2])
if olr_success:
    ax3_formula.text(0.5, 0.7, formula_text, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14, fontweight='bold', family='Times New Roman',
                     transform=ax3_formula.transAxes)
    ax3_formula.text(0.5, 0.3, formula_desc, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=11, fontweight='normal', family='Times New Roman',
                     transform=ax3_formula.transAxes)
else:
    ax3_formula.text(0.5, 0.5, formula_text + '\n' + formula_desc, 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, fontweight='bold', family='Times New Roman',
                     transform=ax3_formula.transAxes, color='red')
ax3_formula.axis('off')

# Ensure all text uses Times New Roman and is bold where appropriate
for ax in [ax1, ax2, ax3]:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')
    
    # Make colorbar labels bold
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=10)
        for text in cbar.ax.get_yticklabels():
            text.set_fontfamily('Times New Roman')
            text.set_fontweight('bold')
        cbar.set_label(cbar.ax.get_ylabel(), fontfamily='Times New Roman', 
                      fontweight='bold', fontsize=12)

plt.show()

# Save the enhanced figure with high DPI
fig_path = os.path.join(output_dir, 'enhanced_feature_importance_heatmaps.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nEnhanced heatmaps with formulas saved as: {fig_path}")

# Reset matplotlib settings to default after visualization
plt.rcdefaults()

# -------------------------------
# Alternative version with even more detailed formulas
# -------------------------------

# Create a second version with more detailed mathematical explanations
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

fig2 = plt.figure(figsize=(36, 16))
gs2 = fig2.add_gridspec(3, 3, height_ratios=[4, 1.5, 0.5], hspace=0.5, wspace=0.3)

# Recreate heatmaps with even more detailed formatting
for i, (data, title, cmap_name, ax_pos) in enumerate([
    (avg_pivot_display, "Average Normalized Importance\n(Raw Rankings)", "YlGnBu", 0),
    (ols_pivot_display, "OLS Predicted Importance\n(Country + Feature + Interactions)", "YlOrRd", 1),
    (olr_pivot_display, f"OLR Predicted Importance{title_suffix}", cmap if olr_success else "Greys", 2)
]):
    
    ax = fig2.add_subplot(gs2[0, ax_pos])
    sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap_name,
                vmin=vmin, vmax=vmax, 
                cbar_kws={'label': 'Normalized Importance Score'},
                square=True, linewidths=0.5, ax=ax,
                annot_kws={'fontweight': 'bold', 'fontsize': 9, 'fontfamily': 'Times New Roman'})
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25, family='Times New Roman')
    ax.set_ylabel("Country", fontsize=16, fontweight='bold', family='Times New Roman')
    ax.set_xlabel("Feature", fontsize=16, fontweight='bold', family='Times New Roman')
    ax.tick_params(axis='x', rotation=45, labelsize=12, which='major')
    ax.tick_params(axis='y', rotation=0, labelsize=12, which='major')
    
    # Bold tick labels
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('Times New Roman')
        text.set_fontweight('bold')

# Detailed formula explanations
formulas = [
    {
        'main': r'$I_{cf} = \frac{1}{n_c} \sum_{i=1}^{n_c} \left(1 - \frac{R_{cfi} - 1}{4}\right)$',
        'detail': r'Normalization: $I_{cf} \in [0,1]$ where 1 = most important' + '\n' + 
                 r'$n_c$ = number of students in country $c$' + '\n' +
                 r'$R_{cfi}$ = rank given by student $i$ in country $c$ for feature $f$'
    },
    {
        'main': r'$\hat{I}_{cf} = \beta_0 + \sum_{j=2}^{C} \beta_j \mathbf{1}_{c=j} + \sum_{k=2}^{F} \gamma_k \mathbf{1}_{f=k} + \sum_{j,k} \delta_{jk} \mathbf{1}_{c=j} \mathbf{1}_{f=k}$',
        'detail': r'Linear model with main effects and interactions' + '\n' + 
                 r'$\mathbf{1}_{c=j}$ = indicator function for country $j$' + '\n' +
                 r'$\delta_{jk}$ = interaction coefficient between country $j$ and feature $k$'
    },
    {
        'main': r'$P(R_{cfi} \leq j | X_{cf}) = \frac{\exp(\alpha_j - X_{cf}^T\beta)}{1 + \exp(\alpha_j - X_{cf}^T\beta)}$' if olr_success else 'Model Failed',
        'detail': r'Cumulative logit model for ordinal rankings' + '\n' + 
                 r'$\alpha_j$ = cutpoint parameters, $X_{cf}$ = design matrix' + '\n' +
                 r'Expected rank: $E[R|X] = \sum_{j=1}^{5} j \cdot P(R = j|X)$' if olr_success else 'Using OLS fallback'
    }
]

for i, formula_info in enumerate(formulas):
    ax_formula = fig2.add_subplot(gs2[1, i])
    
    # Main formula (larger, bold)
    ax_formula.text(0.5, 0.75, formula_info['main'], 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=16, fontweight='bold', family='Times New Roman',
                   transform=ax_formula.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Detailed explanation (smaller, normal weight)
    ax_formula.text(0.5, 0.25, formula_info['detail'], 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, fontweight='normal', family='Times New Roman',
                   transform=ax_formula.transAxes)
    
    ax_formula.axis('off')

# Add overall title
fig2.suptitle('Feature Importance Analysis: Cross-Country Comparison of Student Preferences', 
              fontsize=22, fontweight='bold', family='Times New Roman', y=0.95)

# Add methodology note at bottom
ax_note = fig2.add_subplot(gs2[2, :])
methodology_note = (
    "Methodology: Rankings (1-5) transformed to importance scores (0-1). "
    "OLS model includes country and feature main effects plus interactions. "
    f"OLR model {'successfully fitted' if olr_success else 'failed to converge'}. "
    f"Sample: {len(countries)} countries, {len(features)} features, "
    f"{len(df_raw)} total students."
)

ax_note.text(0.5, 0.5, methodology_note,
            horizontalalignment='center', verticalalignment='center',
            fontsize=11, fontweight='normal', family='Times New Roman',
            transform=ax_note.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax_note.axis('off')

plt.show()

# Save the detailed version
fig2_path = os.path.join(output_dir, 'detailed_feature_importance_with_formulas.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Detailed heatmaps with comprehensive formulas saved as: {fig2_path}")

# Reset to defaults
plt.rcdefaults()
# -------------------------------
# 9. Generate LaTeX tables with better formatting
# -------------------------------
def df_to_latex_with_significance(df, sig_matrix=None, caption="", label="", feature_short_names=None):
    """Convert DataFrame to LaTeX table with optional significance stars"""
    latex = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    
    # Use short names for columns if provided
    display_df = df.copy()
    if feature_short_names:
        display_df.columns = feature_short_names[:len(display_df.columns)]
    
    latex += "\\begin{tabular}{l" + "c" * len(display_df.columns) + "}\n"
    latex += "\\toprule\n"
    
    # Header with line breaks for long names
    header_cols = []
    for col in display_df.columns:
        col_str = str(col)
        if len(col_str) > 12:
            # Split long column names at spaces or underscores
            if ' ' in col_str:
                words = col_str.split(' ')
            elif '_' in col_str:
                words = col_str.split('_')
            else:
                words = [col_str[:12], col_str[12:]] if len(col_str) > 12 else [col_str]
            
            if len(words) > 1:
                mid = len(words) // 2
                col_formatted = ' '.join(words[:mid]) + ' \\\\[-0.5ex] ' + ' '.join(words[mid:])
            else:
                col_formatted = col_str
        else:
            col_formatted = col_str
        header_cols.append(col_formatted)
    
    latex += "Country & " + " & ".join(header_cols) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for idx in display_df.index:
        row_data = [str(idx)]
        for col in df.columns:  # Use original column names for data access
            value = f"{df.loc[idx, col]:.3f}"
            if sig_matrix is not None and idx in sig_matrix.index and col in sig_matrix.columns:
                sig_val = sig_matrix.loc[idx, col]
                if sig_val and sig_val.strip():
                    value += f"$^{{{sig_val}}}$"
            row_data.append(value)
        latex += " & ".join(row_data) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    
    if sig_matrix is not None:
        latex += "\\begin{minipage}{\\textwidth}\n"
        latex += "\\footnotesize\n"
        latex += "\\textit{Notes:} Significance levels: *** p$<$0.001, ** p$<$0.01, * p$<$0.05, . p$<$0.1\n"
        latex += "\\end{minipage}\n"
    
    latex += "\\end{table}\n"
    return latex

# Create shortened feature names for LaTeX tables
feature_short_names = []
for feat in features:
    # Create more meaningful short names
    if len(feat) <= 15:
        feature_short_names.append(feat)
    elif 'Credibility' in feat and 'Social' in feat:
        feature_short_names.append('Social\\\\Credibility')
    elif 'Emotional' in feat and 'Environmental' in feat:
        feature_short_names.append('Environmental\\\\Emotion')
    elif 'Emotional' in feat and 'Social' in feat:
        feature_short_names.append('Social\\\\Emotion')
    elif 'Logical' in feat and 'Economic' in feat:
        feature_short_names.append('Economic\\\\Logic')
    elif 'Timing' in feat and 'Environmental' in feat:
        feature_short_names.append('Environmental\\\\Timing')
    else:
        # Generic shortening
        words = feat.replace('_', ' ').split()
        if len(words) >= 2:
            feature_short_names.append(words[0][:8] + '\\\\' + words[1][:8])
        else:
            feature_short_names.append(feat[:15])

# Generate LaTeX tables
table1_latex = df_to_latex_with_significance(
    avg_pivot,
    caption="Average Normalized Importance by Country and Feature",
    label="tab:avg_importance",
    feature_short_names=feature_short_names
)

table2_latex = df_to_latex_with_significance(
    ols_pivot,
    sig_matrix=significance_matrix,
    caption="OLS Predicted Importance with Statistical Significance",
    label="tab:ols_importance",
    feature_short_names=feature_short_names
)

if olr_success:
    table3_caption = "Ordered Logistic Regression Predicted Importance"
else:
    table3_caption = "OLS Results (OLR Model Failed to Converge)"

table3_latex = df_to_latex_with_significance(
    olr_pivot,
    caption=table3_caption,
    label="tab:olr_importance",
    feature_short_names=feature_short_names
)

# -------------------------------
# 10. Enhanced comprehensive analysis summary
# -------------------------------
comprehensive_summary_path = os.path.join(output_dir, 'comprehensive_analysis_summary.txt')
with open(comprehensive_summary_path, 'w') as f:
    f.write("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*40 + "\n")
    f.write(f"Total number of students: {len(df_raw)}\n")
    f.write(f"Number of countries: {len(countries)}\n")
    f.write(f"Number of features: {len(features)}\n\n")
    
    f.write("Countries and sample sizes:\n")
    for country, count in df_raw['Country'].value_counts().items():
        f.write(f"  {country}: {count} students\n")
    f.write("\n")
    
    f.write("Features analyzed:\n")
    for i, feat in enumerate(features, 1):
        f.write(f"  {i}. {feat}\n")
    f.write("\n")
    
    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("-"*40 + "\n")
    f.write(f"OLS Model R-squared: {ols_model.rsquared:.4f}\n")
    f.write(f"OLS Model Adj. R-squared: {ols_model.rsquared_adj:.4f}\n")
    f.write(f"OLS Model F-statistic: {ols_model.fvalue:.4f} (p = {ols_model.f_pvalue:.6f})\n")
    
    if olr_success:
        f.write(f"OLR Model Pseudo R-squared: {olr_fitted.prsquared:.4f}\n")
        f.write(f"OLR Model Log-likelihood: {olr_fitted.llf:.4f}\n")
        f.write(f"OLR Model AIC: {olr_fitted.aic:.4f}\n")
        f.write(f"OLR Model BIC: {olr_fitted.bic:.4f}\n")
    else:
        f.write("OLR Model: Failed to converge\n")
    f.write("\n")
    
    f.write("FEATURE IMPORTANCE RANKING (OVERALL)\n")
    f.write("-"*40 + "\n")
    for i, (feature, importance) in enumerate(overall_feature_importance.items(), 1):
        f.write(f"{i}. {feature}: {importance:.3f}\n")
    f.write("\n")
    
    f.write("COUNTRY-WISE FEATURE PREFERENCES\n")
    f.write("-"*40 + "\n")
    for country in countries:
        if country in avg_pivot.index:
            country_data = avg_pivot.loc[country]
            top_feature = country_data.idxmax()
            top_value = country_data.max()
            bottom_feature = country_data.idxmin()
            bottom_value = country_data.min()
            f.write(f"{country}:\n")
            f.write(f"  Most valued: {top_feature} ({top_value:.3f})\n")
            f.write(f"  Least valued: {bottom_feature} ({bottom_value:.3f})\n")
            f.write(f"  Range: {top_value - bottom_value:.3f}\n\n")
    
    f.write("CROSS-COUNTRY VARIATIONS\n")
    f.write("-"*40 + "\n")
    for feature in features:
        if feature in avg_pivot.columns:
            feature_values = avg_pivot[feature]
            f.write(f"{feature}:\n")
            f.write(f"  Mean: {feature_values.mean():.3f}\n")
            f.write(f"  Std Dev: {feature_values.std():.3f}\n")
            f.write(f"  Range: {feature_values.min():.3f} - {feature_values.max():.3f}\n")
            f.write(f"  Highest in: {feature_values.idxmax()}\n")
            f.write(f"  Lowest in: {feature_values.idxmin()}\n\n")
    
    # Key insights
    most_important = overall_feature_importance.index[0]
    least_important = overall_feature_importance.index[-1]
    f.write("KEY INSIGHTS\n")
    f.write("-"*40 + "\n")
    f.write(f"1. Most important feature overall: {most_important} ({overall_feature_importance.iloc[0]:.3f})\n")
    f.write(f"2. Least important feature overall: {least_important} ({overall_feature_importance.iloc[-1]:.3f})\n")
    f.write(f"3. Importance range: {overall_feature_importance.iloc[-1]:.3f} to {overall_feature_importance.iloc[0]:.3f}\n")
    f.write(f"4. Model explains {ols_model.rsquared*100:.1f}% of variance in importance ratings\n")
    
    if olr_success:
        f.write(f"5. OLR model successfully converged with pseudo R²: {olr_fitted.prsquared:.3f}\n")
    else:
        f.write("5. OLR model failed to converge - using OLS predictions as fallback\n")

print(f"Comprehensive analysis summary saved to: {comprehensive_summary_path}")

# -------------------------------
# 11. Save all results to files
# -------------------------------
# Save individual LaTeX tables
table1_path = os.path.join(output_dir, 'table1_average_importance.tex')
with open(table1_path, 'w') as f:
    f.write(table1_latex)

table2_path = os.path.join(output_dir, 'table2_ols_significance.tex') 
with open(table2_path, 'w') as f:
    f.write(table2_latex)

table3_path = os.path.join(output_dir, 'table3_olr_results.tex')
with open(table3_path, 'w') as f:
    f.write(table3_latex)

# Save all tables combined
all_tables_path = os.path.join(output_dir, 'all_latex_tables.txt')
with open(all_tables_path, 'w') as f:
    f.write("LATEX TABLES FOR FEATURE IMPORTANCE ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write("% Table 1: Average Importance\n")
    f.write(table1_latex)
    f.write("\n\n% Table 2: OLS Results with Significance\n")
    f.write(table2_latex)
    f.write("\n\n% Table 3: OLR Results\n")
    f.write(table3_latex)

# Save data CSVs
avg_pivot.to_csv(os.path.join(output_dir, 'average_importance.csv'))
ols_pivot.to_csv(os.path.join(output_dir, 'ols_predicted_importance.csv'))
olr_pivot.to_csv(os.path.join(output_dir, 'olr_predicted_importance.csv'))
df_long.to_csv(os.path.join(output_dir, 'analysis_data_long_format.csv'), index=False)

# Save significance matrix
significance_matrix.to_csv(os.path.join(output_dir, 'statistical_significance_matrix.csv'))

# -------------------------------
# 12. Additional Diagnostic Plots
# -------------------------------
# Create model diagnostics
fig_diag, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Residuals vs Fitted
axes[0,0].scatter(df_long['Importance_OLS_pred'], 
                  df_long['Importance'] - df_long['Importance_OLS_pred'], 
                  alpha=0.6)
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_xlabel('Fitted Values')
axes[0,0].set_ylabel('Residuals')
axes[0,0].set_title('Residuals vs Fitted Values')

# Plot 2: Feature importance by country
feature_means = df_long.groupby(['Country', 'Feature'])['Importance'].mean().unstack()
feature_means.plot(kind='bar', ax=axes[0,1], width=0.8)
axes[0,1].set_title('Feature Importance by Country')
axes[0,1].set_ylabel('Average Importance')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Distribution of importance scores
axes[1,0].hist(df_long['Importance'], bins=20, alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel('Importance Score')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Distribution of Importance Scores')

# Plot 4: Model comparison (OLS vs OLR if successful)
if olr_success:
    axes[1,1].scatter(df_long['Importance_OLS_pred'], df_long['Importance_OLR'], alpha=0.6)
    axes[1,1].plot([0, 1], [0, 1], 'r--')
    axes[1,1].set_xlabel('OLS Predictions')
    axes[1,1].set_ylabel('OLR Predictions')
    axes[1,1].set_title('OLS vs OLR Predictions')
    
    # Calculate correlation
    correlation = np.corrcoef(df_long['Importance_OLS_pred'], df_long['Importance_OLR'])[0,1]
    axes[1,1].text(0.05, 0.95, f'r = {correlation:.3f}', transform=axes[1,1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    # Show actual vs predicted for OLS
    axes[1,1].scatter(df_long['Importance'], df_long['Importance_OLS_pred'], alpha=0.6)
    axes[1,1].plot([0, 1], [0, 1], 'r--')
    axes[1,1].set_xlabel('Actual Importance')
    axes[1,1].set_ylabel('OLS Predicted')
    axes[1,1].set_title('Actual vs OLS Predicted')
    
    # Calculate R-squared
    axes[1,1].text(0.05, 0.95, f'R² = {ols_model.rsquared:.3f}', transform=axes[1,1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
diag_path = os.path.join(output_dir, 'model_diagnostics.png')
plt.savefig(diag_path, dpi=300, bbox_inches='tight')
print(f"Model diagnostics saved as: {diag_path}")
plt.show()

# -------------------------------
# 13. Feature Ranking Comparison
# -------------------------------
# Create a comparison of feature rankings across methods
ranking_comparison = pd.DataFrame({
    'Raw_Average': overall_feature_importance.rank(ascending=False),
    'OLS_Average': df_long.groupby('Feature')['Importance_OLS_pred'].mean().rank(ascending=False),
    'OLR_Average': df_long.groupby('Feature')['Importance_OLR'].mean().rank(ascending=False)
})

# Add actual values for reference
ranking_comparison['Raw_Score'] = overall_feature_importance
ranking_comparison['OLS_Score'] = df_long.groupby('Feature')['Importance_OLS_pred'].mean()
ranking_comparison['OLR_Score'] = df_long.groupby('Feature')['Importance_OLR'].mean()

# Sort by raw average ranking
ranking_comparison = ranking_comparison.sort_values('Raw_Average')

# Save ranking comparison
ranking_path = os.path.join(output_dir, 'feature_ranking_comparison.csv')
ranking_comparison.to_csv(ranking_path)

# Create final summary of all saved files
files_summary_path = os.path.join(output_dir, 'saved_files_summary.txt')
with open(files_summary_path, 'w') as f:
    f.write("SAVED FILES SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write("All analysis results have been saved to:\n")
    f.write(f"{output_dir}\n\n")
    f.write("FILES CREATED:\n")
    f.write("-"*30 + "\n")
    f.write("1. VISUALIZATIONS:\n")
    f.write("   - feature_importance_heatmaps.png\n")
    f.write("   - model_diagnostics.png\n\n")
    f.write("2. LATEX TABLES:\n")
    f.write("   - table1_average_importance.tex\n")
    f.write("   - table2_ols_significance.tex\n")
    f.write("   - table3_olr_results.tex\n")
    f.write("   - all_latex_tables.txt\n\n")
    f.write("3. DATA FILES:\n")
    f.write("   - average_importance.csv\n")
    f.write("   - ols_predicted_importance.csv\n")
    f.write("   - olr_predicted_importance.csv\n")
    f.write("   - analysis_data_long_format.csv\n")
    f.write("   - statistical_significance_matrix.csv\n")
    f.write("   - feature_ranking_comparison.csv\n\n")
    f.write("4. ANALYSIS REPORTS:\n")
    f.write("   - data_overview.txt\n")
    f.write("   - feature_analysis.txt\n")
    f.write("   - ols_model_results.txt\n")
    if olr_success:
        f.write("   - olr_model_results.txt\n")
    else:
        f.write("   - olr_error_log.txt\n")
    f.write("   - comprehensive_analysis_summary.txt\n")
    f.write("   - saved_files_summary.txt\n\n")
    f.write(f"Total files created: {13 + (1 if olr_success else 1)}\n\n")
    f.write("RECOMMENDED NEXT STEPS:\n")
    f.write("-"*30 + "\n")
    f.write("1. Review the comprehensive_analysis_summary.txt for key insights\n")
    f.write("2. Use the LaTeX tables in your academic paper\n")
    f.write("3. Include the heatmaps as figures in your analysis\n")
    f.write("4. Check model_diagnostics.png for model validation\n")
    f.write("5. Use feature_ranking_comparison.csv for cross-method validation\n")

print("\n" + "="*80)
print("ENHANCED ANALYSIS COMPLETE!")
print("="*80)
print(f"All results saved to: {output_dir}")
print(f"Summary of saved files: {os.path.basename(files_summary_path)}")
if olr_success:
    print("✓ OLR model converged successfully")
else:
    print("⚠ OLR model failed - check error log for details")
print("="*80)