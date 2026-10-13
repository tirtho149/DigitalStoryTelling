import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"

# Make sure output directory exists
output_dir = "../figures/video_ab"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Load Data
# =========================
file_path = "../data/video_ab_survey/study3Final.csv"

try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# Debug: Print column names to see what we're working with
print("\n📋 Column names in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

# Clean column names (remove trailing/leading spaces)
df.columns = df.columns.str.strip()

# =========================
# Check for required columns
# =========================
required_cols = ["Enhanced", "Selected Video", "Country", "Sector", "Feature"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n❌ Missing required columns: {missing_cols}")
    print("Available columns:", list(df.columns))
    
    # Try to find similar column names
    print("\nTrying to find similar column names...")
    for req_col in missing_cols:
        similar = [col for col in df.columns if req_col.lower() in col.lower() or col.lower() in req_col.lower()]
        if similar:
            print(f"  For '{req_col}', found similar: {similar}")
    exit()

# Keep only relevant columns
available_cols = [col for col in required_cols if col in df.columns]
df = df[available_cols]

print(f"\n📊 Working with columns: {available_cols}")
print(f"Data shape after column selection: {df.shape}")

# =========================
# Data cleaning and inspection
# =========================
print(f"\n🔍 Data inspection before filtering:")
print(f"Enhanced values: {df['Enhanced'].value_counts().to_dict()}")
print(f"Selected Video values: {df['Selected Video'].value_counts().to_dict()}")

# Drop rows with missing values in the required columns
df_clean = df.dropna(subset=available_cols)
print(f"Rows after removing missing values: {df_clean.shape[0]}")

# Clean string values
for col in available_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

# =========================
# Filter: Enhanced == Selected Video (with proper mapping)
# =========================
# Create the condition with proper mapping
# Enhanced "A" should match "Video A", Enhanced "B" should match "Video B"
condition = (
    ((df_clean["Enhanced"] == "A") & (df_clean["Selected Video"] == "Video A")) |
    ((df_clean["Enhanced"] == "B") & (df_clean["Selected Video"] == "Video B"))
)
df_filtered = df_clean[condition].copy()

print(f"\n🎯 Filtering results:")
print(f"Total rows: {len(df_clean)}")
print(f"Rows where Enhanced matches Selected Video: {len(df_filtered)}")

# Show the breakdown
print(f"  A -> Video A matches: {len(df_clean[(df_clean['Enhanced'] == 'A') & (df_clean['Selected Video'] == 'Video A')])}")
print(f"  B -> Video B matches: {len(df_clean[(df_clean['Enhanced'] == 'B') & (df_clean['Selected Video'] == 'Video B')])}")

if len(df_filtered) == 0:
    print("❌ No rows match the filter condition. Let's examine the data:")
    print("Enhanced vs Selected Video comparison:")
    comparison = pd.crosstab(df_clean["Enhanced"], df_clean["Selected Video"], margins=True)
    print(comparison)
    exit()

# =========================
# Helper plotting function for match percentages
# =========================
def make_percentage_chart(full_data, filtered_data, column, title, filename):
    """Create a percentage match chart for the specified column"""
    if column not in full_data.columns:
        print(f"⚠️ Column '{column}' not found in data. Skipping plot.")
        return
    
    # Calculate total counts for each category
    total_counts = full_data[column].value_counts()
    
    # Calculate match counts for each category
    match_counts = filtered_data[column].value_counts()
    
    # Calculate percentages
    percentages = {}
    for category in total_counts.index:
        matches = match_counts.get(category, 0)
        total = total_counts[category]
        percentage = (matches / total) * 100
        percentages[category] = percentage
    
    # Sort by percentage descending
    percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))
    
    if not percentages:
        print(f"⚠️ No data for column '{column}'. Skipping plot.")
        return
    
    # Create colors similar to your chart
    colors = ['#4A90A4', '#8B5A8C', '#D4932B', '#C85A54', '#6B8C42']
    
    plt.figure(figsize=(12, 8))
    categories = list(percentages.keys())
    values = list(percentages.values())
    
    bars = plt.bar(categories, values, 
                   color=colors[:len(categories)], 
                   edgecolor='black', 
                   linewidth=0.8)
    
    # Titles & labels
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Match Percentage (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    
    # Set y-axis to go from 0 to reasonable max
    plt.ylim(0, max(values) * 1.1)
    
    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, values):
        plt.annotate(f'{percentage:.1f}%',
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"❌ Error saving plot: {e}")
        plt.close()

# =========================
# Generate Figures
# =========================
print(f"\n📈 Generating percentage match plots for {len(df_filtered)} filtered records out of {len(df_clean)} total...")

# Generate percentage plots
plot_configs = [
    ("Country", "Enhanced Algorithm Performance by Country", "country_percentage.png"),
    ("Sector", "Enhanced Algorithm Performance by Sector", "sector_percentage.png"),
    ("Feature", "Enhanced Algorithm Performance by Feature\n(Study 3)", "feature_percentage.png")
]

for column, title, filename in plot_configs:
    if column in df_clean.columns:
        make_percentage_chart(df_clean, df_filtered, column, title, filename)
    else:
        print(f"⚠️ Column '{column}' not available for plotting")

# =========================
# Summary Statistics with Percentages
# =========================
print(f"\n📋 Match percentage summary:")
for col in ['Country', 'Sector', 'Feature']:
    if col in df_clean.columns:
        print(f"\n{col} Match Percentages:")
        total_counts = df_clean[col].value_counts()
        match_counts = df_filtered[col].value_counts()
        
        for category in total_counts.index:
            matches = match_counts.get(category, 0)
            total = total_counts[category]
            percentage = (matches / total) * 100
            print(f"  {category}: {matches}/{total} = {percentage:.1f}%")

print(f"\n🎉 Finished. Figures saved in '{output_dir}' directory.")
print(f"Total plots generated: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")