# Air Quality Data Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r'C:\Users\Anshp\Downloads\Air_Quality.csv')

# Display dataset info before cleaning
print("\n--- Dataset Info Before Cleaning ---")
print(df.head())
print(df.info())
print(df.describe().round(1))
print(df.isnull().sum())
print(df.duplicated().sum())
# Remove rows with missing values
df.dropna(inplace=True)

# Convert 'Time Period' to numeric
df['Time Period'] = pd.to_numeric(df['Time Period'], errors='coerce')

# Drop rows with invalid 'Time Period' or 'Data Value'
df.dropna(subset=['Time Period', 'Data Value'], inplace=True)

# Categorize 'Data Value' into bins
bin_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
df['Geo Place Bin'] = pd.qcut(df['Data Value'], q=5, labels=bin_labels)

# Set a consistent style for all charts
sns.set_theme(style="whitegrid", palette="deep")

# Display dataset info after cleaning
print("\n--- Dataset Info After Cleaning ---")
print(df.head())
print(df.info())
print(df.describe().round(1))
print(df.isnull().sum())
print(df.duplicated().sum())
#correlation matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True).round(2))

# --- Visualization 1: Bar Chart ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Geo Place Bin', y='Data Value', errorbar=None, hue='Geo Place Bin', palette="Blues", legend=False)
plt.title('Average AQI by Geo Place Bin', fontsize=16, fontweight='bold')
plt.xlabel('Air Quality Categories', fontsize=14)
plt.ylabel('Average AQI', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()

# --- Visualization 2: Histogram ---
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Data Value', bins=20, kde=True, color="teal", edgecolor="black")
plt.title('Distribution of AQI', fontsize=16, fontweight='bold')
plt.xlabel('AQI', fontsize=14)
plt.ylabel('Frequency of Occurrence', fontsize=14)
plt.tight_layout()
plt.show()

# --- Visualization 3: Pie Chart ---
plt.figure(figsize=(10, 10))
name_counts = df['Name'].value_counts()
plt.pie(name_counts, labels=name_counts.index, autopct='%1.1f%%', 
startangle=140, colors=sns.color_palette("pastel", 
len(name_counts)), wedgeprops={'edgecolor': 'black'})
plt.title('Pollutant Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Visualization 4: Scatter Plot ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Time Period', y='Data Value', 
hue='Geo Type Name', palette="Set2", s=100, edgecolor="black")
plt.title('AQI vs Time Period', fontsize=16, fontweight='bold')
plt.xlabel('Time Period', fontsize=14)
plt.ylabel('AQI', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Visualization 5: Horizontal Bar Chart ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, y='Name', x='Data Value', errorbar=None,
hue='Data Value', palette="coolwarm", dodge=False, orient='h', legend=False)
plt.title('Pollutant Emission Analysis', fontsize=16, fontweight='bold')
plt.ylabel('Pollutant Name', fontsize=14)
plt.xlabel('Average AQI', fontsize=14)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Visualization 6: Heatmap ---
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.tight_layout()
plt.show()

# --- Visualization 7: Box Plot with Outlier Detection (IQR) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Geo Type Name', y='Data Value', 
hue='Geo Type Name', palette="Set1", legend=False)

# Calculate IQR for outlier detection
q1 = df['Data Value'].quantile(0.25)
q3 = df['Data Value'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

plt.axhline(lower_bound, color='red', 
linestyle='--', label='Lower Bound')
plt.axhline(upper_bound, color='blue', 
linestyle='--', label='Upper Bound')
plt.title('Outlier Detection in AQI by Geo Type Name',
fontsize=16, fontweight='bold')
plt.xlabel('Geo Type Name', fontsize=14)
plt.ylabel('AQI', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# --- Visualization 8: Z-Test for Outliers ---
from scipy.stats import zscore

# Calculate Z-scores for 'Data Value'
df['Z-Score'] = zscore(df['Data Value'])

# Identify outliers based on Z-scores
z_threshold = 3
z_outliers = df[(df['Z-Score'] > z_threshold) | (df['Z-Score'] < -z_threshold)]
print(f"Number of outliers detected using Z-test: {len(z_outliers)}")

# Scatter Plot with Z-Test Outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Geo Type Name',
    y='Data Value',
    hue=(df['Z-Score'] > z_threshold) | (df['Z-Score'] < -z_threshold),
    palette={True: 'crimson', False: 'steelblue'},
    s=100,
    edgecolor="black"
)
plt.title('Z-Test Outlier Detection in AQI by Geo Type Name',
fontsize=16, fontweight='bold')
plt.xlabel('Geo Type Name', fontsize=14)
plt.ylabel('AQI', fontsize=14)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles=handles,
    labels=['No', 'Yes'],  # Correct mapping of labels
    title='Outlier',
    fontsize=12,
    title_fontsize=14
)
plt.tight_layout()
plt.show()
