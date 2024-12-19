#task 1.2 exploratory data analysis on those data & communicate useful insights. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Check for data types and missing values
print(telecom_df.info())
print(telecom_df.isnull().sum())

# Identify non-numeric columns
non_numeric_columns = telecom_df.select_dtypes(include=['object']).columns
print("\nNon-Numeric Columns:")
print(non_numeric_columns)
# Convert 'Start' and 'End' to datetime
telecom_df['Start'] = pd.to_datetime(telecom_df['Start'], errors='coerce')
telecom_df['End'] = pd.to_datetime(telecom_df['End'], errors='coerce')

# Convert all other columns to numeric, coercing errors
for col in telecom_df.columns:
    if col not in ['Start', 'End'] + list(non_numeric_columns):
        telecom_df[col] = pd.to_numeric(telecom_df[col], errors='coerce')
# Display missing values for analysis
missing_values = telecom_df.isnull().sum()
print("\nMissing values before filling:")
print(missing_values[missing_values > 0])

# Fill or handle missing values based on the context
# List of numeric columns to fill with mean/median
numeric_columns = [
    'Start', 'Start ms', 'End', 'End ms', 'Dur. (ms)', 'Avg RTT DL (ms)', 
    'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
    'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', 
    '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 
    'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)', 
    '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)', 
    'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)', 
    'Activity Duration UL (ms)', 'Dur. (ms).1', 
    'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B', 
    'Nb of sec with 31250B < Vol DL < 125000B', 'Nb of sec with 37500B < Vol UL', 
    'Nb of sec with 6250B < Vol DL < 31250B', 'Nb of sec with 6250B < Vol UL < 37500B', 
    'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B', 
    'Total UL (Bytes)', 'Total DL (Bytes)'
]

# Fill numeric columns with mean
for col in numeric_columns:
    if telecom_df[col].isnull().sum() > 0:
        telecom_df[col].fillna(telecom_df[col].mean(), inplace=True)
 # Handle categorical columns
categorical_columns = ['Last Location Name', 'Handset Manufacturer', 'Handset Type']
for col in categorical_columns:
    if telecom_df[col].isnull().sum() > 0:
        telecom_df[col].fillna(telecom_df[col].mode()[0], inplace=True)       
# Verify that there are no more missing values
print("\nMissing values after filling:")
print(telecom_df.isnull().sum())

# Step 1: Calculate Total Duration for each user
# Using 'IMSI' as the user identifier
user_total_duration = telecom_df.groupby('IMSI')['Dur. (ms)'].sum().reset_index()
user_total_duration.rename(columns={'Dur. (ms)': 'Total_Duration'}, inplace=True)
# Step 2: Create Decile Classes
user_total_duration['Decile_Class'] = pd.qcut(user_total_duration['Total_Duration'], 10, labels=False) + 1
# Step 3: Compute Total Data for each user
data_per_user = telecom_df.groupby('IMSI')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum().reset_index()
data_per_user['Total_Data'] = data_per_user['Total DL (Bytes)'] + data_per_user['Total UL (Bytes)']
# Merge the total duration with the data per user
merged_data = pd.merge(user_total_duration, data_per_user, on='IMSI')
# Step 4: Compute Total Data per Decile Class
decile_summary = merged_data.groupby('Decile_Class')['Total_Data'].sum().reset_index()
# Display the summary
print("\nTotal Data per Decile Class:")
print(decile_summary)

decile_summary['Total_Data'] = decile_summary['Total_Data'] / 1e6  # Convert to Megabytes for better readability

# Step 2: Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Decile_Class', y='Total_Data', data=decile_summary, palette='viridis')
plt.title('Total Data (DL + UL) per Decile Class')
plt.xlabel('Decile Class')
plt.ylabel('Total Data (MB)')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Step 3: Show the plot
plt.tight_layout()
plt.show()

# Function to identify outliers using IQR
def identify_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
# List of numeric columns to check for outliers
numeric_columns = [
    'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
    'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
    'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)',
    'TCP UL Retrans. Vol (Bytes)', 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
    'Activity Duration DL (ms)', 'Activity Duration UL (ms)',
    'Total UL (Bytes)', 'Total DL (Bytes)'
]
# Identify outliers for each numeric column
for col in numeric_columns:
    outliers = identify_outliers_iqr(telecom_df, col)
    print(f"\nOutliers in {col}:")
    print(outliers)
# Function to create boxplots for numeric columns
def plot_boxplots(df, columns):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(4, 4, i)  # Adjust the number of rows/columns based on the number of plots
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Plot boxplots for numeric columns
plot_boxplots(telecom_df, numeric_columns)

# Function to identify rare categories and group them
def group_rare_categories(df, column, threshold=0.05):
    # Calculate frequency of each category
    freq = df[column].value_counts(normalize=True)
    
    # Identify categories below the threshold
    rare_categories = freq[freq < threshold].index
    
    # Group rare categories into 'Other'
    df[column] = df[column].replace(rare_categories, 'Other')
    
    return df

# Apply to categorical columns
for col in ['Last Location Name', 'Handset Manufacturer', 'Handset Type']:
    telecom_df = group_rare_categories(telecom_df, col)

# Display the updated categorical data
print(telecom_df['Handset Manufacturer'].value_counts())

# Function to plot categorical data
def plot_categorical_counts(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[column], order=df[column].value_counts().index)
    plt.title(f'Count of Categories in {column}')
    plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

# Plot for each categorical column
for col in ['Last Location Name', 'Handset Manufacturer', 'Handset Type']:
    plot_categorical_counts(telecom_df, col)
# Calculate basic metrics for relevant columns
metrics_summary = {
    'Total Duration (ms)': {
        'Mean': telecom_df['Dur. (ms)'].mean(),
        'Median': telecom_df['Dur. (ms)'].median(),
        'Mode': telecom_df['Dur. (ms)'].mode()[0],
        'Standard Deviation': telecom_df['Dur. (ms)'].std(),
        'Range': telecom_df['Dur. (ms)'].max() - telecom_df['Dur. (ms)'].min(),
        '25th Percentile': telecom_df['Dur. (ms)'].quantile(0.25),
        '75th Percentile': telecom_df['Dur. (ms)'].quantile(0.75)
    },
    'Total Data (Bytes)': {
        'Mean': telecom_df['Total DL (Bytes)'].sum() + telecom_df['Total UL (Bytes)'].sum(),
        'Median': telecom_df['Total DL (Bytes)'].median() + telecom_df['Total UL (Bytes)'].median(),
        # More calculations as needed
    }
}

# Display the metrics summary
import pprint
pprint.pprint(metrics_summary)

# List of quantitative columns to analyze
quantitative_columns = [
    'Dur. (ms)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
    'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
    'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'
]

# Create a dictionary to store metrics for each column
metrics_summary = {}

# Compute metrics for each quantitative variable
for column in quantitative_columns:
    metrics_summary[column] = {
        'Mean': telecom_df[column].mean(),
        'Median': telecom_df[column].median(),
        'Standard Deviation': telecom_df[column].std(),
        'Range': telecom_df[column].max() - telecom_df[column].min(),
        '25th Percentile': telecom_df[column].quantile(0.25),
        '75th Percentile': telecom_df[column].quantile(0.75),
        'IQR': telecom_df[column].quantile(0.75) - telecom_df[column].quantile(0.25)
    }

# Display the metrics summary
import pprint
pprint.pprint(metrics_summary)

# Set the style for the plots
sns.set(style="whitegrid")

# List of quantitative columns to analyze
quantitative_columns = [
    'Dur. (ms)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
    'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
    'Activity Duration UL (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'
]
# Create a figure for the plots
plt.figure(figsize=(18, 12))

# Generate plots for each variable
for i, column in enumerate(quantitative_columns):
    plt.subplot(5, 3, i + 1)  # Arrange in a 5x3 grid
    sns.histplot(telecom_df[column], kde=True, bins=30)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# Box plots for key variables
plt.figure(figsize=(16, 8))
sns.boxplot(data=telecom_df[quantitative_columns])
plt.xticks(rotation=45)
plt.title('Box Plots of Quantitative Variables')
plt.show()

# Create a new column for total data
telecom_df['Total Data (Bytes)'] = telecom_df['Total DL (Bytes)'] + telecom_df['Total UL (Bytes)']
# List of application columns for data usage
application_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]
# 1. Correlation Analysis
correlation_results = {}
for app in application_columns:
    # Calculate correlation with total data for each application
    correlation_dl = telecom_df[app].corr(telecom_df['Total Data (Bytes)'])
    correlation_results[app] = correlation_dl

print("Correlation Coefficients with Total Data:")
for app, corr in correlation_results.items():
    print(f"{app}: {corr:.4f}")
# 2. Scatter Plots
plt.figure(figsize=(18, 12))
for i, app in enumerate(application_columns):
    plt.subplot(4, 4, i + 1)  # Arrange in a 4x4 grid
    sns.scatterplot(x=telecom_df[app], y=telecom_df['Total Data (Bytes)'])
    plt.title(f'Scatter Plot: {app} vs Total Data')
    plt.xlabel(app)
    plt.ylabel('Total Data (Bytes)')

plt.tight_layout()
plt.show()
# 3. Group By Analysis
grouped_data = telecom_df[application_columns].sum().reset_index()
grouped_data.columns = ['Application', 'Total Data (Bytes)']
print("\nTotal Data by Application:")
print(grouped_data.sort_values(by='Total Data (Bytes)', ascending=False))
# Select relevant columns for correlation analysis
correlation_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]
# Create a DataFrame with the selected columns
data_for_correlation = telecom_df[correlation_columns]

# Compute the correlation matrix
correlation_matrix = data_for_correlation.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix for Application Data Usage')
plt.show()

# Select relevant columns for PCA (using only application data columns)
pca_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(telecom_df[pca_columns])
#Perform PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.grid()
plt.show()
# Optional but let Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(len(pca_columns))])