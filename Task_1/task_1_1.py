import pandas as pd
# Load the datasets from CSV files
telecom_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GitHub/Data-weak2/Data/Copy of Week2_challenge_data_source(CSV).csv')
print("Telecom Data:")
print(telecom_df.head())

# Step 1: Identify the Top 10 Handsets Used by Customers (based on Handset Type)
top_handsets = telecom_df['Handset Type'].value_counts().head(10)
print("\nTop 10 Handsets:\n", top_handsets)

# Step 2: Identify the Top 3 Handset Manufacturers
top_manufacturers = telecom_df['Handset Manufacturer'].value_counts().head(3)
print("\nTop 3 Handset Manufacturers:\n", top_manufacturers)

# Step 3: Identify the Top 5 Handsets per Top 3 Manufacturers
top_3_manufacturers = top_manufacturers.index.tolist()
top_handsets_per_manufacturer = {}

for manufacturer in top_3_manufacturers:
    handsets = telecom_df[telecom_df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
    top_handsets_per_manufacturer[manufacturer] = handsets

print("\nTop Handsets per Manufacturer:")
for manufacturer, handsets in top_handsets_per_manufacturer.items():
    print(f"\n{manufacturer}:\n{handsets}")

print("Columns in the Telecom Data:")
print(telecom_df.columns)

# Plotting User Behavior Overview
plt.figure(figsize=(16, 8))

# Total Downloads and Total Uploads
plt.subplot(2, 2, 1)
sns.barplot(data=user_behavior, x='MSISDN/Number', y='total_dl', color='blue')
plt.title('Total Download Bytes per User')
plt.xlabel('User (MSISDN)')
plt.ylabel('Total Download Bytes')
plt.xticks(rotation=90)

plt.subplot(2, 2, 2)
sns.barplot(data=user_behavior, x='MSISDN/Number', y='total_ul', color='orange')
plt.title('Total Upload Bytes per User')
plt.xlabel('User (MSISDN)')
plt.ylabel('Total Upload Bytes')
plt.xticks(rotation=90)

# Number of Sessions and Total Duration
plt.subplot(2, 2, 3)
sns.barplot(data=user_behavior, x='MSISDN/Number', y='number_of_sessions', color='green')
plt.title('Number of Sessions per User')
plt.xlabel('User (MSISDN)')
plt.ylabel('Number of Sessions')
plt.xticks(rotation=90)

plt.subplot(2, 2, 4)
sns.barplot(data=user_behavior, x='MSISDN/Number', y='total_duration', color='purple')
plt.title('Total Duration (ms) per User')
plt.xlabel('User (MSISDN)')
plt.ylabel('Total Duration (ms)')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()