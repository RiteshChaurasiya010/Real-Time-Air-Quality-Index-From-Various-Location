import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv(r"C:\Users\91707\OneDrive\Desktop\toolboxproject.csv")
print(df)

# 1. Data Information
info = df.info()

# 2. Summary of Data
description = df.describe()

# 3. Total Rows and Columns
shape = df.shape

# 4. Max Values
max_vals = df.max(numeric_only=True)

# 5. Min Values
min_vals = df.min(numeric_only=True)

# 6. Mode
mode_vals = df.mode().iloc[0]

# 7. Head
head_vals = df.head()

# 8. Tail
tail_vals = df.tail()

# 9. Arrange Data (Sort by pollutant_avg descending)
sorted_df = df.sort_values(by='pollutant_avg', ascending=False)

# 10. loc Function (Example: Andhra Pradesh)
andhra_df = df.loc[df['state'] == 'Andhra_Pradesh']

# 11. Check for Missing Values
missing_values = df.isnull().sum()

# 12. Drop Duplicates
df_cleaned = df.drop_duplicates()

# 13. Correlation Matrix
correlation = df_cleaned.corr(numeric_only=True)

# 14. Histogram
plt.figure(figsize=(8, 4))
df['pollutant_avg'].hist(bins=30)
plt.title("Histogram of Pollutant Average")
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\histogram_pollutant_avg.png")
plt.show()

# 15. Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='pollutant_id', y='pollutant_avg', data=df)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\boxplot_pollutant_avg.png")
plt.show()

# 16. Countplot of Cities
plt.figure(figsize=(10, 5))
sns.countplot(y='city', data=df, order=df['city'].value_counts().index[:10])
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\countplot_cities.png")
plt.show()

# 17. Bar Plot for Max Pollutant per Type
pollutant_max_by_type = df.groupby('pollutant_id')['pollutant_max'].max().sort_values(ascending=False)
pollutant_max_by_type.plot(kind='bar', figsize=(8, 4), title='Max Pollutant by Type')
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\barplot_max_pollutant_by_type.png")
plt.show()

# 18. Scatter Plot of Latitude vs Longitude
plt.figure(figsize=(8, 6))
sns.scatterplot(x='longitude', y='latitude', data=df, hue='pollutant_id', edgecolor='w', s=50)
plt.title("Station Locations")
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\scatter_station_locations.png")
plt.show()

# 19. Pollutant ID Counts
pollutant_counts = df['pollutant_id'].value_counts()

# 20. Average Pollutant per City
avg_pollutant_city = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False)

# 21. Station with Highest Pollutant Avg
max_station = df.loc[df['pollutant_avg'].idxmax()]

# 22. Unique States
unique_states = df['state'].unique()

# 23. Number of Unique Cities
num_unique_cities = df['city'].nunique()

# 24. Pollutant Avg by Type
avg_by_type = df.groupby('pollutant_id')['pollutant_avg'].mean()

# 25. Filtered Data Where Pollutant Avg > 50
high_pollution_df = df[df['pollutant_avg'] > 50]

# 26. Median of Pollutant Avg
median_pollutant = df['pollutant_avg'].median()

# 27. Create New Column: Pollutant Range
df['pollutant_range'] = df['pollutant_max'] - df['pollutant_min']

# 28. Top 5 States by Mean Pollutant Avg
top_states_pollution = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(5)

# 29. Crosstab of City vs Pollutant Type
city_pollutant_crosstab = pd.crosstab(df['city'], df['pollutant_id'])

# 30. Outlier Detection - IQR Method
Q1 = df['pollutant_avg'].quantile(0.25)
Q3 = df['pollutant_avg'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['pollutant_avg'] < Q1 - 1.5 * IQR) | (df['pollutant_avg'] > Q3 + 1.5 * IQR)]
print("\nOutliers in Pollutant Average (IQR Method):")
print(outliers_iqr[['pollutant_avg', 'city', 'state']])

# 31. Correlation Graph - Pairplot
sns.pairplot(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']])
plt.suptitle("Pairwise Correlation Graph", y=1.02)
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\pairplot_correlation.png")
plt.show()

# 32. Stripplot (Replaces Swarmplot)
plt.figure(figsize=(10, 6))
sns.stripplot(x="pollutant_id", y="pollutant_avg", data=df, jitter=True, size=4)
plt.title("Stripplot of Pollutant Avg by Type (Replaces Swarmplot)")
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\stripplot_pollutant_avg.png")
plt.show()

# 33. Donut Chart for Pollutant ID Distribution
pollutant_counts = df['pollutant_id'].value_counts()
plt.figure(figsize=(8, 8))
colors = sns.color_palette("pastel")[0:len(pollutant_counts)]
plt.pie(pollutant_counts, labels=pollutant_counts.index, colors=colors, startangle=90, wedgeprops=dict(width=0.4))
plt.title("Distribution of Pollutant Types (Donut Chart)")
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\donut_pollutant_distribution.png")
plt.show()

# 34. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[['pollutant_min', 'pollutant_max', 'pollutant_avg', 'pollutant_range']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Pollutant Metrics")
plt.tight_layout()
plt.savefig(r"C:\Users\91707\OneDrive\Desktop\correlation_heatmap.png")
plt.show()
