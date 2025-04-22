import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('AvianData.xlsx')

# Step 1: Extract Year and Month from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 2: Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Step 3: Visualizations for Research Questions

# Research Question 1: Are Colisepticaemia cases seasonal?
monthly_cases = df.groupby('Month')['Colisepticaemia Cases'].mean()
plt.figure(figsize=(10, 6))
monthly_cases.plot(kind='line', marker='o')
plt.title('Average Colisepticaemia Cases by Month (2012–2024)')
plt.xlabel('Month')
plt.ylabel('Average Cases')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.savefig('seasonality_plot.png')
plt.show()

# Research Question 2: Which UK regions report the highest case numbers?
regional_cases = df.groupby('Region')['Colisepticaemia Cases'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
regional_cases.plot(kind='bar')
plt.title('Total Colisepticaemia Cases by Region (2012–2024)')
plt.xlabel('Region')
plt.ylabel('Total Cases')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.savefig('regional_cases_plot.png')
plt.show()

# Research Question 3: Does age affect susceptibility?
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age Category', y='Colisepticaemia Cases', data=df)
plt.title('Colisepticaemia Cases by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Cases')
plt.savefig('age_category_plot.png')
plt.show()

# Additional EDA: Environmental Factors
# Scatter plot: Cases vs. Temperature
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature (°C)'], df['Colisepticaemia Cases'], alpha=0.5)
plt.title('Colisepticaemia Cases vs. Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Cases')
plt.grid(True)
plt.savefig('cases_vs_temperature_plot.png')
plt.show()

# Scatter plot: Cases vs. Rainfall
plt.figure(figsize=(10, 6))
plt.scatter(df['Rainfall (mm)'], df['Colisepticaemia Cases'], alpha=0.5)
plt.title('Colisepticaemia Cases vs. Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Cases')
plt.grid(True)
plt.savefig('cases_vs_rainfall_plot.png')
plt.show()

# Correlation Heatmap
numerical_cols = ['Colisepticaemia Cases', 'Temperature (°C)', 'Rainfall (mm)']
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Step 4: Check for Issues
print("\nMissing Values:")
print(df.isnull().sum())

print("\nPotential Outliers in Cases:")
print(df[df['Colisepticaemia Cases'] > df['Colisepticaemia Cases'].quantile(0.95)])