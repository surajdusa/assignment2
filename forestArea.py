import pandas as pd

def forest_area_cover(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, skiprows=4)
    
    # Extract data for years 2010-2019
    years_df = df.loc[:, 'Country Name':'2019']
    
    # Convert year column names to strings
    years_df.columns = [str(col) if col.isdigit() else col for col in years_df.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countries_df = years_df.transpose()
    
    # Replace missing values with 0
    countries_df.fillna(0, inplace=True)
    
    # Set the column names for the countries DataFrame
    countries_df.columns = countries_df.iloc[0]
    countries_df = countries_df[1:]
    countries_df.index.name = 'Year'
    
    # Rename the 'Country Name' column to 'Year' and set it as the index for the years DataFrame
    years_df = years_df.rename(columns={'Country Name': 'Year'})
    years_df.set_index('Year', inplace=True)
    
    return years_df, countries_df



#calling the function we created above
years_df, countries_df = forest_area_cover('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5339009.csv')

years_df

countries_df


# GETTING THE STATISTICAL PROPERTIES AND THE CORRELATIONS OF THE DATA
countries_df.describe()

years_df.corr()

# PLOTTING VARIOUS VISUALIZATIONS

# A Line Plot
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas dataframe
df = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5339009.csv', skiprows=4)

# List of 10 countries to plot
countries_to_plot = ["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]

# Filter the dataframe to only include the desired countries and years
df_filtered = df.loc[df['Country Name'].isin(countries_to_plot), ['Country Name', '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']]

# Set the index to be the country names
df_filtered = df_filtered.set_index('Country Name')

# Transpose the dataframe so that the years are on the x-axis
df_filtered = df_filtered.transpose()

# Create the line plot
ax = df_filtered.plot(kind='line', figsize=(10,6))

# Set the title and axis labels
ax.set_title('Forest area (% of land area) for 10 countries')
ax.set_xlabel('Year')
ax.set_ylabel('Forest area (% of land area)')

# Show the plot
plt.show()


# A Grouped Bar Graph
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV file
df = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5339009.csv', skiprows=4)

# Select the desired countries and years
countries = ["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]
years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']

df = df[df['Country Name'].isin(countries)][['Country Name', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

# Set the Country Name as the index
df.set_index('Country Name', inplace=True)

# Transpose the data frame
df_transposed = df.transpose()

# Plot the bar chart
ax = df_transposed.plot(kind='bar', width=0.8, figsize=(12,6))

# Set the chart title and labels
ax.set_title('Forest area (% of land area) for 10 selected countries', fontsize=14)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Forest area (% of land area)', fontsize=12)

# Set the x-tick labels
ax.set_xticklabels(years, rotation=0)

# Show the plot
plt.show()


# An Area Graph
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas dataframe
df = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5339009.csv', skiprows=4)

# List of 10 countries to plot
countries = ["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]

# Filter the dataframe to only include data for the selected countries and years
df = df[df['Country Name'].isin(countries)][['Country Name', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

# Set the country name as the index
df.set_index('Country Name', inplace=True)

# Transpose the dataframe so that the years become the index
df = df.T

# Set the plot style to 'fivethirtyeight' for a nice aesthetic
plt.style.use('fivethirtyeight')

# Plot the data as an area chart
df.plot(kind='area', figsize=(12,8), alpha=0.5)

# Set the chart title and axis labels
plt.title('Forest Area (% of Land Area) for 10 Countries')
plt.xlabel('Year')
plt.ylabel('Forest Area (% of Land Area)')

# Add a legend and display the plot
plt.legend()
plt.show()


# A Data Frame With Different Indicators
import pandas as pd

# Read in the CSV files
df_forest = pd.read_csv('API_AG.LND.FRST.ZS_DS2_en_csv_v2_5339009.csv', skiprows=4)
df_agri = pd.read_csv('API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5337949.csv', skiprows=4)
df_urban = pd.read_csv('API_SP.URB.GROW_DS2_en_csv_v2_5226876.csv', skiprows=4)

# Select only the columns of interest
countries =["United States", "China", "United Kingdom", "Russian Federation", "France", "Japan", "Germany", "Israel", "United Arab Emirates", "Saudi Arabia"]
df_forest = df_forest.loc[df_forest['Country Name'].isin(countries), ['Country Name', '2020']]
df_agri = df_agri.loc[df_agri['Country Name'].isin(countries), ['Country Name', '2020']]
df_urban = df_urban.loc[df_urban['Country Name'].isin(countries), ['Country Name', '2020']]

# Merge the three dataframes into one based on 'Country Name'
df = pd.merge(df_forest, df_agri, on='Country Name')
df = pd.merge(df, df_urban, on='Country Name')

# Rename the columns
df.columns = ['Country Name', 'forest_area', 'agri_land', 'urban_pop_growth']

# Add a 'year' column with the value 2020
df['year'] = 2020

# Reorder the columns
df = df[['Country Name', 'year', 'forest_area', 'agri_land', 'urban_pop_growth']]

# Print the resulting dataframe

df

# A Correlation Heat Map
import matplotlib.pyplot as plt
import numpy as np

# Drop the year column
df_corr = df.drop('year', axis=1)

# Create a correlation matrix from the dataframe
corr_matrix = df_corr.corr()

# Create a heatmap using matshow from numpy and annotate the cells with the correlation values
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.matshow(corr_matrix, cmap='coolwarm')

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        c = round(corr_matrix.iloc[i, j], 2)
        ax.text(j, i, str(c), va='center', ha='center')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va='bottom')

# Set the x and y axis tick labels to the column names
ax.set_xticklabels([''] + list(corr_matrix.columns))
ax.set_yticklabels([''] + list(corr_matrix.index))

# Rotate the x axis tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

# Set the title
ax.set_title('Correlation Heatmap')

# Show the plot
plt.show()



