#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# Dispaly max column
pd.pandas.set_option('display.max_columns',None)
# Display max rows
pd.pandas.set_option('display.max_rows', None)


# # Task - 1

# In[6]:


#loading dataset using pandas libraray
df = pd.read_excel("telcom_data.xlsx")

# print the top5 records
df.head(5)


# In[6]:


df.tail()


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe().T


# In[11]:


df.nunique()


# In[12]:


df.isnull().sum()


# In[13]:


## Here we will check the percentage of null values present in each feature
    
# Step 1: Make a list of features with missing values
features_with_na=[features for features in df.columns if df[features].isnull().sum()>1]

# Step 2: Print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, np.round(df[feature].isnull().mean()*100, 4),  ' % missing values')


# In[14]:


df["Handset Type"].unique()


# In[15]:


df["Handset Manufacturer"].value_counts()


# In[16]:


df["Handset Manufacturer"].unique()


# # Top 10 handsets used by the customers.

# In[17]:


#Identify the top 10 handsets used by the customers
top_10_handsets = df['Handset Type'].value_counts().head(10)

#Display the top 10 handsets used by customers
print("Top 10 handsets used by customres:")
print(top_10_handsets)


# In[18]:


#Identify the top 10 handsets used by the customers
top_10_handsets = df['Handset Type'].value_counts().head(10)

##plot the top 10 handsets
plt.figure(figsize=(10,4))
top_10_handsets.plot(kind='bar')
plt.title('Top 10 Handsets Used by Customers')
plt.xlabel('Handset Type')
plt.ylabel('Number of Customers Count')
plt.xticks(rotation=80)
plt.show()


# # Top 3 handset manufactures

# In[19]:


# identify the top 3 handset customers
top_manufacturers = df['Handset Manufacturer'].value_counts().head()

# Display the top 3 handset manfacturers
top_manufacturers


# In[20]:


# Identify the top 5 handsets per top 3 handset manufacturer

# Top 3 manufacturers
manufacturers = ['Apple', 'Samsung', 'Huawei']

# Initialize a dictionary to store the results
top_handsets_per_manufacturer = {}

# Loop through each manufacturer and find their top 5 handsets
for manufacturer in manufacturers:
    top_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
    top_handsets_per_manufacturer[manufacturer] = top_handsets#.to_list()

# Display the top 5 handsets per top 3 handset manufacturer
print("The top 5 handsets for each of the top 3 manufacturers are as follows:")

for manufacturer, top_handsets in top_handsets_per_manufacturer.items():
    print(f"Top 5 handsets for {manufacturer}: {top_handsets}")


# In[21]:


## Identify the top 3 handset manufacturers
top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)

## Plot the top 3 handset manufacturers
plt.figure(figsize=(5, 10))
top_3_manufacturers.plot.bar(x='Handset Manufacturer', title="Top 3 Handset Manufacturers", stacked=True, color='#343aeb')
plt.ylabel('Handset Manufacturer Count')
plt.xticks(rotation=80)
plt.show()


# # Numerical Variables

# In[22]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# # Identifying Outliers in Numerical Features

# In[23]:


# List of columns to check for outliers
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Initialize a dictionary to store the results
outliers = {}

# Loop through each numeric column and find outliers using the IQR method
for feature in numerical_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    outlier_condition = (df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))
    
    # Store the number of outliers in the dictionary
    outliers[feature] = df[outlier_condition].shape[0]

# Display the number of outliers in each column
outliers


# # Categorical Variables

# In[24]:


categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
df[categorical_features].head()


# # Top 5 handsets per top 3 Handset Manufacturer

# In[25]:


# Top apple manufacturer handset
top_apple = df.loc[df['Handset Manufacturer'] == 'Apple']
top_apple = top_apple.groupby(['Handset Manufacturer', 'Handset Type']).agg({'Handset Type': ['count']})
top_apple.columns = ['count']
top_apple=top_apple.nlargest(5, 'count')
top_apple


# In[26]:


plt.figure(figsize=(5, 5))
top_apple.plot.bar(y='count', title="Top 5 Apple Handsets", stacked=True, color='#eb3490')
plt.ylabel('Handset Count')
plt.xticks(rotation=80)
plt.show()


# In[27]:


## Top samsung manufacturer handset
top_samsung = telcom
.loc[telcom['Handset Manufacturer'] == 'Samsung']
top_samsung = top_samsung.groupby(['Handset Manufacturer', 'Handset Type']).agg({'Handset Type': ['count']})
top_samsung.columns = ['count']
top_samsung=top_samsung.nlargest(5, 'count')
top_samsung


# In[28]:


## Top huawei manufacturer handset
top_huawei = df.loc[df['Handset Manufacturer'] == 'Huawei']
top_huawei = top_huawei.groupby(['Handset Manufacturer', 'Handset Type']).agg({'Handset Type': ['count']})
top_huawei.columns = ['count']
top_huawei=top_huawei.nlargest(5, 'count')
top_huawei


# In[29]:


plt.figure(figsize=(5, 5))
top_huawei.plot.bar(y='count', title="Top 5 Huawei Handsets", stacked=True, color='#eb3490')
plt.ylabel('Handset Count')
plt.xticks(rotation=75)
plt.show()


# # Task 1.1 - Your employer wants to have an overview of the users' behaviour on those applications

# # Number of xDR sessions

# In[30]:


# Aggregate the number of xDR sessions per user
data = df.copy()
user_xdr_sessions = data.groupby(['IMSI','MSISDN/Number'])["Bearer Id"].nunique().reset_index()
user_xdr_sessions.columns = ['IMSI','MSISDN/Number','Number of xDR Sessions',]
user_xdr_sessions.head()


# # Session duration

# In[31]:


# Aggregate the session duration per user
data = df.copy()
user_session_duration = data.groupby(['IMSI', 'MSISDN/Number'])['Dur. (ms)'].sum().reset_index()
user_session_duration.columns = ['IMSI', 'MSISDN/Number', 'Total Session Duration (ms)']
user_session_duration.head()


# In[32]:


data = df.copy()
data['Total Data (DL + UL)']= data['Total DL (Bytes)'] + data["Total UL (Bytes)"]


# # The total download (DL) and upload (UL) data

# In[33]:


data['Total Data (DL + UL)']= data['Total DL (Bytes)'] + data["Total UL (Bytes)"]
## Aggregate the total download (DL) and upload (UL) data per user
total_dl_ul = data.groupby(['IMSI', 'MSISDN/Number']).agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Total Data (DL + UL)': 'sum'
}).reset_index()
    
total_dl_ul.columns = ['IMSI', 'MSISDN/Number', 'Total Download Data', 'Total Upload Data' ,'Total Data (DL + UL)']
total_dl_ul.head()


# # The total data volume (in Bytes) during this session for each application
# 

# In[34]:


# Aggregation of Total Social Media data
data = df.copy()
data["Total Youtube Volume(Bytes)"]= data["Youtube DL (Bytes)"] + data["Youtube UL (Bytes)"]
data["Total Google Volume(Bytes)"]= data["Google DL (Bytes)"] + data["Google UL (Bytes)"]
data["Total Email Volume(Bytes)"]= data["Email DL (Bytes)"] + data["Email UL (Bytes)"]
data["Total Social Media Volume(Bytes)"]= data["Social Media DL (Bytes)"] + data["Social Media UL (Bytes)"]
data["Total Netflix Volume(Bytes)"]= data["Netflix DL (Bytes)"] + data["Netflix UL (Bytes)"]
data["Total Gaming Volume(Bytes)"]= data["Gaming DL (Bytes)"] + data["Gaming UL (Bytes)"]
data["Total Other Volume(Bytes)"]= data["Other DL (Bytes)"] + data["Other UL (Bytes)"]


# In[35]:


## Aggregate total data volume (in Bytes) during this session for each application
apps_data_volume = data.groupby(['Dur. (ms)', 'MSISDN/Number']).agg({
    'Total Youtube Volume(Bytes)': 'sum',
    'Total Google Volume(Bytes)': 'sum',
    'Total Email Volume(Bytes)': 'sum',
    'Total Social Media Volume(Bytes)': 'sum',
    'Total Netflix Volume(Bytes)': 'sum',
    'Total Gaming Volume(Bytes)': 'sum',
    'Total Other Volume(Bytes)': 'sum',
}).reset_index()

apps_data_volume.head()


# #          Task 1.2

# In[36]:


## Convert the data type of 'IMEI', 'IMSI', 'MSISDN/Number' to str/ object datatype
new_telcom = df.copy()
columns_to_convert = ['IMEI', 'IMSI', 'MSISDN/Number']

for column in columns_to_convert:
    new_telcom[column] = new_telcom[column].astype(str)


# In[37]:


new_telcom.info()


# In[38]:


new_telcom.head()


# # Describe all relevant variables and associated Data types(slide)

# In[39]:


## Describe all relevant variables and their associated data types
description = new_telcom.describe()
#print(description[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].head())
description


# # Analyze the basic metrics (mean, median,etc)in the Dataset(explain)and their importance for the global objective

# In[40]:


# Calculate basic metrics for the numerical columns in the dataset
basic_metrics = new_telcom.describe().transpose()

# Selecting only the relevant metrics to display
selected_metrics = basic_metrics[['mean', 'std', 'min', '50%', 'max']]
selected_metrics.rename(columns={'50%': 'median'}, inplace=True)

selected_metrics#.head()


# # Conduct a Non-Graphical Univariate Analysis by computing dispersion parameters for each quantitative variable and provide useful interpretation

# In[41]:


# Conducting a Non-Graphical Univariate Analysis by computing dispersion parameters
def compute_dispersion_params(new_telcom):
    ## Selecting only the numeric columns for dispersion analysis
    numeric_columns = new_telcom.select_dtypes(include=['float64', 'int64']).columns

    ## Calculating dispersion parameters: Standard Deviation, Variance, Range, Interquartile Range
    dispersion_params = new_telcom[numeric_columns].agg(['std', 'var', 'min', 'max'])
    dispersion_params.loc['range'] = dispersion_params.loc['max'] - dispersion_params.loc['min']
    dispersion_params.loc['iqr'] = new_telcom[numeric_columns].quantile(0.75) - new_telcom[numeric_columns].quantile(0.25)

    ## Displaying the head of the dispersion parameters dataframe
    return dispersion_params.head()

## Calling the function with the telcom dataset
telcom_dispersion_params = compute_dispersion_params(new_telcom)
telcom_dispersion_params


# # Conduct a Graphical Univariate Analysis by identifying the most suitable plotting options for each variable and interpreting your findings

# In[42]:


# Set the figure size
plt.figure(figsize=(12, 6))

# Create subplots
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the histogram
sns.histplot(user_xdr_sessions["Number of xDR Sessions"], ax=ax)

# Show the plot
plt.show()


# In[43]:


# Set the figure size
plt.figure(figsize=(12, 6))

# Create subplots
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the histogram
sns.histplot(user_session_duration["Total Session Duration (ms)"], ax=ax)

# Show the plot
plt.show()


# In[44]:


# Set the figure size
plt.figure(figsize=(10, 5))

# Create subplots
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the histogram
sns.histplot(total_dl_ul["Total Download Data"], ax=ax)

# Show the plot
plt.show()


# In[45]:


# Set the figure size
plt.figure(figsize=(10, 5))

# Create subplots
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the histogram
sns.histplot(total_dl_ul["Total Upload Data"], ax=ax)

# Show the plot
plt.show()


# In[46]:


# Set the figure size
plt.figure(figsize=(10, 5))

# Create subplots
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the histogram
sns.histplot(total_dl_ul["Total Data (DL + UL)"], ax=ax)

# Show the plot
plt.show()


# In[47]:


apps_data_volume.head()


# In[48]:


data = df.copy()
data["Total Youtube Volume(Bytes)"]= data["Youtube DL (Bytes)"] + data["Youtube UL (Bytes)"]

# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Youtube Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Youtube Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Youtube Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Youtube Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Youtube Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Youtube Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Youtube Volume(Bytes).jpeg')


# In[49]:


data = df.copy()
data["Total Google Volume(Bytes)"]= data["Google DL (Bytes)"] + data["Google UL (Bytes)"]

# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Google Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Google Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Google Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Google Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Google Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Google Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Google Volume(Bytes).jpeg')


# In[50]:


data = df.copy()
data["Total Email Volume(Bytes)"]= data["Email DL (Bytes)"] + data["Email UL (Bytes)"]

# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Email Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Email Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Email Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Email Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Email Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Email Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Email Volume(Bytes).jpeg')


# In[51]:


data = df.copy()
data["Total Social Media Volume(Bytes)"]= data["Youtube DL (Bytes)"] + data["Youtube UL (Bytes)"]

# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Social Media Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Social Media Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Social Media Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Social Media Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Social Media Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Social Media Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Social Media Volume(Bytes)).jpeg')


# In[52]:


# Save the figure
fig.savefig('Total Netflix Volume(Bytdata = telcom.copy()
data["Total Netflix Volume(Bytes)"]= data["Netflix DL (Bytes)"] + data["Netflix UL (Bytes)"]


# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Netflix Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Netflix Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Netflix Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Netflix Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Netflix Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Netflix Volume(Bytes) in Bytes')
es).jpeg')


# In[53]:


data = df.copy()
data["Total Gaming Volume(Bytes)"]= data["Gaming DL (Bytes)"] + data["Gaming UL (Bytes)"]


# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(28, 10))

# Plot histogram
ax[0].hist(data["Total Gaming Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Gaming Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Gaming Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Gaming Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Gaming Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Gaming Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Gaming Volume(Bytes).jpeg')


# In[54]:


data = df.copy()
data["Total Other Volume(Bytes)"]= data["Other DL (Bytes)"] + data["Other UL (Bytes)"]

# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 3, figsize=(25, 10))

# Plot histogram
ax[0].hist(data["Total Other Volume(Bytes)"])
ax[0].set_title('Histogram showing the Total Data Usage on Total Other Volume(Bytes) in Bytes')

# Plot boxplot
sns.boxplot(data["Total Other Volume(Bytes)"], ax=ax[1])
ax[1].set_title('Box Plot showing the Total Data Usage on Total Other Volume(Bytes) in Bytes')

# Plot distribution plot
sns.distplot(data["Total Other Volume(Bytes)"], ax=ax[2])
ax[2].set_title('Distribution Plot showing the Total Data Usage on Total Other Volume(Bytes) in Bytes')

# Save the figure
fig.savefig('Total Other Volume(Bytes).jpeg')


# # Bivariate Analysis-explore the relationship between each application and the total DL+UL

# In[55]:


apps_data_volume.head()


# In[56]:


new_telcom.head()


# In[57]:


# Aggregation of Total Social Media data
data = new_telcom.copy()
data["Total Youtube Volume(Bytes)"]= data["Youtube DL (Bytes)"] + data["Youtube UL (Bytes)"]
data["Total Google Volume(Bytes)"]= data["Google DL (Bytes)"] + data["Google UL (Bytes)"]
data["Total Email Volume(Bytes)"]= data["Email DL (Bytes)"] + data["Email UL (Bytes)"]
data["Total Social Media Volume(Bytes)"]= data["Social Media DL (Bytes)"] + data["Social Media UL (Bytes)"]
data["Total Netflix Volume(Bytes)"]= data["Netflix DL (Bytes)"] + data["Netflix UL (Bytes)"]
data["Total Gaming Volume(Bytes)"]= data["Gaming DL (Bytes)"] + data["Gaming UL (Bytes)"]
data["Total Other Volume(Bytes)"]= data["Other DL (Bytes)"] + data["Other UL (Bytes)"]

## Calculate the total uploads and downloads
data['Total Data (DL + UL)'] = data['Total DL (Bytes)'] + data["Total UL (Bytes)"]

# List of application columns to analyze
application_columns = ["Total Youtube Volume(Bytes)", "Total Google Volume(Bytes)",
                      "Total Email Volume(Bytes)", "Total Social Media Volume(Bytes)",
                      "Total Netflix Volume(Bytes)", "Total Gaming Volume(Bytes)", "Total Other Volume(Bytes)" ]

# Create scatter plots to visualize the relationship between each application's total data and the total DL+UL data
for col in application_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[col], y=data['Total Data (DL + UL)'])
    plt.title(f'Total Data (DL + UL) vs {col}')
    plt.xlabel(col)
    plt.show()


# # Variable transforms-segment the users into the top five decile classes based on the total 

# In[58]:


new_telcom.head()


# In[59]:


data = new_telcom.copy()
data['Total Data (DL + UL)'] = data['Total DL (Bytes)'] + data["Total UL (Bytes)"]

# Step 1: Calculate Total Duration per User
data['Dur. (ms)'] = data.groupby('MSISDN/Number')['Dur. (ms)'].transform('sum')

# Step 2: Decile Classes
data['decile_class'] = pd.qcut(data['Dur. (ms)'], q=10, labels=False)

# Step 3: Segment Users into Top Five Decile Classes
top_five_deciles = data[data['decile_class'] >= 5]

# Step 4: Compute Total Data per Decile Class
total_data_per_decile = top_five_deciles.groupby('decile_class')[['Total Data (DL + UL)']].sum().reset_index()

total_data_per_decile


# # Correlation Analysis- compute a correlation matrix for the following variables and interpret your findings:Social Media data,Google data,Email data,Youtube data, Netflix data, Gaming data, and Order data

# In[60]:


apps_data_volume.head()


# In[61]:


app_columns = ['Total Youtube Volume(Bytes)', 'Total Google Volume(Bytes)', 'Total Email Volume(Bytes)', 'Total Social Media Volume(Bytes)'
               , 'Total Netflix Volume(Bytes)', 'Total Gaming Volume(Bytes)', 'Total Other Volume(Bytes)']

corr = apps_data_volume[app_columns].corr()
corr


# In[62]:


# correlation heatmap plot
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="BuPu")
plt.show()


# # Dimensionality Reduction-perform a principal component analysis to reduce the dimensions of your data and  provide a useful interpretation of the results (Provide your interpretation in four(4)bullet points maximum)

# # Numerical Variables

# In[63]:


new_telcom.info()


# In[64]:


## Get a list of numerical variables
numerical_features = [feature for feature in new_telcom.columns if new_telcom[feature].dtypes != 'O']

## Print the number of numerical variables
print('Number of numerical variables:', len(numerical_features))

## Visualize the numerical variables
new_telcom[numerical_features].head()


# In[65]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[70]:


numerical_features = new_telcom.select_dtypes(include=['float64'])


# In[71]:


new_telcom[numerical_features].describe()


# In[ ]:


# Standardizing the data before applying PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_telcom[numerical_features])

# Applying PCA
pca = PCA(n_components=2) # reduce to 2 dimensions for visualization purposes
principal_components = pca.fit_transform(scaled_data)

# Creating a DataFrame with the principal components
pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Displaying the DataFrame head and the explained variance

print('Explained variance by component: ', explained_variance)
pca_df.head()


# In[ ]:




