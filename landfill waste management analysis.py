#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML as html_print


# In[2]:


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[3]:


df = pd.read_excel('waste dataset.xlsx', sheet_name="LMOP Database")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


#drop missing columns
df.dropna(subset=['LFG Flared (mmscfd)','Waste in Place Year'], inplace=True)


# In[8]:


df.describe()


# In[9]:


#check data near maximum number of 'LFG Collected'
df[df['LFG Collected (mmscfd)'] > 20]


# In[10]:


df2 = df.copy()


# In[11]:


# narrow down the size of columns 
df=df[['Landfill Name', 'State', 'City', 'County', 'Ownership Type',
       'Waste in Place (tons)', 'Waste in Place Year',
       'LFG Collection System In Place?', 'LFG Collected (mmscfd)',
       'LFG Flared (mmscfd)']]


# In[12]:


# check if there are duplicate data
df.duplicated().sum()


# In[13]:


df[df.duplicated(keep=False)]


# In[14]:


df.info()


# In[15]:


# drop all duplicates
df.drop_duplicates(inplace=True)


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


#exclude all data with LFG Flared = 0
df = df[df['LFG Flared (mmscfd)'] > 0]


# In[19]:


df.info()


# # Overall Data Analysis

# In[20]:


# Pairplot to visualize relationships between multiple variables
sns.pairplot(df[['Waste in Place (tons)', 'LFG Collected (mmscfd)', 'LFG Flared (mmscfd)']])
plt.show()


# ### **Analysis of Pairplot from Correlation Analysis**
# 
# This pairplot visualizes the relationships between three key variables related to landfills:
# 1. **Waste in Place (tons)** – The amount of waste deposited in the landfill.
# 2. **LFG Collected (mmscfd)** – The amount of landfill gas collected, measured in million standard cubic feet per day.
# 3. **LFG Flared (mmscfd)** – The amount of landfill gas flared, meaning not utilized for energy production but released or burned.
# 
# Each scatter plot shows the relationship between two variables, while the diagonal plots show the distribution (histograms) of each variable.
# 
# 
# ### **Summary of Findings:**
# 
# 1. **Waste in Place vs. LFG Collected**: There is a **strong positive correlation** between the amount of waste and the amount of gas collected. Larger landfills generally collect more gas, as expected. However, there are a few exceptions where landfills with similar waste levels collect significantly different amounts of gas, indicating potential inefficiencies.
# 
# 2. **Waste in Place vs. LFG Flared**: The correlation between waste and flaring is **weaker**, suggesting that the amount of waste does not directly dictate how much gas is flared. This could be due to varying gas collection systems or different levels of operational efficiency at landfills.
# 
# 3. **LFG Collected vs. LFG Flared**: There is a **positive correlation** between gas collected and gas flared. However, flaring should ideally be minimized, meaning landfills should aim to collect as much gas as possible while flaring less. The outliers suggest some landfills still flare significant amounts of gas despite high collection rates.
# 
# ### **Recommendations**:
# 
# - **Improvement in Gas Collection Systems**: Landfills with high flaring relative to their waste or gas collection need to be examined. They should focus on improving gas capture efficiency to reduce emissions and make better use of available gas for energy production.
#   
# - **Targeting Outliers**: The outliers in the dataset (landfills that collect or flare much more or less gas than expected) should be investigated for potential issues, such as operational inefficiencies or underutilized capacity.
# 
# - **Benchmark Efficient Landfills**: The most efficient landfills (those that collect a lot of gas relative to their waste and flare less) could serve as models for other landfills looking to improve their performance.

# In[21]:


# Calculate correlation matrix
corr_matrix = df[['Waste in Place (tons)', 'LFG Collected (mmscfd)', 'LFG Flared (mmscfd)']].corr()

# Visualize using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[22]:


# Correlation between Waste in Place and LFG Collected
correlation_waste_lfg = df['Waste in Place (tons)'].corr(df['LFG Collected (mmscfd)'])
print(f"Correlation between Waste in Place and LFG Collected: {correlation_waste_lfg:.2f}")

# Scatter plot visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Waste in Place (tons)', y='LFG Collected (mmscfd)', data=df)
plt.title('Waste in Place vs LFG Collected')
plt.xlabel('Waste in Place (tons)')
plt.ylabel('LFG Collected (mmscfd)')
plt.grid(True)
plt.show()


# In[23]:


# Density plot for LFG Collected
plt.figure(figsize=(10, 6))
sns.kdeplot(df['LFG Collected (mmscfd)'], shade=True, color='b', label='LFG Collected')
sns.kdeplot(df['LFG Flared (mmscfd)'], shade=True, color='r', label='LFG Flared')
plt.title('Density Plot: LFG Collected and Flared')
plt.xlabel('LFG (mmscfd)')
plt.ylabel('Density')
plt.legend()
plt.show()


# ### **Analysis of the Density Plot: LFG Collected vs LFG Flared**
# 
# This **density plot** compares the distribution of two key variables:
# 1. **LFG Collected (blue line)**: The amount of landfill gas (LFG) that is collected and potentially utilized (e.g., for energy generation).
# 2. **LFG Flared (red line)**: The amount of LFG that is flared (burned or released) instead of being collected.
# 
# The x-axis represents the amount of **LFG** (in millions of standard cubic feet per day, mmscfd), and the y-axis shows the **density**, which indicates the relative frequency of different LFG values in the dataset.
# 
# ### **Key Observations**:
# 
# #### 1. **Peak and Distribution Shape**:
#    - The **red peak (LFG Flared)** is sharp and concentrated around **0 to 2 mmscfd**, indicating that most landfills flare **very small amounts of LFG**.
#    - The **blue peak (LFG Collected)** is slightly more spread out but still concentrated between **0 and 5 mmscfd**.
#    - This suggests that the majority of landfills collect a modest amount of gas (up to 5 mmscfd) and flare even less.
# 
# #### 2. **Flaring vs. Collection**:
#    - **Higher Peak for LFG Flared**: The peak for LFG flared is higher than the peak for LFG collected, which suggests that, for a number of landfills, **flaring occurs more frequently than gas collection at smaller volumes**.
#    - **Less Flaring at Higher LFG Levels**: As the amount of gas increases (beyond 5 mmscfd), flaring decreases sharply, and collection becomes more common. This implies that landfills with larger LFG volumes tend to collect more gas and flare less.
# 
# #### 3. **Long Tail for LFG Collected**:
#    - The **blue line extends** further to the right, meaning that some landfills collect significantly more gas (up to **20+ mmscfd**), although this is rare (as indicated by the density being lower).
#    - The fact that a few landfills collect so much gas could indicate **high-performing landfills** with advanced gas collection systems in place.
# 
# ### **Summary and Insights**:
# 
# 1. **Most Landfills Flare Little Gas**:
#    - The majority of landfills flare small amounts of LFG (between **0 and 2 mmscfd**). This could indicate that they are either efficient in collecting gas, or their gas production is generally low.
#   
# 2. **Larger Landfills Collect More Gas**:
#    - Landfills that collect more LFG (over 5 mmscfd) tend to flare very little gas, meaning they have **efficient gas collection systems**. These landfills should serve as models for improving gas collection in other sites.
# 
# 3. **Potential for Improvement in Smaller Landfills**:
#    - For landfills producing lower amounts of gas (below 5 mmscfd), there appears to be more flaring. These landfills might benefit from **better gas capture technologies** to reduce flaring and increase the utilization of LFG.
# 
# 4. **Environmental Impact**:
#    - Since flaring releases methane (a potent greenhouse gas), landfills that flare larger amounts of LFG (even at low production rates) could be **environmental risks**. This underscores the importance of targeting smaller landfills for gas collection improvements.
# 
# 
# ### Conclusion:
# This density plot shows that **most landfills flare small amounts of gas**, but **larger landfills collect significantly more gas and flare less**. There is a clear opportunity to improve gas collection at landfills that produce lower amounts of LFG to reduce environmental impact and enhance energy production.

# ## City-Based Waste and LFG Analysis
# - City Performance: Analyze how different cities manage their landfills in terms of Waste in Place and LFG collection.

# In[24]:


# Average Waste in Place and LFG Collection by City
avg_waste_by_city = df.groupby('City')['Waste in Place (tons)'].mean().sort_values(ascending=False)
avg_lfg_collected_by_city = df.groupby('City')['LFG Collected (mmscfd)'].mean().sort_values(ascending=False)

print("Average Waste in Place by City:\n", avg_waste_by_city)

print("Average LFG Collected by City:\n", avg_lfg_collected_by_city)


# In[25]:


# Visualization for Cities
top_10_waste_cities = avg_waste_by_city.nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_waste_cities.values, y=top_10_waste_cities.index, orient='h')
plt.title('Top 10 Cities by Waste in Place')
plt.ylabel('Waste in Place (tons)')
plt.xlabel('City')
plt.show()

top_10_lfg_cities = avg_lfg_collected_by_city.nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_lfg_cities.values, y=top_10_lfg_cities.index, orient='h')
plt.title('Top 10 Cities by LFG Collected')
plt.ylabel('LFG Collected (mmscfd)')
plt.xlabel('City')
plt.show()


# ## State-Based Analysis

# In[26]:


# Calculate average LFG collected and flared by state
avg_lfg_collected_by_state = df.groupby('State')['LFG Collected (mmscfd)'].mean().sort_values(ascending=False)
avg_lfg_flared_by_state = df.groupby('State')['LFG Flared (mmscfd)'].mean().sort_values(ascending=False)

print("Average Top 10 LFG Collected by State:\n", avg_lfg_collected_by_state.nlargest(n=10))
print("Average Top 10 LFG Flared by State:\n", avg_lfg_flared_by_state.nlargest(n=10))


# In[27]:


# Horizontal Bar plot for Average LFG Collected by State
plt.figure(figsize=(16, 12))
sns.barplot(x=avg_lfg_collected_by_state.values, y=avg_lfg_collected_by_state.index, orient='h')
plt.title('Average LFG Collected by State')
plt.xlabel('LFG Collected (mmscfd)')
plt.ylabel('State')
plt.show()

# Horizontal Bar plot for Average LFG Flared by State
plt.figure(figsize=(16, 12))
sns.barplot(x=avg_lfg_flared_by_state.values, y=avg_lfg_flared_by_state.index, orient='h')
plt.title('Average LFG Flared by State')
plt.xlabel('LFG Flared (mmscfd)')
plt.ylabel('State')
plt.show()


# In[29]:


# Calculate LFG Utilization Efficiency
df['LFG Utilization Efficiency (%)'] = ((df['LFG Collected (mmscfd)'] - df['LFG Flared (mmscfd)']) / df['LFG Collected (mmscfd)']) * 100

# Average LFG Utilization Efficiency by State
lfg_efficiency_by_state = df.groupby('State')['LFG Utilization Efficiency (%)'].mean().sort_values(ascending=False)
print("TOP 10 LFG Efficiency(%) by State:\n", lfg_efficiency_by_state.nlargest(n=10))



# ### **Analysis of the LFG Utilization Efficiency (%) by State**
# 
# This ranking shows the **Top 10 States** by **LFG Utilization Efficiency**  
# 
# ### **Key Observations:**
# 
# #### **Top 3 States with Very High Efficiency (Above 90%)**:
# 
# 1. **New Hampshire (NH)** – 97.20%
# 2. **Iowa (IA)** – 96.90%
# 3. **Mississippi (MS)** – 94.06%
# 
# - **New Hampshire (NH)** leads the revised list with a **97.20% efficiency**, meaning that nearly all of the landfill gas (LFG) collected is being utilized for energy production or other purposes, with minimal flaring.
# - **Iowa (IA)** and **Mississippi (MS)** also have **LFG Utilization Efficiencies above 94%**, indicating highly efficient gas collection and utilization systems. These states are making excellent use of their collected LFG and contribute minimally to environmental issues caused by methane flaring.
#   
# - **Implications**: States like NH, IA, and MS should be considered **role models** for effective gas capture and utilization. They demonstrate that with the right infrastructure and practices, it is possible to achieve very high utilization rates of LFG, reducing methane emissions and enhancing sustainability.
# 
# #### **Moderately High Efficiency States (80% to 90%)**:
# 
# 4. **Rhode Island (RI)** – 92.50%
# 5. **North Dakota (ND)** – 85.64%
# 6. **Kansas (KS)** – 83.77%
# 7. **Louisiana (LA)** – 79.84%
# 
# - **Rhode Island (RI)** stands out with an **LFG efficiency of 92.50%**, continuing the trend of high performance in LFG utilization, while **North Dakota (ND)** and **Kansas (KS)** maintain strong performances with **85.64% and 83.77% efficiencies**, respectively.
# - **Louisiana (LA)**, while still fairly efficient, sees a slight drop to **79.84%**, indicating that there is a small but notable percentage of LFG being flared rather than utilized.
# 
# - **Implications**: States in this efficiency range are still performing well, but there is **room for improvement**, particularly in **Louisiana**. Optimizing LFG collection systems could help these states move closer to 90% efficiency or higher, which would reduce methane flaring and contribute to environmental sustainability.
# 
# #### **Lower Efficiency States (70% to 80%)**:
# 
# 8. **Delaware (DE)** – 78.94%
# 9. **Illinois (IL)** – 78.81%
# 10. **New York (NY)** – 77.82%
# 
# - The bottom three states in this ranking—**Delaware (DE)**, **Illinois (IL)**, and **New York (NY)**—have **LFG Utilization Efficiencies between 77% and 79%**. While these are still relatively good performances, they are notably lower than the top-ranking states.
# - This means that around **20% to 23%** of the LFG collected in these states is being flared instead of being utilized. This is a significant opportunity for improvement in terms of reducing methane emissions and improving the energy potential of collected LFG.
# 
# - **Implications**: These states might need to **invest in better LFG collection infrastructure** or examine current operational practices to reduce flaring. Implementing more robust gas utilization systems would help them move up the rankings and increase their overall environmental performance.
# 
# 
# ### **Conclusion**:
# 
# - The top-performing states, such as **New Hampshire (97.2%)** and **Iowa (96.9%)**, are leading the way in maximizing the utilization of collected landfill gas, with minimal flaring.
# - There is **room for improvement** in states like **Louisiana**, **Delaware**, **Illinois**, and **New York**, where a significant percentage of collected gas is being flared. By focusing on **better gas collection technologies** and **operational efficiencies**, these states can reduce their environmental impact and increase the economic benefits of LFG utilization.
# 

# In[55]:


# Visualization

plt.figure(figsize=(10, 6))
sns.barplot(x=lfg_efficiency_by_state.nlargest(n=10).index, y=lfg_efficiency_by_state.nlargest(n=10).values)
plt.title('LFG Utilization Efficiency by State')
plt.ylabel('LFG Utilization Efficiency (%)')
plt.xlabel('State')
plt.xticks()
plt.show()


# ## Ownership Type Analysis

# In[31]:


# Compare average waste in place by ownership type
avg_waste_by_ownership = df.groupby('Ownership Type')['Waste in Place (tons)'].mean()
print("Average Waste in Place by Ownership Type:\n", avg_waste_by_ownership)

# Compare LFG efficiency by ownership type
lfg_efficiency_by_ownership = df.groupby('Ownership Type')['LFG Utilization Efficiency (%)'].mean()
print("LFG Efficiency by Ownership Type:\n", lfg_efficiency_by_ownership)


# In[32]:


# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_waste_by_ownership.index, y=avg_waste_by_ownership.values)
plt.title('Average Waste in Place by Ownership Type')
plt.ylabel('Waste in Place (tons)')
plt.xlabel('Ownership Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=lfg_efficiency_by_ownership.index, y=lfg_efficiency_by_ownership.values)
plt.title('LFG Utilization Efficiency by Ownership Type')
plt.ylabel('LFG Utilization Efficiency (%)')
plt.xlabel('Ownership Type')
plt.show()


# # Landfill Comparison Analysis

# In[64]:


# Ranking landfills by Waste in Place and LFG Collected
df['Waste Rank'] = df['Waste in Place (tons)'].rank(ascending=False)
df['LFG Collected Rank'] = df['LFG Collected (mmscfd)'].rank(ascending=False)

# Display top 10 landfills by Waste Rank and LFG Collected Rank
top_10_landfills_by_waste = df.nsmallest(10, 'Waste Rank')[['Landfill Name','State', 'Waste in Place (tons)']]
top_10_landfills_by_lfg = df.nsmallest(10, 'LFG Collected Rank')[['Landfill Name', 'State', 'LFG Collected (mmscfd)']]

print("Top 10 Landfills by Waste in Place:\n", top_10_landfills_by_waste)
print("\nTop 10 Landfills by LFG Collected:\n", top_10_landfills_by_lfg)


# In[65]:


# Set plot style for better visualization
sns.set(style="whitegrid")

# Create figure and axis objects for subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

# Plot 1: Top 10 Landfills by Waste in Place
sns.barplot(
    ax=axes[0], 
    y='Landfill Name', 
    x='Waste in Place (tons)', 
    data=top_10_landfills_by_waste,
    palette='Blues_r'
)
axes[0].set_title('Top 10 Landfills by Waste in Place')
axes[0].set_xlabel('Waste in Place (tons)')
axes[0].set_ylabel('Landfill Name')

# Plot 2: Top 10 Landfills by LFG Collected
sns.barplot(
    ax=axes[1], 
    y='Landfill Name', 
    x='LFG Collected (mmscfd)', 
    data=top_10_landfills_by_lfg,
    palette='Greens_r'
)
axes[1].set_title('Top 10 Landfills by LFG Collected')
axes[1].set_xlabel('LFG Collected (mmscfd)')
axes[1].set_ylabel('Landfill Name')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# ### **Analysis of Top 10 Landfills by Waste in Place and LFG Collected**
# 
# 
# ### **1. Comparison Between Waste in Place and LFG Collected**
# 
# - **Key Overlap**: 
#    - **Sunshine Canyon Landfill (CA)** and **Puente Hills LF (CA)** rank highly in both categories:
#      - **Sunshine Canyon**: Ranked **8th** by waste in place but **1st** by LFG collected.
#      - **Puente Hills**: Ranked **1st** by waste in place and **2nd** by LFG collected.
#    - This indicates that these two landfills are not only large in terms of the amount of waste they manage but are also highly efficient at collecting landfill gas. Both landfills are in **California**, highlighting effective waste management and gas capture systems in this state.
# 
# - **High Waste, Moderate LFG Collection**:
#    - **McCarty Road LF (TX)** and **Apex Regional LF (NV)** rank **2nd** and **4th** by waste in place, respectively, but **do not appear in the top 10 for LFG collected**.
#      - This suggests that although these landfills manage large amounts of waste, their LFG collection systems may not be as efficient, or the waste might not be producing as much gas due to factors such as waste composition or landfill age.
#      - These landfills could be targeted for improvements in gas capture technology to increase LFG utilization and reduce environmental impacts.
# 
# - **Efficient LFG Collection Despite Lower Waste**:
#    - **Keystone Sanitary Landfill (PA)**, **Seneca Meadows SWMF (NY)**, and **Central LF (RI)** appear in the top 10 for LFG collected but are **not present** in the top 10 for waste in place. 
#      - These landfills are relatively smaller compared to the largest landfills but are still highly efficient at capturing LFG. This indicates that waste size alone does not determine LFG collection efficiency; other factors such as gas capture infrastructure, landfill age, and waste decomposition rates can significantly affect LFG collection.
# 
# ### **2. Insights from the Rankings**
# 
# #### **High Waste and High LFG Collection**:
#    - **Sunshine Canyon Landfill** and **Puente Hills LF** both stand out as landfills that are highly effective in both waste management and LFG collection. These landfills likely have advanced gas collection systems, and their large size means they can capture more gas due to the decomposition of a higher volume of waste.
#    - These landfills can serve as **benchmark examples** for others. Their best practices can be studied and potentially implemented at other landfills to improve both waste management and gas collection efficiency.
# 
# #### **Large Waste Landfills with Lower LFG Collection**:
#    - **McCarty Road LF** and **Apex Regional LF** 
#      - Potential inefficiencies in gas collection systems.
#      - The composition of the waste or the landfill’s age might be contributing to lower gas generation.
#      - There is room for improvement in optimizing LFG collection systems at these large landfills to reduce flaring and environmental impact.
# 
# #### **Smaller Landfills with High LFG Collection**:
#    - **Keystone Sanitary Landfill (PA)**, **Seneca Meadows SWMF (NY)**, and **Central LF (RI)** highlight that **size is not the only factor** driving LFG collection. These landfills collect a significant amount of LFG despite being smaller than some others in terms of waste in place.
#    - These landfills might have more efficient gas capture technologies or handle types of waste that produce more methane, resulting in higher LFG collection rates.
#    - These landfills can serve as examples of how even smaller landfills can excel in gas capture, which has environmental and economic benefits.
# 
# ### **3. State-Level Observations**
# 
# - **California Dominates**:
#    - **California** has multiple entries in both rankings: **Sunshine Canyon**, **Puente Hills**, **Monarch Hill**, **Olinda Alpha**, and **Frank R. Bowerman**.
#    - This suggests that California has a strong focus on **waste management and LFG collection**, likely due to stricter environmental regulations and better-developed gas capture systems. The state appears to prioritize minimizing methane emissions and maximizing gas utilization for energy production.
# 
# - **Texas Representation**:
#    - Texas appears in the top 10 for both waste and LFG collection (though not always the same landfills). **McCarty Road LF** is notable for its large size (ranked **2nd** by waste), but it doesn't rank in the top 10 for LFG collected. On the other hand, **McCommas Bluff Landfill** is highly ranked for LFG collected (ranked **9th**), despite not being among the largest landfills.
#    - This indicates some variability in landfill performance within Texas, suggesting that there may be opportunities for optimization in LFG collection in certain large landfills.
# 
# - **Other States**:
#    - States like **Nevada**, **Colorado**, and **Florida** have large landfills by waste in place but aren't as efficient in terms of LFG collection. Improvements in infrastructure could enhance their gas capture efficiency.
# 

# # 1. Cluster Analysis of Landfills
# - Use clustering techniques such as K-means to group landfills based on similar characteristics like waste in place, LFG collected, and LFG flared.

# In[36]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Select relevant numerical features for clustering
X = df[['Waste in Place (tons)', 'LFG Collected (mmscfd)', 'LFG Flared (mmscfd)']]

# Standardize the features to ensure they're on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Waste in Place (tons)'], y=df['LFG Collected (mmscfd)'], hue=df['Cluster'], palette='Set1')
plt.title('Landfill Clustering by Waste in Place and LFG Collected')
plt.xlabel('Waste in Place (tons)')
plt.ylabel('LFG Collected (mmscfd)')
plt.show()


# ### **Analysis of Landfill Clustering by Waste in Place and LFG Collected**
# 
# This scatter plot visualizes the results of a **K-means clustering** analysis, where the landfills are grouped based on two key variables:
# 1. **Waste in Place (tons)** (x-axis): The total amount of waste deposited in the landfill.
# 2. **LFG Collected (mmscfd)** (y-axis): The amount of landfill gas (LFG) collected from the landfill, measured in million standard cubic feet per day (mmscfd).
# 
# The plot includes **three clusters**, represented by different colors:
# - **Cluster 0 (Red)**
# - **Cluster 1 (Blue)**
# - **Cluster 2 (Green)**
# 
# ### **Key Observations and Insights:**
# 
# #### **Cluster 0 (Red)** – High Waste, Moderate to High LFG Collected
# - **Characteristics**:
#   - The red points are spread across **medium to high waste in place** values, ranging from approximately **0.3e8 to 1.4e8 tons** of waste.
#   - **LFG collected** ranges from **5 to 15 mmscfd**, with a few outliers above **20 mmscfd**.
#   
# - **Insight**:
#   - Landfills in this cluster have **larger amounts of waste** and tend to collect **moderate to high levels of LFG**.
#   - These landfills are likely more efficient in gas collection compared to Cluster 1, and they could serve as examples of improved LFG management systems.
# 
# #### **Cluster 1 (Blue)** – Low Waste, Low to Moderate LFG Collected
# - **Characteristics**:
#   - The blue points cluster towards the lower left corner of the plot, where both **waste in place** and **LFG collected** values are relatively small.
#   - **Waste in Place** is generally below **0.2e8 tons**, and **LFG Collected** is mostly below **5 mmscfd**.
#   
# - **Insight**:
#   - Landfills in this cluster are **smaller in size** and collect **smaller amounts of LFG**.
#   - These could represent **less efficient landfills** or **landfills with smaller gas collection systems**. There may be opportunities to improve LFG collection in this group through better technology or operational improvements.
# 
# #### **Cluster 2 (Green)** – Medium to High Waste, High LFG Collected
# - **Characteristics**:
#   - The green points represent landfills with **medium to high waste in place**, ranging from **0.3e8 to 1.0e8 tons**, and **high LFG collected** levels, generally **above 10 mmscfd**.
#   
# - **Insight**:
#   - Landfills in this cluster are highly efficient in collecting LFG relative to their waste levels. This indicates that these landfills likely have **well-optimized gas collection systems** in place and are making the best use of the waste for energy production.
#   - These landfills could serve as **benchmark sites** for improving LFG collection efficiency across other landfills, particularly those in Cluster 1 and some from Cluster 0.
# 
# ### **Overall Trend**:
# - There is a **positive correlation** between **Waste in Place** and **LFG Collected**, meaning that as the amount of waste increases, the amount of LFG collected tends to increase as well. However, the efficiency of LFG collection varies depending on the cluster.
#   - **Smaller landfills (Cluster 1)** collect less LFG, possibly due to smaller systems or lower optimization.
#   - **Larger landfills (Cluster 2 and some from Cluster 0)** are collecting significantly more LFG, indicating higher efficiency and potential for energy production.
# 
# ### **Outliers**:
# - There are a few outliers in **Cluster 0** (red points) where the LFG collected is very high relative to waste in place, and a couple of **low-performing landfills** where large amounts of waste are associated with low LFG collection. These outliers may warrant further investigation to understand the reasons behind their unusual performance.
# 
# ### **Conclusion**:
# The clustering analysis highlights **different levels of performance** among landfills in terms of LFG collection. By targeting underperforming clusters for improvement and learning from the high performers, landfill operators can enhance LFG collection efficiency, contributing to better environmental outcomes and increased energy production from waste.

# # 2. Landfill Risk Assessment (Outlier Detection) Using Z-scores

# ### identify high-risk landfills by detecting outliers in key metrics, specifically:
# 
# - Waste in Place (tons): A large or small landfill could indicate operational challenges or opportunities.
# - LFG Flared (mmscfd): High flaring indicates an inefficient use of landfill gas (LFG), contributing to unnecessary greenhouse gas emissions and environmental risk.
# 
# ### Purpose of the Analysis:
# - Flagging High-Risk Landfills: The main purpose is to flag landfills that are at risk of operational inefficiencies or environmental concerns, especially those contributing significantly to methane flaring.
# 
# - Prioritization of Remediation: Once high-risk landfills are flagged, regulatory agencies or landfill operators can focus efforts on remediation—such as installing more efficient gas capture systems or enforcing stricter compliance measures.
# 
# - Data-Driven Decisions: Landfills that are outliers require data-driven actions. Decision-makers can focus on outliers for audits, inspections, and improvements.
# 
# 

# In[42]:


from scipy.stats import zscore

# Calculate z-scores for outlier detection
df['Waste in Place Z-Score'] = zscore(df['Waste in Place (tons)'])
df['LFG Flared Z-Score'] = zscore(df['LFG Flared (mmscfd)'])


# ### High-Risk Landfills (outliers) will be flagged based on their extreme Z-scores
# - Excessive Flaring: If a landfill has a high Z-score for LFG Flared, it could indicate inefficiency in the gas capture system, environmental compliance issues, or outdated technology. These landfills need immediate attention to reduce methane emissions.
# 
# - Very Large Landfills: If a landfill has a high Z-score for Waste in Place, it might be one of the largest landfills in the region. Large landfills may experience issues in managing LFG systems, leading to inefficiencies or non-compliance.
# 
# - Small Landfills with High Flaring: A small landfill with high flaring may indicate an issue where even smaller landfills are releasing large amounts of methane into the atmosphere. These landfills need to be investigated for improvements.

# In[43]:


# Identify high-risk landfills (those with Z-Score > 2 or < -2)
high_risk_landfills = df[(df['Waste in Place Z-Score'].abs() > 2) | (df['LFG Flared Z-Score'].abs() > 2)]



# Print high-risk landfills
print("High-Risk Landfills (Outliers):\n", high_risk_landfills[['Landfill Name', 'Waste in Place (tons)', 'LFG Flared (mmscfd)', 'Waste in Place Z-Score', 'LFG Flared Z-Score']])


# ### Interpretation of the Plot:
# - Red Points: These are the high-risk landfills identified as outliers (Z-scores above 2 or below -2).
# Blue Points: These represent landfills that are within the normal range, meaning their Waste in Place and LFG Flared values are close to the mean.

# In[44]:


# Visualization: Scatter plot highlighting outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Waste in Place (tons)', y='LFG Flared (mmscfd)', data=df, hue=(df['Waste in Place Z-Score'].abs() > 2) | (df['LFG Flared Z-Score'].abs() > 2), palette={True: 'r', False: 'b'})
plt.title('Outlier Detection: Waste in Place vs LFG Flared')
plt.xlabel('Waste in Place (tons)')
plt.ylabel('LFG Flared (mmscfd)')
plt.show()


# ### **Analysis of the Outlier Detection Scatter Plot: Waste in Place vs LFG Flared**
# 
# This scatter plot visualizes the **relationship between Waste in Place (tons)** and **LFG Flared (mmscfd)** for different landfills. The key feature of this plot is the **highlighting of outliers**, which are indicated by red points. The outliers are determined based on the **Z-scores** of both **Waste in Place** and **LFG Flared**.
# 
# #### **Key Observations:**
# 
# 1. **Outliers (Red Points)**:
#    - The **red points** represent landfills that are considered **outliers** based on their Z-scores. These landfills have values for **Waste in Place** or **LFG Flared** that are more than 2 standard deviations away from the mean, either being too high or too low.
#    - These outliers appear mostly at **higher values** of both **Waste in Place** and **LFG Flared**. For example, several landfills with waste amounts above **0.5e8 tons** and LFG flared values above **5 mmscfd** are classified as outliers.
#    - There are also a few outliers with **very high LFG flaring** (above **12 mmscfd**), even though they have lower waste in place compared to other outliers. These landfills could be considered environmentally risky, as they are flaring a disproportionate amount of gas relative to their size.
# 
# 2. **Non-Outliers (Blue Points)**:
#    - The **blue points** represent landfills that are **not considered outliers**. These landfills have values for both waste and LFG flared that are within 2 standard deviations of the mean.
#    - Most of the blue points are clustered in the lower left corner of the plot, where **Waste in Place** is below **0.3e8 tons** and **LFG flared** is generally below **2.5 mmscfd**.
#    - This indicates that the majority of landfills have moderate to low levels of both waste in place and LFG flaring. These landfills are likely operating more efficiently or producing less gas overall.
# 
# 3. **Relationship Between Waste in Place and LFG Flared**:
#    - There is a general **positive correlation** between **Waste in Place** and **LFG Flared**, meaning that landfills with more waste tend to flare more gas. However, the strength of this relationship varies among the landfills.
#    - Many of the outliers (red points) appear to deviate from this trend, suggesting that some landfills are flaring more gas than would be expected based on their waste size. These landfills may require further investigation to understand why they are flaring so much gas and whether improvements can be made to capture more of the gas.
# 
# 4. **Environmental Risk**:
#    - The landfills that are flaring large amounts of gas despite relatively smaller waste volumes (e.g., landfills with LFG flared above **10 mmscfd** but waste in place below **0.5e8 tons**) could represent **environmental risks**. These landfills may need to improve their gas collection systems to reduce flaring and minimize greenhouse gas emissions.
#    
# 5. **Landfills with High Waste but Moderate LFG Flaring**:
#    - Some landfills with **high waste in place** (above **0.8e8 tons**) are **not outliers**, indicating that they are managing their LFG collection and flaring efficiently relative to their waste size. These landfills may serve as examples of best practices for gas management.
# 
# 
# #### **Conclusion**:
# 
# This plot effectively highlights the **outliers** in terms of waste and LFG flaring, offering insights into which landfills may need improvements in gas collection systems. The outliers with **high LFG flaring** represent potential **environmental risks**, and focusing on these landfills can help reduce methane emissions and improve overall landfill efficiency.

# ## 3. Classify and Identify Landfills by Category

# ### Purpose of the Analysis
# - To classify landfills into the four categories—Efficient Landfills, Inefficient Landfills, Environmental Risk, and Smaller Landfills—we need to first define the criteria for each category based on the cluster analysis and then identify the landfills in each category.
# 
# ### Defining the Categories:
# - Efficient Landfills (High Waste, High Collection):These landfills have both large amounts of waste and high LFG collection efficiency.**Criteria: High Waste in Place (tons) and high LFG Collected (mmscfd)**.
# 
# - Inefficient Landfills (High Waste, Low Collection):These landfills have large amounts of waste but low LFG collection, meaning they're not fully utilizing their potential. **Criteria: High Waste in Place (tons) and low LFG Collected (mmscfd)**.
# 
# - Environmental Risk (Low Waste, High Flaring):These landfills have lower amounts of waste but flare a significant amount of gas, which is an environmental concern. **Criteria: Low Waste in Place (tons) and high LFG Flared (mmscfd)**.
# 
# - Smaller Landfills (Low Waste, Low Collection):These landfills are smaller in terms of waste, and they neither collect much gas nor flare much gas. **Criteria: Low Waste in Place (tons) and low LFG Collected (mmscfd)**.

# In[37]:


# Calculate thresholds using quantiles (50th percentile)
waste_threshold = df['Waste in Place (tons)'].quantile(0.5)
lfg_collected_threshold = df['LFG Collected (mmscfd)'].quantile(0.5)
lfg_flared_threshold = df['LFG Flared (mmscfd)'].quantile(0.5)


# In[38]:


# Classify landfills into categories based on the thresholds
conditions = [
    (df['Waste in Place (tons)'] > waste_threshold) & (df['LFG Collected (mmscfd)'] > lfg_collected_threshold),  # Efficient
    (df['Waste in Place (tons)'] > waste_threshold) & (df['LFG Collected (mmscfd)'] <= lfg_collected_threshold),  # Inefficient
    (df['Waste in Place (tons)'] <= waste_threshold) & (df['LFG Flared (mmscfd)'] > lfg_flared_threshold),       # Environmental Risk
    (df['Waste in Place (tons)'] <= waste_threshold) & (df['LFG Collected (mmscfd)'] <= lfg_collected_threshold)  # Smaller Landfills
]

categories = ['Efficient', 'Inefficient', 'Environmental Risk', 'Smaller Landfills']


# In[39]:


# Assign the category to each landfill
df['Category'] = np.select(conditions, categories, default='Uncategorized')


# In[40]:


# List of landfills by category, names, and states
efficient_landfills = df[df['Category'] == 'Efficient'][['Landfill Name', 'State']]
inefficient_landfills = df[df['Category'] == 'Inefficient'][['Landfill Name', 'State']]
environmental_risk_landfills = df[df['Category'] == 'Environmental Risk'][['Landfill Name', 'State']]
smaller_landfills = df[df['Category'] == 'Smaller Landfills'][['Landfill Name', 'State']]


# In[41]:


# Output results
print("Efficient Landfills:\n", efficient_landfills)
print("\nInefficient Landfills:\n", inefficient_landfills)
print("\nEnvironmental Risk Landfills:\n", environmental_risk_landfills)
print("\nSmaller Landfills:\n", smaller_landfills)


# In[67]:


# Count the number of landfills in each category
category_counts = {
    'Efficient': len(efficient_landfills),
    'Inefficient': len(inefficient_landfills),
    'Environmental Risk': len(environmental_risk_landfills),
    'Smaller': len(smaller_landfills)
}

# Labels and sizes for the pie chart
labels = list(category_counts.keys())
sizes = list(category_counts.values())
colors = ['green', 'red', 'orange', 'blue']

# Create a pie chart with a "donut hole"
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

# Draw a white circle at the center to make it a donut chart
center_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(center_circle)

# Add a title and show the plot
plt.title('Landfill Distribution by Category', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()


# ## Next Steps After Classification:
# ### Focus on Inefficient Landfills:
# - Investigate why some landfills with large amounts of waste are not collecting enough LFG. This could indicate operational inefficiencies, outdated technology, or a lack of proper gas collection systems.
# 
# ### Address Environmental Risk:
# - Environmental risk landfills need attention because they flare large amounts of gas despite their smaller size, contributing to greenhouse gas emissions. Investigating why these landfills are flaring so much gas could lead to improvements.
# 
# ### Replicate Best Practices from Efficient Landfills:
# - Efficient landfills can serve as benchmarks or models for improving practices at other landfills. Studying what makes these landfills efficient (e.g., technology, operations) could help improve other sites.
# 
# ### Consider Smaller Landfills for Future Expansion:
# - Smaller landfills, while not urgent concerns, could be monitored to see if they grow over time. If they begin accumulating more waste, they might require more robust LFG collection systems.
# 
