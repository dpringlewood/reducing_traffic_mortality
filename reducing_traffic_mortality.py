"""
This is a project to find patterns in road traffic accidents
so that we know where to focus our attention to bring down
the accident rate.

Can we group states into similar profiles.
"""
# Handle all the imports at the top
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# set pandas options for pycharm output
pd.set_option('display.max_columns', 10)

#Import and inspect the data we have
car_accidents = pd.read_csv(r'./road-accidents.csv', sep='|', comment='#')
miles_driven = pd.read_csv(r'./miles-driven.csv', sep='|')

print(car_accidents.head())
print(car_accidents.info())
print(car_accidents.describe())

print(miles_driven.head())
print(miles_driven.info())
print(miles_driven.describe())

"""
The data is very clean, with 50 states. That makes sense. There are no null
values by the looks of it. The number of deaths per billion miles is given
by the drvr_fatl_col_bmiles column.

By doing some visual-EDA we can get a better idea of the data, and highlight
any obvious relationships within the data.
"""

# Lets do some visual-EDA
sns.pairplot(car_accidents)
plt.show()

"""
We can see from this pairplot that there are relationships betweem the target
variable and the predictors. We can see that very few fatal accidents happen
when there is no or little amounts of alcohol present. Additionally there seems
to be some collinearity between speed and alcohol, suggesting people speed more
after drinking. 

Lets explore this more with the correlation matrix between variables.
"""

print(car_accidents.corr())

"""
We can see a correlation of 0.2 between alcohol consumption and traffic deaths.
We see an even stronger correlation of 0.28 between speeding and alcohol.

Lets see how these variables work together in predicting traffic deaths, this 
can be done using a linear regression model.
"""

features = car_accidents.drop(['drvr_fatl_col_bmiles', 'state'], axis=1)
target = car_accidents.drvr_fatl_col_bmiles

linear = LinearRegression()
linear.fit(features, target)

linear_coeff = linear.coef_
coeff_names = features.columns.values

print('\n')
for i in range(len(linear_coeff)):
    print(coeff_names[i],':', linear_coeff[i])

"""
From this data we can see that the leading cause of road traffic deaths is 
alcohol. One method to bring down road deaths would be to target states with
a high alcohol consumption.

We also know that alcohol consumption has an effect on the other predictors in 
the model.

We can scale the data and preform Principle Component Analysis (PCA) on our data.
PCA highlights which components make up the most variance in the target data."""

# Scale the features data and fit it to the PCA
scaler = StandardScaler()
pca = PCA()

scaled_features = scaler.fit_transform(features)
pca.fit(scaled_features)

# Lets visually show the proportion of variance explained by each component

plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
plt.xlabel('Principle Component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])
plt.show()

"""
We can see from the chart that the first 2 components explain a high degree 
of traffic accidents. We can chart these to see if we can find any patterns
in the data.
"""

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

p_comp1 = principal_components[:,0]
p_comp2 = principal_components[:,1]

sns.scatterplot(p_comp1, p_comp2)
plt.show()

"""
In this scatterplot it is hard to see any clear patterns or groups.
We can cluster this data using KMeans and use a scree plot to see how
many groups there should be.

We want clusters that are close together and tightly packed. This can be
measured by the inertia. (Distance from the center of the culster). The 
lower the inertia, the better. Creating too many clusters would mean over-fitting
and more computing for no real gain in predictability.
"""

# Lets explore clustering with up to 10 clusters and select the best KMeans Clusters

ks = range(1, 10)
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=8)
    km.fit(scaled_features)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, marker='D')
plt.show()

"""
There is no clear 'elbow' in our plot which would usually indicate the desired
amount of clustering. 3 seems to be a good choice. 
"""

km = KMeans(n_clusters=3, random_state=8)
km.fit(scaled_features)

plt.scatter(p_comp1, p_comp2, c=km.labels_)
plt.show()

# Lets add the cluster labels back into our car_accidents dataframe.

car_accidents['cluster'] = km.labels_
print(car_accidents.head())

melt_car = pd.melt(car_accidents,
                   value_vars=['perc_fatl_speed', 'perc_fatl_alcohol','perc_fatl_1st_time'],
                   value_name='percent',
                   id_vars='cluster',
                   var_name='measurement')

print(melt_car.head())