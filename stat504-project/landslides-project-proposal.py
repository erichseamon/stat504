
# coding: utf-8

# ## Statistics 504 - Fall 2015
# ### Class Project Proposal
# ### Erich Seamon
# ### erichs@uidaho.edu
# ### http://webpages.uidaho.edu/erichs
# #       
# 
# ### Title: " Exploring landslide likelihood across Washington using machine learning techniques"
# 
# ### Introduction
# 
# 
# The premise of this research will be to identify areas within the state of Washington that are susceptible to future landslides, based on the knowledge of past landslide events, terrain parameters, geological attributes, as well as daily meteorological data (precipitation, temperature, solar radiation, relative humidity , wind speed) (Abatzoglou, Brown, 2011).  
# 
# Previous work by Ardizzone et al (2002), Ayalew and Yamagishi (2005), Ohlmacher and DAvid (2003), all used logistical regression as a classifer method for landslide analysis. To expand upon this work, this project will evaluate four differing models to predict landslide likelihood, based on historical and future climate scenario data sets:
# 
# -logistical regression, <br> 
# -support vector analysis, <br>
# -random forest, and <br>
# -neural networks <br>
# 
# 
# Under the above approach, landslide explanatory variables will be fitted or “trained” on a training data set of an observed landslide locations, with thematic data such as morphometric attributes (slope, aspect) as well as information on (geology, landuse, ). In this context, the misclassification rate for landslide potential, as measured using the test data, will be the primary quantitative measure for evaluating the predictive power the model (Efron and Gong, 1983; Efron and Tib- shirani, 1986).
# 
# ### Data Sources
# 
# 2007 and 2008 Landslide data for the state of Washington, containing over 52,000 observations, was accessed from the Washington Department of Natural Recreation (WA DNR).  The data was provided as a downloadable .gdb file (geodatabase file).  
# 

# #### Data Transformation before Analysis
# 
# The data was imported as a csv file - that was derived from the provided geodatabase file.  This csv file contained over 52,000 observations, with latitude and longitude included as well.  The csv file was transformed into a a pandas data frame, with text based categorical fields transformed to numeric values.  From this pandas data frame, our feature columns (X) and our response variable (y).
# 
# From the 52,000+ landslide observations:
# 
# -3695 had dates. This is essential to match up with specific climate values for that day, and for the days surrounding this date.
# 
# -Of these 3695 dates, 874 had both dates, geological unit values, as well as slope and gradient. This subset of 874 points are our final observation dataset.  
# 
# 
# ### Statistical Design
# 
# For this project, a machine learning process flow will be developed that fits the aforementioned algorithms (Random Forest, SVM, Logistical Regression, Neural Networks).  Then each model will be used for prediction and learning, as we iteratively identify optimal model tuning parameters.  
# 
# After refining the model - then a series of climate variables for a set of future scenarios will be run against the model, which should provide a landslide confidence level based on future climatic outcomes.

# In[432]:

import PIL
import os,sys
import numpy as np
from PIL import Image
basewidth = 900
img = Image.open("/git/data/landslides/landslide_information_flow.jpg")
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save("/git/data/landslides/landslide_information_flow_refined.jpg")


# In[433]:

import os,sys
from PIL import Image
jpgfile = Image.open("/git/data/landslides/landslide_information_flow_refined.jpg")
jpgfile


# 
# 
# ### Feature Variables:
# 
# 
# ### Morphometric feature variables:
# 
# #### 1) Slope Shape (SLOPE_MORP)  CATEGORICAL - Planar, Convex, Concave, etc. <br>
# #### 2) Land Use (LAND_USE)  CATEGORICAL - Forestry, Road/Rail/Trail, Undistubed, Urban Development  <br>
# #### 3) Landslide Type (LANDSLIDE1)  CATEGORICAL - Debris Flows, Debris Slides and Avalanches, Shallow Undifferentiated, etc.<br>
# #### 4) Gradient (GRADIENT_D) CONTINUOUS - gradient of the landslide location, in degrees.<br> 
# 
# ### Climatic feature  and Geology from external sources:
# 
# #### 5) Geologic Unit (GEOLOGIC_U) CATEGORICAL - geologic unit.  <br>
# #### 6) min temp 
# #### 7) max temp 
# #### 8) precipitation 
# #### 9) solar radiation 
# #### 10) specific humidity 
# 
# 
# ### Response Variable:
# 
# #### Landslide Confidence (LANDSLID_3) - Low  (0) or Moderate/High (1) <br>
# 
# 

# ## Preliminary Data Import and Analysis

# In[434]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')

from pyproj import Proj
import StringIO
from pandas import DataFrame
import pandas as pd
import seaborn as sns
import pydot
from IPython.display import Image

from urllib2 import Request, urlopen
import json
from pandas.io.json import json_normalize
import numpy
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# ### Import csv file for all 52,000+WA landslides

# In[435]:

import csv

#ifile  = open('/git/stat504-python/STAT504-PROJECT/data/WALandslides_export1.csv', "rb")
#reader = csv.reader(ifile)
#ifile.close()


# In[436]:

walandslides_all = pd.read_csv('/git/data/landslides/WALandslides_export4.csv')


# In[437]:

#walandslides = pd.read_csv('/git/data/landslides/WALandslides_geol4subset.csv')


# In[438]:

walandslides = pd.read_csv('/git/data/landslides/WALandslides_geol4.csv')


# ### Table of all values for initial landslide dataset

# In[439]:

walandslides_allxy = [walandslides_all.OBJECTID, walandslides_all.POINT_X, walandslides_all.POINT_Y]
walandslides_allxy = pd.DataFrame(walandslides_allxy)
walandslides_allxy = walandslides_allxy.transpose()
walandslides = pd.DataFrame(walandslides)
walandslides = pd.merge(walandslides_allxy, walandslides, on='OBJECTID', how='outer')


# ### Set landslide type as numeric

# In[440]:

stringh1 = set(walandslides.LANDSLIDE1)
J = list(stringh1)
J2 = list(range(1, 106))


# In[441]:

i2 = iter(J)
j2 = iter(J2)
k2 = list(zip(i2, j2))

#outRes = dict((l[i], l[i+1]) if i+1 < len(l) else (l[i], '') for i in xrange(len(l)))
kdict2 = dict(k2)
kdict2.values()

walandslides['LANDSLIDE1'].replace(kdict2, inplace=True)
#print walandslides.LANDSLIDE1


# ### Convert categorical text columns to categorical numerical

# In[442]:

walandslides['DATA_CONFI'] = walandslides.DATA_CONFI.map({'Low':0, 'Moderate-High':1})
walandslides['SLOPE_MORP'] = walandslides.SLOPE_MORP.map({'Planar':0, 'Concave-Planar':1, 'Concave, convergent':2, 'Planar-Concave':3, 'Planar-convex':4})
walandslides['LANDSLID_3'] = walandslides.LANDSLID_3.map({'Questionable':0, 'Probable':1, 'Certain':2, 'Unknown':3})


# ### Set feature variables and response variable, reduce dataset by eliminating NANs

# In[443]:

walandslides = walandslides[np.isfinite(walandslides['SLOPE_MORP'])]
walandslides = walandslides[np.isfinite(walandslides['LANDSLIDE1'])]
#walandslides = walandslides[np.isfinite(walandslides['DATA_CONFI'])]

#walandslides.loc[walandslides['GEOLOGIC_U'] == 'Evb(gr)'] = 65
#walandslides.loc[walandslides['LANDSLID_3'] == 'Unknown'] = 65
walandslides = walandslides[walandslides.LANDSLID_3 != 3]
walandslides = walandslides[walandslides.GRADIENT_D != 0]


#walandslides = walandslides[walandslides.LANDSLID_4 != NaN]

#feature_cols = ['gradient_cat', 'GEOLOGIC_1', 'KernelD_GE', 'reacch_soi', 'daily_accu', 'precipitat', 'daily_maxi', 'daily_mini', 'daily_mean', 'daily_mi_1', 'daily_ma_1']
feature_cols = ['gradient_cat', 'GEOLOGIC_1', 'reacch_soi', 'LAND_USE', 'LANDSLIDE1']
#feature_cols = ['GRADIENT_D', 'GEOLOGIC_1', 'KernelD_GE', 'reacch_soi', 'POINT_X', 'POINT_Y']
#class_cols = ['DATA_CONFI']
#feature_cols = ['LANDSLIDE1']


# ### Convert GRADIENT_D to categorical, using quantiles

# In[444]:

labelz = ["1", "2", "3", "4", "5"]


# In[445]:

gradient = pd.DataFrame(walandslides.GRADIENT_D)


# In[446]:

gradient = pd.DataFrame(pd.qcut(walandslides.GRADIENT_D, 5, labels = labelz))


# In[447]:

gradient.rename(columns={'GRADIENT_D':'gradient_cat'}, inplace=True)


# In[448]:

walandslides = pd.concat([walandslides, gradient], axis=1, join_axes=[gradient.index])


# ### Plot the refined WA landslides, after eliminating NaNs and other missing values for all variables (~12000 landslide observations)

# In[449]:

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic(u'matplotlib nbagg')

#my_map = Basemap(projection='ortho',lon_0=-105,lat_0=40,resolution='l')

# make sure the value of resolution is a lowercase L,
#  for 'low', not a numeral 1
#my_map = Basemap(projection='ortho', lat_0=50, lon_0=-100,
#             resolution='l', area_thresh=500)

my_map2 = Basemap(llcrnrlon=-125, llcrnrlat=45, urcrnrlon=-115,
     urcrnrlat=51, projection='tmerc', lat_1=33, lat_2=45,
     lon_0=-119, lat_0=45, resolution='h', area_thresh=10000)

my_map2.drawcoastlines()
my_map2.drawcountries()
my_map2.fillcontinents(color='coral')

lon2 = np.array(walandslides.POINT_X)
lat2 = np.array(walandslides.POINT_Y)

#lon = np.array(walandslides.POINT_X_x)
#lat = np.array(walandslides.POINT_Y_x)

y2,x2 = my_map2(lon2, lat2)

#y,x = my_map2(lon, lat)

my_map2.plot(y2,x2, 'ro', markersize=4, markeredgecolor = 'k')
#my_map2.plot(y,x, 'ro', markersize=4, markeredgecolor = 'k')

#my_map2.plot(y,x, 'ro', markersize=4) # plots oregon data
#cbar = plt.colorbar(sc, shrink = .5)

plt.show()


# In[450]:

X = walandslides[feature_cols]
y = walandslides.LANDSLID_3


# In[451]:

y = pd.concat([y], axis=1)


# ### Final X and y - reduced to a size of 12241 observations

# In[452]:

X = pd.get_dummies(X)
X


# In[453]:

#y = pd.DataFrame(y)


# ### Initial breakdown of gradient by degree (categorized)

# In[454]:

gradient_pd = walandslides['gradient_cat'].value_counts()


# In[455]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')

gradient_pd.plot(kind='bar');


# ### Initial breakdown of slope morphology

# In[456]:

slopemorp_pd = walandslides['SLOPE_MORP'].value_counts()


# In[457]:

slopemorp_pd.plot(kind='bar');


# ### Initial breakdown of geology

# In[458]:

geology_pd = walandslides['GEOLOGIC_U'].value_counts()


# In[459]:

geology_pd.plot(kind='bar');


# ### Initial breakdown of land use

# In[460]:

landuse_pd = walandslides['LAND_USE'].value_counts()


# In[461]:

landuse_pd.plot(kind='bar');


# ### Initial breakdown of land use

# In[462]:

landslidetype_pd = walandslides['LANDSLIDE1'].value_counts()


# In[463]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')
landslidetype_pd.plot(kind='bar');


# ### Initial breakdown of landslide liklihood field (will be the response variable) 0=Questionable, 1=Probable, 3=Certain.

# In[464]:

landslid3_pd = walandslides['LANDSLID_3'].value_counts()


# In[465]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')
landslid3_pd.plot(kind='bar');


# ### Landslide liklihood counts  (0=questionable,1=probable, 2=certain)

# In[466]:

y['LANDSLID_3'].value_counts()


# In[467]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ### Geology compared to other feature variables

# In[468]:

sns.jointplot(x="GEOLOGIC_1", y="LANDSLID_3", data=walandslides);


# ### Gradient compared to other feature variables

# In[469]:

sns.jointplot(x="GRADIENT_D", y="LANDSLID_3", data=walandslides);


# In[470]:

sns.jointplot(x="KernelD_GE", y="LANDSLID_3", data=walandslides);


# ## Box plot showing how slope morphology varies by landslide type, for differing landslide liklihoods. (0=questionable, 1=probable, 2=certain)

# In[471]:

ax = sns.boxplot(x="SLOPE_MORP", y="LANDSLIDE1", hue="LANDSLID_3",
     data=walandslides, palette="Set3")


# ## Linear relationship between slope morphology and landslide type, conditioned for differing landslide liklihoods. (0=questionable, 1=probable, 2=certain),

# In[472]:

sns.lmplot(x="SLOPE_MORP", y="LANDSLIDE1", hue="LANDSLID_3",
     data=walandslides, ci=70, aspect=1, x_jitter=.03, size=10)


# ## Box plot showing how gradient varies by slope morphology, for differing landslide liklihoods. (0=questionable, 1=probable, 2=certain)

# In[473]:

ax = sns.boxplot(x="SLOPE_MORP", y="GRADIENT_D", hue="LANDSLID_3",
     data=walandslides, palette="Set3")


# ## Linear relationship between gradient and slope morphology - conditioned on the differing landslide liklihoods. (0=questionable, 1=probable, 2=certain)

# In[474]:

sns.lmplot(x="GEOLOGIC_1", y="GRADIENT_D", hue="LANDSLID_3",
     data=walandslides, ci=100, aspect=1, x_jitter=.03, size=10)


# ## Box plot showing how landslide types varies by gradient, for differing landslide liklihoods. (0=questionable, 1=probable, 2=certain)

# In[475]:

ax = sns.boxplot(x="GEOLOGIC_1", y="GRADIENT_D", hue="LANDSLID_3",
     data=walandslides, palette="Set3")


# ## Linear relationship between landslide type and gradient - conditioned on differing landslide liklihoods.  (0=questionable, 1=probable, 2=certain)

# In[476]:

sns.lmplot(x="KernelD_GE", y="GRADIENT_D", hue="LANDSLID_3",
     data=walandslides, ci=100, aspect=1, x_jitter=0, size=10)


# ## run train test split

# In[477]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)


# ## Begin Model analysis

# ### Calculate Null Accuracy

# In[478]:

#y_test.value_counts()


# In[479]:

# this works regardless of the number of classes
y['LANDSLID_3'].value_counts().head(1) / len(y['LANDSLID_3'])


# ### 10-fold cross-validation with logistic regression
# 

# In[377]:

y = np.ravel(y) #flatten y for appropriate dimensionality for interation with X


# In[390]:

logregtime1 = get_ipython().magic(u'%timeit -o 1 + 2')

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg_scores_mean = cross_val_score(logreg, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy').mean()
print logreg_scores_mean


# In[383]:

logreg_scores = cross_val_score(logreg, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')


# ### Naive Bayes Multinomial

# In[396]:

nbmtime1 = get_ipython().magic(u'%timeit -o 1 + 2')
from sklearn.naive_bayes import MultinomialNB
nbm = MultinomialNB()
nbm.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nbm_scores = cross_val_score(nbm, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')
print nbm_scores
#print(clf.predict(X[2:3]))


# In[397]:

mean_score = nbm_scores.mean()
std_dev = nbm_scores.std()
std_error = nbm_scores.std() / math.sqrt(nbm_scores.shape[0])
ci =  2.262 * std_error
lower_bound = mean_score - ci
upper_bound = mean_score + ci

print "Multinomial NB Score is %f +/-  %f" % (mean_score, ci)
print "Multinomial NB AUC is "
print '95 percent probability that if this experiment were repeated over and over the average score would be between %f and %f' % (lower_bound, upper_bound)


# ### 10-fold cross-validation with K Nearest Neighbor
# 

# In[398]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
knn_scores = cross_val_score(knn, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')
print knn_scores


# In[399]:

mean_score = knn_scores.mean()
std_dev = knn_scores.std()
std_error = knn_scores.std() / math.sqrt(knn_scores.shape[0])
ci =  2.262 * std_error
lower_bound = mean_score - ci
upper_bound = mean_score + ci

print "K nearest neighbhor Score is %f +/-  %f" % (mean_score, ci)
print '95 percent probability that if this experiment were repeated over and over the average score would be between %f and %f' % (lower_bound, upper_bound)


# ### Search for an optimal value of K for KNN
# 

# In[270]:

# search for an optimal value of K for KNN
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores


# In[271]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# ### Decision Tree - Fit a classification tree with an initial max_depth=3 on all data
# 

# In[408]:

from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
scores = cross_val_score(treeclf, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')
treeclf.fit(X, y)
print scores


# ### MODEL 6: Decision Tree - search for an optimal gamma/depth for Decision Tree

# In[405]:

from sklearn.tree import DecisionTreeClassifier
t_range = range(1, 100)
t_scores = []
for k in t_range:
    clf = DecisionTreeClassifier(max_depth=k)
    scores = cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')
    t_scores.append(scores.mean())
print t_scores


# ### MODEL 6: Decision Tree - Plot accuracy of cross validation runs vs. values of depth

# In[406]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')
plt.plot(t_range, t_scores)
plt.xlabel('Value of Depth for Decision Tree')
plt.ylabel('Cross-Validated Accuracy')


# ### MODEL 6: Decision Tree - Model optimization results:
# 
# After examining accuracy for a variety of depths, its appears that a value of ~45 for a max depth is optimal in terms of cross validation accuracy.

# In[409]:

clftime1 = get_ipython().magic(u'%timeit -o 1 + 2')
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
treeclf_scores = cross_val_score(clf, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')
treeclf.fit(X, y)
print treeclf_scores


# In[410]:

print("Accuracy: %0.2f (+/- %0.2f)" % (treeclf_scores.mean(), treeclf_scores.std() * 2))


# In[305]:

from StringIO import StringIO
tree_landslide = StringIO()

from sklearn.tree import DecisionTreeClassifier, export_graphviz
export_graphviz(treeclf, out_file=tree_landslide)


# In[306]:

graph = pydot.graph_from_dot_data(tree_landslide.getvalue())  
Image(graph.create_png())


# In[ ]:

### Random Forest


# In[414]:

from sklearn.ensemble import RandomForestClassifier
rfreg = RandomForestClassifier()
rfreg


# In[415]:


# list of values to try for n_estimators
estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg = RandomForestClassifier(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# In[228]:

# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')


# In[229]:

# show the best RMSE and the corresponding max_features
sorted(zip(RMSE_scores, estimator_range))[0]


# In[230]:

# list of values to try for max_features
feature_range = range(1, len(feature_cols)+1)

# list to store the average RMSE for each value of max_features
RMSE_scores = []

# use 10-fold cross-validation with each value of max_features (WARNING: SLOW!)
for feature in feature_range:
    rfreg = RandomForestRegressor(n_estimators=30, max_features=feature, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# In[231]:

# plot max_features (x-axis) versus RMSE (y-axis)
plt.plot(feature_range, RMSE_scores)
plt.xlabel('max_features')
plt.ylabel('RMSE (lower is better)')


# In[232]:

# show the best RMSE and the corresponding max_features
sorted(zip(RMSE_scores, feature_range))[0]


# In[233]:

# max_features=2 is best and n_estimators=150 is sufficiently large
rfreg = RandomForestRegressor(n_estimators=30, max_features=2, oob_score=True, random_state=1)
rfreg.fit(X, y)


# In[234]:

# compute feature importances
pd.DataFrame({'feature':feature_cols, 'importance':rfreg.feature_importances_}).sort('importance')


# In[235]:

# compute the out-of-bag R-squared score
rfreg.oob_score_


# In[236]:

# set a threshold for which features to include
print rfreg.transform(X, threshold=0.1).shape
print rfreg.transform(X, threshold='mean').shape
print rfreg.transform(X, threshold='median').shape


# In[237]:

# create a new feature matrix that only includes important features
X_important = rfreg.transform(X, threshold='mean')


# In[238]:

# check the RMSE for a Random Forest that only includes important features
rfreg = RandomForestRegressor(n_estimators=30, max_features=2, random_state=1)
scores = cross_val_score(rfreg, X_important, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


# ### Bootstrap aggregation of decision tree

# In[418]:

bag_scores = []

bag_range = range(1, 100)

for k in bag_range:
    bag_clf = BaggingClassifier(treeclf, n_estimators=k, max_samples=1.0, max_features=5, bootstrap = False, bootstrap_features = True, random_state=42)
    bag_clf.fit(X, y)
    MSE_scores = cross_val_score(bag_clf, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='mean_squared_error')
    bag_scores.append(bag_clf_scores.mean())
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
print bag_clf_scores


# In[ ]:

bag_clf_scores = cross_val_score(bag_clf, X, y, cv=KFold(X.shape[0], n_folds=10, shuffle=True), scoring='accuracy')


# In[420]:

import matplotlib
get_ipython().magic(u'matplotlib nbagg')
# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(bag_range, bag_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')


# ### Gaussian Naive Bayes

# In[421]:

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d"
       % (X.shape[0],(y != y_pred).sum()))


# ### Multinomial Naive Bayes

# In[422]:

from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
y_pred = gnb.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d"
       % (X.shape[0],(y != y_pred).sum()))


# ### Bernoulli Naive Bayes

# In[423]:

from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
y_pred = gnb.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d"
       % (X.shape[0],(y != y_pred).sum()))


# In[252]:

y = pd.DataFrame(y)


# In[253]:

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((11,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1


# In[246]:

y


# In[255]:

l1 = pd.DataFrame(l1)


# In[270]:

l1


# In[263]:

l1.columns = ['LANDSLID_3']


# In[265]:

l1


# In[266]:

l1['LANDSLID_3'].value_counts()


# In[268]:

nonlin(np.dot(l0,syn0))


# In[ ]:




# ##  END: The api allow a user to send features to the model - and it returns a prediction!!

# In[239]:

url = "http://129.101.160.58:5000/api"
data = json.dumps({'cool':1, 'useful':8, 'funny':2})
r = requests.post(url, data)

print r.json()

