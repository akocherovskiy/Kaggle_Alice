
# First, we will import necessary libraries and set display settings.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import eli5
import datetime as dt
from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['image.cmap'] = 'plasma'

# Here we will create a function to ease a way to write our submission in file

def write_to_sub_file(y_pred, file_name):
    
    prediction = pd.DataFrame(y_pred, 
                              index=np.arange(1, y_pred.shape[0] + 1),
                              columns=['target'])
    
    prediction.to_csv(file_name, index_label='session_id')


# Let's read our datasets and see some information about them.

df_train = pd.read_csv('/home/kap/DS/DataSets/Alice/train_sessions.csv')
df_test = pd.read_csv('/home/kap/DS/DataSets/Alice/test_sessions.csv')

df_train = df_train.sort_values(by='time1')
df_train.head(2)

df_train.info()
print('-' * 40)
df_test.info()

# As usual, we have some number of empty values. Let's read 'site_dic.pkl' (ids of sites) and check if datasets contain sites with index '0'. If not, we can fill empty cells with zeros.

site_dict = pd.read_pickle('/home/kap/Downloads/site_dic.pkl')
df_site_name = pd.DataFrame.from_dict(site_dict, orient='index', columns=['A'])
df_site_name.sort_values(by='A', ascending=True).head(3)

# To create the first simple prediction, we will get rid of columns 'time' and 'session_id' and try to predict session-owner using only site IDs.

time_col = [col for col in df_train.columns if 'time' in col or 'sess' in col]

X_train = df_train.drop(columns=time_col, axis=1).fillna(0).astype('int').iloc[:, :-1]
y_train = df_train.target.astype('int')
X_test = df_test.drop(columns=time_col, axis=1).fillna(0).astype('int')

X_train.shape, X_test.shape, y_train.shape

df_train.info()
print('-' * 40)
df_test.info()

# Now we will look at distribution ours target. It is important to know to understand how real target can correlate  with prediction. Using this information, we can pick more valuable metrics. 

sns.histplot(data=df_train.target);

df_train.target.value_counts(normalize=True)

# Next, we will perform cross-validation with Decision Tree and Random Forest.

tree = DecisionTreeClassifier(random_state=17)
rf = RandomForestClassifier(random_state=17)

get_ipython().run_cell_magic('time', '', "\n#Decision Tree\n\ncv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)\ncv_scores, cv_scores.mean()")

get_ipython().run_cell_magic('time', '', "\n#Random Forest\n\ncv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)\ncv_scores, cv_scores.mean()")

# Using Random Forest we got roc_auc = 0.88 on our training set. Let's add start session time and map them to 0 - morning, 1 - day, 2 - evening and 3 - night.

X_train['hour'] = df_train.time1.astype('datetime64[ns]').dt.hour
X_test['hour'] = df_test.time1.astype('datetime64[ns]').dt.hour
X_train.shape, X_test.shape

h = np.arange(7, 24)

#day_time = ['morning'] * 4 + ['day'] * 4 + ['evening'] * 5 + ['night'] * 4

day_time = [0] * 4 + [1] * 4 + [2] * 5 + [3] * 4
day_time_dict = dict(zip(h, day_time))

X_train['hour'] = X_train['hour'].map(day_time_dict)
X_test['hour'] = X_test['hour'].map(day_time_dict)

X_train.shape, X_test.shape, y_train.shape

get_ipython().run_cell_magic('time', '', "\ncv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)\ncv_scores, cv_scores.mean()")

# Here we got roc_auc = 0.92. Let's submit this result.

rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:,1]

write_to_sub_file(y_pred, 'sub4.csv') # Submission score - 0.77334

# Finally, we will choose the best parameters for our model.

params = {'max_depth': np.arange(8, 20, 4), 'min_samples_split': np.arange(2, 14, 4)}

clf = GridSearchCV(rf, param_grid=params, scoring='roc_auc', cv=5)

get_ipython().run_cell_magic('time', '', '\nclf.fit(X_train, y_train)')

clf.best_estimator_
y_pred = clf.predict_proba(X_test)[:,1]

write_to_sub_file(y_pred, 'sub6.csv') # Submission score - 0.81486

# Ok. Now we will use a slightly more advanced technique that named "Bag of words".
# When we work with time series, we should use a more advanced method to cross-validation (Time Series Split). So, first we will sort our dataset by ascending order, and then we will split it into 8 parts.
# Here is an image to explain:

# ![image.png](attachment:image.png)

X_train = df_train.drop(columns=time_col, axis=1).fillna(0).astype('int').iloc[:, :-1]

time_split = TimeSeriesSplit(n_splits=8)

[(e[0].shape, e[1].shape) for e in time_split.split(X_train)]

# Transform our data to feed it CountVecrorizer.

X_train.to_csv('X_train_txt.txt', sep=' ', header=None, index=None)
X_test.to_csv('X_test_txt.txt', sep=' ', header=None, index=None)

get_ipython().system('head X_train_txt.txt')

vect = CountVectorizer(ngram_range=(1, 3), max_features=80000)

get_ipython().run_cell_magic('time', '', "X_train_vectorized = vect.fit_transform(open('X_train_txt.txt'))\nX_test_vectorized = vect.transform(open('X_test_txt.txt'))\n\nX_train_vectorized.shape, X_test_vectorized.shape")

# Perform time series cross-validation with logistic regression.

logreg_vect = LogisticRegression(C=1, random_state=17, solver='liblinear'
get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_vectorized, y_train, scoring='roc_auc', cv=time_split)\ncv_score.mean()")

logreg_vect.fit(X_train_vectorized, y_train)

y_pred = logreg_vect.predict_proba(X_test_vectorized)[:, 1]

write_to_sub_file(y_pred, 'sub8_logreg_vectorized.csv') # Submission score - 0.91326

# Using eli5 we can see the weight of features.

eli5.show_weights(estimator=logreg_vect, feature_names=vect.get_feature_names(), top=10)

# Next we will add start session time. 

hour_train = df_train['time1'].astype('datetime64[ns]').apply(lambda ts: ts.hour)
hour_test = df_test['time1'].astype('datetime64[ns]').apply(lambda ts: ts.hour)

morning_train = ((hour_train >= 7) & (hour_train <= 11)).astype('int')
day_train = ((hour_train >= 12) & (hour_train <= 18)).astype('int')
evening_train = ((hour_train >= 19) & (hour_train <= 23)).astype('int')
night_train = ((hour_train >= 0) & (hour_train <= 6)).astype('int')

morning_test = ((hour_test >= 7) & (hour_test <= 11)).astype('int')
day_test = ((hour_test >= 12) & (hour_test <= 18)).astype('int')
evening_test = ((hour_test >= 19) & (hour_test <= 23)).astype('int')
night_test = ((hour_test >= 0) & (hour_test <= 6)).astype('int')

get_ipython().run_cell_magic('time', '', '\nX_train_new = hstack([X_train_vectorized, morning_train.values.reshape(-1, 1),\n                     day_train.values.reshape(-1, 1), evening_train.values.reshape(-1, 1),\n                     night_train.values.reshape(-1, 1)])\n\nX_test_new = hstack([X_test_vectorized, morning_test.values.reshape(-1, 1),\n                     day_test.values.reshape(-1, 1), evening_test.values.reshape(-1, 1),\n                     night_test.values.reshape(-1, 1)]) # 0.9770486694430115')

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_new, y_train, scoring='roc_auc', cv=time_split)\ncv_score.mean()")

# Roc_auc metric increased significantly (0.85 > 0.91). Let's do another submission.

logreg_vect.fit(X_train_new, y_train)

y_pred = logreg_vect.predict_proba(X_test_new)[:, 1]

write_to_sub_file(y_pred, 'sub10_logreg_vectorized.csv') # Submission score - 0.93897

# Next, we will see what favorite days for browsing the internet for Alice and other users.

day_of_week_train = (df_train.time1.astype('datetime64[ns]').dt.day_of_week).astype('int')
day_of_week_test = (df_test.time1.astype('datetime64[ns]').dt.day_of_week).astype('int')

plt.subplots(1, 3, figsize = (10, 4))

plt.subplot(1, 2, 1)
sns.countplot(work_day_train[y_train == 1])
plt.title("Alice")
plt.xlabel('day of week')

plt.subplot(1, 2, 2)
sns.countplot(day_of_week_train[y_train == 0])
plt.title('Others')
plt.xlabel('day of week');

# Alice browsing more on Monday, then her activity decreases, while other users prefer Wednesday. We can use this distinction to refine the model. Add some new features: day of week, session duration and year-month values.

session_duration_train = ((df_train.time10.astype('datetime64[ns]') - df_train.time1.astype('datetime64[ns]')) 
     / np.timedelta64(1, 'h')).fillna(0)

session_duration_test = ((df_test.time10.astype('datetime64[ns]') - df_test.time1.astype('datetime64[ns]')) 
     / np.timedelta64(1, 'h')).fillna(0)

time_train = df_train.time1.astype('datetime64[ns]')
time_test = df_test.time1.astype('datetime64[ns]')
time_train_min = time_train.dt.year.min()
time_test_min = time_test.dt.year.min()

# To avoid a big variety of data and do not break the logistic regression model, we need to 'normalize' data.
# Let's implement year_month feature as number of months since first session and divide it by 100.

year_month_train = time_train.apply(lambda ts: ((ts.year - time_train_min) * 12 + ts.month) / 1e2 )
year_month_test = time_test.apply(lambda ts: ((ts.year - time_train_min) * 12 + ts.month) / 1e2 )

X_train_newest = hstack([X_train_new, day_of_week_train.values.reshape(-1, 1),
                                      session_duration_train.values.reshape(-1, 1),
                                      year_month_train.values.reshape(-1, 1)])


X_test_newest = hstack([X_test_new, day_of_week_test.values.reshape(-1, 1),
                                    session_duration_test.values.reshape(-1, 1),
                                    year_month_test.values.reshape(-1, 1)])

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_newest, y_train, scoring='roc_auc', cv=time_split)\ncv_score.mean()")

logreg_vect.fit(X_train_newest, y_train)

y_pred = logreg_vect.predict_proba(X_test_newest_ever)[:, 1]

write_to_sub_file(y_pred, 'sub11_logreg_vectorized_timefeatures.csv') # Submission score - 0.93897

# At the end, we will tune our model.

c = np.logspace(-2, 2, 10, base=10)

grcv_cls = GridSearchCV(logreg_vect, param_grid={'C': c}, scoring='roc_auc', cv=time_split)

get_ipython().run_cell_magic('time', '', '\ngrcv_cls.fit(X_train_new, y_train)')

grcv_cls.best_estimator_

y_pred = grcv_cls.predict_proba(X_test_new)[:, 1]

write_to_sub_file(y_pred, 'sub11_logreg_vectorized_fitted.csv') # Submission score - 0.93925

X_train_newest = hstack([X_train_new, work_day_train.values.reshape(-1, 1)])
X_test_newest = hstack([X_test_new, work_day_test.values.reshape(-1, 1)])

X_train_newest.shape, X_test_newest.shape

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_newest, y_train, scoring='roc_auc', cv=5)\ncv_score.mean() # 0.9782336911625977")

y_pred = grcv_cls.predict_proba(X_test_newest)[:, 1]

write_to_sub_file(y_pred, 'sub11_logreg_vectorized_dayweek.csv') # Submission score - 0.94116

# month_train = (df_train.time1.astype('datetime64[ns]').dt.month).astype('int')
# month_day_test = (df_test.time1.astype('datetime64[ns]').dt.month).astype('int')

# X_train_newest_ever = hstack([X_train_newest, month_train.values.reshape(-1, 1)])
# X_test_newest_ever = hstack([X_test_newest, month_day_test.values.reshape(-1, 1)])

X_train_newest_ever.shape, X_test_newest_ever.shape

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_newest, y_train, scoring='roc_auc', cv=5)\ncv_score.mean() # 0.9793380777964421")

logreg_vect.fit(X_train_newest_ever, y_train)

y_pred = logreg_vect.predict_proba(X_test_newest_ever)[:, 1]

write_to_sub_file(y_pred, 'sub12_logreg_vectorized_dayweek_m.csv') # Submission score - 0.94633

session_duration_train = ((df_train.time10.astype('datetime64[ns]') - df_train.time1.astype('datetime64[ns]')) 
     / np.timedelta64(1, 'h')).fillna(0)

session_duration_test = ((df_test.time10.astype('datetime64[ns]') - df_test.time1.astype('datetime64[ns]')) 
     / np.timedelta64(1, 'h')).fillna(0)

X_train_newest_ever = hstack([X_train_newest_ever, session_duration_train.values.reshape(-1, 1)])
X_test_newest_ever = hstack([X_test_newest_ever, session_duration_test.values.reshape(-1, 1)])

X_train_newest_ever.shape, X_test_newest_ever.shape

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_newest_ever, y_train, scoring='roc_auc', cv=5)\ncv_score.mean() # 0.9791757984447962")

logreg_vect.fit(X_train_newest_ever, y_train)

y_pred = logreg_vect.predict_proba(X_test_newest_ever)[:, 1]

write_to_sub_file(y_pred, 'sub12_logreg_vectorized_dayweek_m_td.csv') # Submission score - 0.94641

time_train = df_train.time1.astype('datetime64[ns]')
time_test = df_test.time1.astype('datetime64[ns]')
time_train_min = time_train.dt.year.min()
time_test_min = time_test.dt.year.min()

year_month_train = time_train.apply(lambda ts: ((ts.year - time_train_min) * 12 + ts.month) / 1e2 )
year_month_test = time_test.apply(lambda ts: ((ts.year - time_train_min) * 12 + ts.month) / 1e2 )

X_train_newest_ever_n = hstack([X_train_newest_ever, year_month_train.values.reshape(-1, 1)])
X_test_newest_ever_n = hstack([X_test_newest_ever, year_month_test.values.reshape(-1, 1)])

X_train_newest_ever_n.shape, X_test_newest_ever_n.shape

logreg_vect.fit(X_train_newest_ever_n, y_train)

y_pred = logreg_vect.predict_proba(X_test_newest_ever_n)[:, 1]

write_to_sub_file(y_pred, 'sub12_logreg_vectorized_dayweek_td_ym.csv') # Submission score - 0.94641

get_ipython().run_cell_magic('time', '', "\ncv_score = cross_val_score(logreg_vect, X_train_newest_ever_n, y_train, scoring='roc_auc', cv=5)\ncv_score.mean() # 0.9791757984447962")

eli5.show_weights(estimator=logreg_vect)
