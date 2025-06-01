import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas, joblib, sklearn, warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

#Chemins des features et labels associes au dataset Californie
features = pandas.read_csv('./californie/alt_acsincome_ca_features_85.csv')
labels = pandas.read_csv('./californie/alt_acsincome_ca_labels_85.csv')

#Graphique distribution age
# plt.hist(features['AGEP'], bins=range(10,100,5), edgecolor='black', color='orange')
# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

#Graphique distribution categorie professionnelle
# val, cnt = np.unique(features['COW'],return_counts=True)
# plt.bar(val, cnt, align='center', alpha=0.8, edgecolor='black', color='orange')
# plt.title("Class of worker distribution")
# plt.xlabel("Class of worker")
# plt.ylabel("Frequency")
# plt.show()

#Graphique distribution niveau d'études
# val, cnt = np.unique(features['SCHL'],return_counts=True)
# plt.bar(val, cnt, align='center', alpha=0.8, edgecolor='black', color='orange')
# plt.title("Educational attainment distribution")
# plt.xlabel("Educational attainment")
# plt.ylabel("Frequency")
# plt.show()

#Graphique distribution statut marital
# val, cnt = np.unique(features['MAR'],return_counts=True)
# plt.bar(val, cnt, align='center', alpha=0.8, edgecolor='black', color='orange')
# plt.title("Marital status distribution")
# plt.xlabel("Marital status")
# plt.ylabel("Frequency")
# plt.show()

#Repartition genre
# print(f"Proportion femmes : {(features['SEX'].value_counts()[2.0])*100/len(features['SEX'])}")
# print(f"Proportion hommes : {(features['SEX'].value_counts()[1.0])*100/len(features['SEX'])}")

# plt.hist(features['OCCP'])
# plt.title("OCCP");

# plt.show();

#Partitionnement
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

#Preparation
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train.to_numpy())
x_test_scaled = sc.transform(x_test.to_numpy())
# joblib.dump(sc, 'scaler.joblib')

#Random Forest default
# rf_default=RandomForestClassifier()
# rf_default.fit(x_train_scaled,y_train)
# cv_score=cross_val_score(rf_default, x_train_scaled, y_train, cv=5)
# y_pred = rf_default.predict(x_test_scaled)

# print(f"\ncv score RF default : {np.mean(cv_score)}")
# print(f"\naccuracy RF default : {accuracy_score(y_test,y_pred)}")
# print(f"\ncr RF default :\n {classification_report(y_test,y_pred)}")

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Random Forest Default")
# plt.show()

#Random Forest best
# rf_param = {'n_estimators' : (10, 50, 100, 150),
#             'max_depth': (None, 10, 20),
#             'min_samples_leaf' : (10, 50, 100, 150)}

# grid_search = GridSearchCV(RandomForestClassifier(), rf_param, cv=5)
# grid_search.fit(x_train_scaled, y_train)
# rf_best = grid_search.best_estimator_
# print(f"\n\nBest score RF : {grid_search.best_score_}")
# print(f"Best param RF : {grid_search.best_params_}\n")
# joblib.dump(grid_search.best_estimator_, 'RandomForest_BestModel_.joblib')#TODO: rename file
# y_pred = rf_best.predict(x_test_scaled)

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Random Forest Best")
# plt.show()


#Ada Boost default
# ab_default=AdaBoostClassifier()
# ab_default.fit(x_train_scaled,y_train)
# cv_score=cross_val_score(ab_default, x_train_scaled, y_train, cv=5)
# y_pred = ab_default.predict(x_test_scaled)

# print(f"\ncv score AB default : {np.mean(cv_score)}")
# print(f"\naccuracy AB default : {accuracy_score(y_test,y_pred)}")
# print(f"\ncr AB default :\n {classification_report(y_test,y_pred)}")

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Ada Boost Default")
# plt.show()

#Ada Boost best
# ab_param = {'n_estimators' : (10, 50, 100, 200, 500),
#             'learning_rate' : (0.1, 0.5, 0.75, 1, 1.25, 1.5)}

# grid_search = GridSearchCV(AdaBoostClassifier(), ab_param, cv=5)
# grid_search.fit(x_train_scaled, y_train)
# ab_best = grid_search.best_estimator_
# print(f"\n\nBest score AB : {grid_search.best_score_}")
# print(f"Best param AB : {grid_search.best_params_}\n")
# joblib.dump(grid_search.best_estimator_, 'AdaBoost_BestModel_.joblib')#TODO: rename file
# y_pred = ab_best.predict(x_test_scaled)

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Ada Boost Best")
# plt.show()


#Gradient Boosting default
# gb_default=GradientBoostingClassifier()
# gb_default.fit(x_train_scaled,y_train)
# cv_score=cross_val_score(gb_default, x_train_scaled, y_train, cv=5)
# y_pred = gb_default.predict(x_test_scaled)

# print(f"\ncv score GB default : {np.mean(cv_score)}")
# print(f"\naccuracy GB default : {accuracy_score(y_test,y_pred)}")
# print(f"\ncr GB default :\n {classification_report(y_test,y_pred)}")

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Gradient Boosting Default")
# plt.show()


#Gradient Boosting best
# gb_param = {'n_estimators' : (10, 50, 100, 150),
#             'max_depth': (2, 10, 20),
#             'learning_rate' : (0.1, 0.5, 1)}

# grid_search = GridSearchCV(GradientBoostingClassifier(), gb_param, cv=5)
# grid_search.fit(x_train_scaled, y_train)
# gb_best = grid_search.best_estimator_
# print(f"\n\nBest score GB : {grid_search.best_score_}")
# print(f"Best param GB : {grid_search.best_params_}\n")
# joblib.dump(grid_search.best_estimator_, 'GradientBoosting_BestModel_.joblib')#TODO: rename file
# y_pred = gb_best.predict(x_test_scaled)

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Gradient Boosting Best")
# plt.show()


#Stacking default
# s_default=StackingClassifier([("rf", RandomForestClassifier()), ("ab", AdaBoostClassifier()), ("gb", GradientBoostingClassifier())])
# s_default.fit(x_train_scaled,y_train)
# cv_score=cross_val_score(s_default, x_train_scaled, y_train, cv=5)
# y_pred = s_default.predict(x_test_scaled)

# print(f"\ncv score S default : {np.mean(cv_score)}")
# print(f"\naccuracy S default : {accuracy_score(y_test,y_pred)}")
# print(f"\ncr S default :\n {classification_report(y_test,y_pred)}")

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Stacking Default")
# plt.show()

#Stacking best
# rf=joblib.load("RandomForest_BestModel_08195.joblib")
# ab=joblib.load("AdaBoost_BestModel_08188.joblib")
# gb=joblib.load("GradientBoosting_BestModel_08252.joblib")
# s_default=StackingClassifier([("rf", rf), ("ab", ab), ("gb", gb)])
# s_default.fit(x_train_scaled,y_train)
# cv_score=cross_val_score(s_default, x_train_scaled, y_train, cv=5)
# y_pred = s_default.predict(x_test_scaled)

# print(f"\ncv score S best : {np.mean(cv_score)}")
# print(f"\naccuracy S best : {accuracy_score(y_test,y_pred)}")
# print(f"\ncr S best :\n {classification_report(y_test,y_pred)}")
# joblib.dump(s_default,'Stacking_BestModel_.joblib')#TODO: rename file

# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred)).plot()
# plt.title("Confusion matrix - Stacking Best")
# plt.show()

#Test des modeles sur les jeux de données Nevada et Colorado
# rf_best=joblib.load("RandomForest_BestModel_08195.joblib")
# ab_best=joblib.load("AdaBoost_BestModel_08188.joblib")
# gb_best=joblib.load("GradientBoosting_BestModel_08252.joblib")
# s_best=joblib.load("Stacking_BestModel_08270.joblib")

# features_ne = sc.transform(pandas.read_csv('./colorado_nevada/acsincome_ne_features.csv'))
# labels_ne = pandas.read_csv('./colorado_nevada/acsincome_ne_labelTP2.csv').values.ravel()
# features_co = sc.transform(pandas.read_csv('./colorado_nevada/acsincome_co_features.csv'))
# labels_co = pandas.read_csv('./colorado_nevada/acsincome_co_label.csv').values.ravel()

# y_pred = rf_best.predict(features_co)
# print(f"Colorado: accuracy for Random Forest: {accuracy_score(labels_co,y_pred):.4f}")
# y_pred = ab_best.predict(features_co)
# print(f"Colorado: accuracy for Ada Boost: {accuracy_score(labels_co,y_pred):.4f}")
# y_pred = gb_best.predict(features_co)
# print(f"Colorado: accuracy for Gradient Boosting: {accuracy_score(labels_co,y_pred):.4f}")
# y_pred = s_best.predict(features_co)
# print(f"Colorado: accuracy for Stacking: {accuracy_score(labels_co,y_pred):.4f}")
# y_pred = rf_best.predict(features_ne)
# print(f"Nevada: accuracy for Random Forest: {accuracy_score(labels_ne,y_pred):.4f}")
# y_pred = ab_best.predict(features_ne)
# print(f"Nevada: accuracy for Ada Boost: {accuracy_score(labels_ne,y_pred):.4f}")
# y_pred = gb_best.predict(features_ne)
# print(f"Nevada: accuracy for Gradient Boosting: {accuracy_score(labels_ne,y_pred):.4f}")
# y_pred = s_best.predict(features_ne)
# print(f"Nevada: accuracy for Stacking: {accuracy_score(labels_ne,y_pred):.4f}")



# Correlation
# scaled_features = sc.transform(features)
# df_features = pandas.DataFrame(scaled_features, columns=features.columns)

# Diagramme importance features avant apprentissage
# df_features['PINCP'] = labels['PINCP'].values
# correlations = df_features.corr()['PINCP'].drop('PINCP')
# corr_d = pandas.DataFrame({'feature': correlations.index, 'importance': correlations.abs()}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=corr_d, color='orange')
# plt.title("Corrélation features avant apprentissage")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()

rf=joblib.load("RandomForest_BestModel_08195.joblib")
ab=joblib.load("AdaBoost_BestModel_08188.joblib")
gb=joblib.load("GradientBoosting_BestModel_08252.joblib")
s=joblib.load("Stacking_BestModel_08270.joblib")

# Diagramme importance features Random Forest
# corr_rf = pandas.DataFrame({'feature': features.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=corr_rf, color='orange')
# plt.title("Corrélation features Random Forest")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()

# Diagramme importance features Ada Boost
# corr_ab = pandas.DataFrame({'feature': features.columns, 'importance': ab.feature_importances_}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=corr_ab, color='orange')
# plt.title("Corrélation features Ada Boost")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()

# Diagramme importance features Gradient Boosting
# corr_gb = pandas.DataFrame({'feature': features.columns, 'importance': gb.feature_importances_}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=corr_gb, color='orange')
# plt.title("Corrélation features Gradient Boosting")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()

# Permutation Random Forest
# y_rf_test = rf.predict(x_test_scaled)
# rf_perm = permutation_importance(rf, x_test_scaled, y_rf_test, n_repeats=10, random_state=42)
# rf_perm_importances = pandas.DataFrame({'feature': features.columns, 'importance': rf_perm.importances_mean}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=rf_perm_importances, color='orange')
# plt.title("Permutation importance - Random Forest")
# plt.show()

# Permutation Ada Boost
# y_ab_test = ab.predict(x_test_scaled)
# ab_perm = permutation_importance(ab, x_test_scaled, y_ab_test, n_repeats=10, random_state=42)
# ab_perm_importances = pandas.DataFrame({'feature': features.columns, 'importance': ab_perm.importances_mean}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=ab_perm_importances, color='orange')
# plt.title("Permutation importance - Ada Boost")
# plt.show()

# Permutation Gradient Boosting
# y_gb_test = gb.predict(x_test_scaled)
# gb_perm = permutation_importance(gb, x_test_scaled, y_gb_test, n_repeats=10, random_state=42)
# gb_perm_importances = pandas.DataFrame({'feature': features.columns, 'importance': gb_perm.importances_mean}).sort_values(by='importance', ascending=False)
# sns.barplot(x='importance', y='feature', data=gb_perm_importances, color='orange')
# plt.title("Permutation importance - Gradient Boosting")
# plt.show()




# Equite
total_train = x_train.join(y_train)
total_test = x_test.join(y_test)

# Taux hommes/femmes avec revenu > 50000
# tot_size = len(total_train)
# nb_femmes = len(total_train[total_train['SEX']==2])
# nb_hommes = len(total_train[total_train['SEX']==1])
# print(f"h = {nb_hommes}, f= {nb_femmes}, t={tot_size}")
# print(f"\nPersonnes avec revenu > 50000: {len(total_train[total_train['PINCP'] == True])*100/tot_size:.2f}%")
# print(f"Femmes avec revenu > 50000: {len(total_train[(total_train['PINCP'] == True) & (total_train['SEX'] == 2)])*100/nb_femmes:.2f}%")
# print(f"Hommes avec revenu > 50000: {len(total_train[(total_train['PINCP'] == True) & (total_train['SEX'] == 1)])*100/nb_hommes:.2f}%")

# Matrice de confusion par genre
def conf_mat(matrix, title):
    ConfusionMatrixDisplay(matrix).plot()
    plt.title(f"Confusion matrix {title}")
    plt.show()
    tn, fp, fn, tp = matrix.ravel()
    print(f"Metrics {title}")
    print(f"Taux de prédictions positives: {(fp + tp) / (tn + fp + fn + tp):.2f}")
    print(f"Taux de vrais positifs: {tp / (fn + tp):.2f}")
    print(f"Taux de faux positifs: {fp / (fp + tn):.2f}\n")

# # Random Forest men
# rf_tot = x_test.copy()
# rf_tot['PINCP'] = rf.predict(x_test_scaled)
# rf_mat= confusion_matrix(total_test.loc[total_test['SEX']==1, ['PINCP']], rf_tot.loc[rf_tot['SEX']==1, ['PINCP']])
# conf_mat(rf_mat, "men - Random Forest")
# # Random Forest women
# rf_mat= confusion_matrix(total_test.loc[total_test['SEX']==2, ['PINCP']], rf_tot.loc[rf_tot['SEX']==2, ['PINCP']])
# conf_mat(rf_mat, "women - Random Forest")
# # Ada Boost men
# ab_tot = x_test.copy()
# ab_tot['PINCP'] = ab.predict(x_test_scaled)
# ab_mat= confusion_matrix(total_test.loc[total_test['SEX']==1, ['PINCP']], ab_tot.loc[ab_tot['SEX']==1, ['PINCP']])
# conf_mat(ab_mat, "men - Ada Boost")
# # Ada Boost women
# ab_mat= confusion_matrix(total_test.loc[total_test['SEX']==2, ['PINCP']], ab_tot.loc[ab_tot['SEX']==2, ['PINCP']])
# conf_mat(ab_mat, "women - Ada Boost")
# # Gradient Boosting men
# gb_tot = x_test.copy()
# gb_tot['PINCP'] = gb.predict(x_test_scaled)
# gb_mat= confusion_matrix(total_test.loc[total_test['SEX']==1, ['PINCP']], gb_tot.loc[gb_tot['SEX']==1, ['PINCP']])
# conf_mat(gb_mat, "men - Gradient Boosting")
# # Gradient Boosting women
# gb_mat= confusion_matrix(total_test.loc[total_test['SEX']==2, ['PINCP']], gb_tot.loc[gb_tot['SEX']==2, ['PINCP']])
# conf_mat(gb_mat, "women - Gradient Boosting")
# # Stacking men
# s_tot = x_test.copy()
# s_tot['PINCP'] = s.predict(x_test_scaled)
# s_mat= confusion_matrix(total_test.loc[total_test['SEX']==1, ['PINCP']], s_tot.loc[s_tot['SEX']==1, ['PINCP']])
# conf_mat(s_mat, "men - Stacking")
# # Stacking women
# s_mat= confusion_matrix(total_test.loc[total_test['SEX']==2, ['PINCP']], s_tot.loc[s_tot['SEX']==2, ['PINCP']])
# conf_mat(s_mat, "women - Stacking")

# Random Forest without gender
# x_test_no_gender = pandas.DataFrame(x_test_scaled, columns=features.columns).drop(columns=['SEX'])
# x_train_no_gender = pandas.DataFrame(x_train_scaled, columns=features.columns).drop(columns=['SEX'])
# rf_gender=RandomForestClassifier()
# rf_gender.fit(x_train_no_gender,y_train)

# # Random Forest without gender men
# rf_gen = x_test.copy()
# rf_gen['PINCP'] = rf_gender.predict(x_test_no_gender)
# rf_mat= confusion_matrix(total_test.loc[total_test['SEX']==1, ['PINCP']], rf_gen.loc[rf_gen['SEX']==1, ['PINCP']])
# conf_mat(rf_mat, "men - Random Forest without gender")
# # Random Forest without gender women
# rf_mat= confusion_matrix(total_test.loc[total_test['SEX']==2, ['PINCP']], rf_gen.loc[rf_gen['SEX']==2, ['PINCP']])
# conf_mat(rf_mat, "women - Random Forest without gender")

rf=joblib.load("RandomForest_BestModel_08195.joblib")
ab=joblib.load("AdaBoost_BestModel_08188.joblib")
gb=joblib.load("GradientBoosting_BestModel_08252.joblib")
s=joblib.load("Stacking_BestModel_08270.joblib")

joblib.dump(rf, "RandomForest_BestModel_08195.joblib", compress=5)
joblib.dump(ab, "AdaBoost_BestModel_08188.joblib", compress=5)
joblib.dump(gb, "GradientBoosting_BestModel_08252.joblib", compress=5)
joblib.dump(s, "Stacking_BestModel_08270.joblib", compress=5)