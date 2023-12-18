import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from load_data import get_xinjiang_raw, get_confusion_matrix, get_daqing_raw, get_blind, get_blind_raw
from util import write_file, save_matrix, parse_arguments
args = parse_arguments()
if args.dataset == "daqing":
    X_train, X_test, y_train, y_test,data_trainloader,test_loader = get_daqing_raw(args.data_path1)
    path = 'datasave/daqing/'
elif args.dataset == "xinjiang":
    X_train, X_test, y_train, y_test,data_trainloader,test_loader = get_xinjiang_raw(args.data_path2)
    path = 'datasave/xinjiang/'
elif args.dataset == "blind":
    X_train, X_test, y_train, y_test,data_trainloader,test_loader = get_blind_raw(args.data_path1)
    path = 'datasave/blind/'
else:
    raise ValueError("Invalid dataset name")
data = X_train
label = y_train
test_X = X_test
y = y_test
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
def train_and_evaluate_classifier(classifier, data, label, test_X, y,save_path):
    classifier.fit(data, label)
    predicted = classifier.predict(test_X)
    write_file(save_path, predicted)
    accuracy = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, average='macro')
    recall = recall_score(y, predicted, average='macro')
    f1 = f1_score(y, predicted, average='macro')
    conf_matrix = get_confusion_matrix(y, predicted)
    conf_matrix_path = path +f'conf_matrix/{classifier.__class__.__name__.lower()}.pkl'
    save_matrix(conf_matrix_path, conf_matrix)
# SVM
clf_svm = svm.SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced')
train_and_evaluate_classifier(clf_svm, data, label, test_X, y,'datasave/daqing/y_pre/svm.txt')
# Random Forest
clf_rf = RandomForestClassifier(n_estimators=5, criterion='gini', max_depth=5, min_samples_split=2,
                                min_samples_leaf=1, max_features=11, bootstrap=True)
train_and_evaluate_classifier(clf_rf, data, label, test_X, y,'datasave/daqing/y_pre/randomforest.txt')
# KNN
clf_knn = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate_classifier(clf_knn, data, label, test_X, y, path+'y_pre/knn.txt')
# Gradient Boosting
clf_gb = GradientBoostingClassifier(n_estimators=5, random_state=244)
train_and_evaluate_classifier(clf_gb, data, label, test_X, y, path+'y_pre/gdbt.txt')
# Decision Tree
clf_dt = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=5, min_samples_split=4,
                                min_samples_leaf=2, max_features=None)
train_and_evaluate_classifier(clf_dt, data, label, test_X, y, path+'y_pre/decisiontree.txt')
# Naive Bayes
clf_nb = GaussianNB()
train_and_evaluate_classifier(clf_nb, data, label, test_X, y, path+'y_pre/naivebayes.txt')
# XGBoost
import xgboost as xgb
params = {
    'objective': 'multi:softmax',
    'num_class': 5,
    'max_depth': 4,
    'eta': 0.001,
    'verbosity': 0
}
dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(test_X, label=y)
model = xgb.train(params, dtrain, num_boost_round=80)
y_pred = model.predict(dtest)
predicted = y_pred.astype(int)
file_path = path+"y_pre/svm.txt"
write_file(file_path,predicted)
accuracy = accuracy_score(y, predicted)
precision = precision_score(y, predicted, average='macro')
recall = recall_score(y, predicted, average='macro')
f1 = f1_score(y, predicted, average='macro')
write_file(path+"y_pre/XGBoost.txt", predicted)
conf_matrix = confusion_matrix(y, predicted)
conf_matrix_path = path + 'conf_matrix/XGBoot.pkl'
save_matrix(conf_matrix_path, conf_matrix)

