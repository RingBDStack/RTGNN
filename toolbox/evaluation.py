from scipy import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

def svm_kmeans_test(X, y, n_clusters, test_sizes=(0.4, 0.8), repeat=20):
    random_states = [10000 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    nmi_list = []
    ari_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append(np.max(macro_f1_list))
        result_micro_f1_list.append(np.max(micro_f1_list))
        if test_size == 0.4:
            for i in range(repeat):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
                kmeans = KMeans(n_clusters=n_clusters)
                y_pred = kmeans.fit_predict(X_test)
                nmi_score = normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic')
                ari_score = adjusted_rand_score(y_test, y_pred)
                nmi_list.append(nmi_score)
                ari_list.append(ari_score)
    return result_macro_f1_list, result_micro_f1_list, np.max(nmi_list), np.max(ari_list)

def evaluate_results_nc(embeddings, labels, num_classes):
    print('test')
    svm_macro_f1_list, svm_micro_f1_list, nmi, ari = svm_kmeans_test(embeddings, labels, num_classes)
    print('SVM:Macro-F1: ' + ', '.join(['{:.6f} ({:.1f})'.format(macro_f1, train_size) for macro_f1, train_size in
                                    zip(svm_macro_f1_list, [0.6, 0.2])]))
    print('SVM:Micro-F1: ' + ', '.join(['{:.6f} ({:.1f})'.format(micro_f1, train_size) for micro_f1, train_size in
                                    zip(svm_micro_f1_list, [0.6, 0.2])]))
    print('K-means:NMI: ' + '{:.6f}'.format(nmi))
    print('K-means:ARI: ' + '{:.6f}'.format(ari))
    return svm_macro_f1_list, svm_micro_f1_list, nmi, ari

def svm_kmeans_test_DBLP(X, y, test_sizes=(0.4, 0.8), repeat=20):
    random_states = [10000 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append(np.max(macro_f1_list))
        result_micro_f1_list.append(np.max(micro_f1_list))
    return result_macro_f1_list, result_micro_f1_list

def evaluate_results_nc_DBLP(embeddings, labels):
    print('test')
    svm_macro_f1_list, svm_micro_f1_list = svm_kmeans_test_DBLP(embeddings, labels)
    print('SVM:Macro-F1: ' + ', '.join(['{:.6f} ({:.1f})'.format(macro_f1, train_size) for macro_f1, train_size in
                                    zip(svm_macro_f1_list, [0.6, 0.2])]))
    print('SVM:Micro-F1: ' + ', '.join(['{:.6f} ({:.1f})'.format(micro_f1, train_size) for micro_f1, train_size in
                                    zip(svm_micro_f1_list, [0.6, 0.2])]))
    return svm_macro_f1_list, svm_micro_f1_list