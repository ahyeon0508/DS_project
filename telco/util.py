import numpy as np

def get_one_fold(data, turn, fold):
    tot_length = len(data)
    each = int(tot_length/fold)
    mask = np.array([True if each*turn <= i < each*(turn+1)
                        else False
            for i in list(range(tot_length)) ])
    return data[~mask], data[mask]

def runCV(clf, shuffled_data, shuffled_labels, fold, isAcc=True):
    from sklearn.metrics import precision_recall_fscore_support
    results = []
    for i in range(fold):
        train_data, test_data = get_one_fold(shuffled_data, i, fold=fold)
        train_labels, test_labels = get_one_fold(shuffled_labels, i, fold=fold)
        clf = clf.fit(train_data, train_labels)
        pred = clf.predict(test_data)
        correct = pred==test_labels
        if isAcc:
            acc = sum([1 if x == True else 0 for x in correct])/len(correct)
            results.append(acc)
        else:
            results.append(precision_recall_fscore_support(pred, test_labels))
    return results