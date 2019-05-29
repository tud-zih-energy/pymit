import numpy
import pymit

MADELON_TRAIN = "./MADELON/madelon_train.data"
MADELON_TRAIN_LABELS = "./MADELON/madelon_train.labels"

data_raw = numpy.loadtxt(MADELON_TRAIN,dtype=numpy.float)
labels = numpy.loadtxt(MADELON_TRAIN_LABELS,dtype=numpy.float)

X = data_raw
Y = labels
bins = 10

[tmp, features] = X.shape
D = numpy.zeros([tmp, features])

for i in range(features):
    N, E = numpy.histogram(X[:,i], bins=bins)
    D[:,i] = numpy.digitize(X[:,i], E, right=False)

max_features = 200
selected_features = []
j_h = 0
hjmi = None

for i in range(0,max_features):
    JMI = numpy.zeros([features], dtype=numpy.float)
    for X_k in range(features):
        if X_k in selected_features:
            continue
        jmi_1 = pymit.I(D[:,X_k], Y, bins=[bins,2])
        jmi_2 = 0
        for X_j in selected_features:
            tmp1 = pymit.I(D[:,X_k], D[:,X_j], bins=[bins,bins])
            tmp2 = pymit.I_cond(D[:,X_k], D[:,X_j], Y, bins=[bins,bins,2])
            jmi_2 += tmp1 - tmp2
        if len(selected_features) == 0:
            JMI[X_k] += j_h + jmi_1
        else:
            JMI[X_k] += j_h + jmi_1 - jmi_2/len(selected_features)
    f = JMI.argmax()
    j_h = JMI[f]
    if (hjmi == None) or ((j_h - hjmi)/hjmi > 0.03):
        hjmi = j_h
        selected_features.append(f)
        print("{:0>3d} {:>3d} {}".format(len(selected_features), f, j_h))
    else:
        break    

expected_features=[241, 338, 378, 105, 472, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]
assert(expected_features == selected_features)
