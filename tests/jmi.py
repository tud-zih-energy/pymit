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

max_features = 20
selected_features = []

MI = numpy.full([features], numpy.nan, dtype=numpy.float)
for i in range(features):
    MI[i] = pymit.I(D[:,i], Y, bins=[bins,2])

f = MI.argmax()
selected_features.append(f)

print("001 {:0>3d} {}".format(f, MI[f]))

for i in range(1,max_features):
    JMI = numpy.zeros([features], dtype=numpy.float)
    for X_k in range(features):
        if X_k in selected_features:
            continue

        for X_j in selected_features:
            sum1 = pymit.I(D[:,X_j], Y, bins=[bins,2])
            sum2 = pymit.I_cond(D[:,X_k], Y, D[:,X_j], bins=[bins,2,bins])
            JMI[X_k] += sum1 + sum2
    f = JMI.argmax()
    selected_features.append(f)
    print("{:0>3d} {:>3d} {}".format(len(selected_features), f, JMI[f]))

expected_features=[241, 338, 378, 105, 472, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]
assert(expected_features == selected_features)
