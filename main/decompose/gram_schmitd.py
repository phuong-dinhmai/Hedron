import numpy


def gs_cofficient(v1, v2):
    return numpy.dot(v2, v1) / numpy.dot(v1, v1)


def multiply(cofficient, v):
    return [x * cofficient for x in v]


def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2), v1)


def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y:
            proj_vec = proj(inY, X[i])
            temp_vec = temp_vec - proj_vec
        Y.append(temp_vec)
    return Y
