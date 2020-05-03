import numpy

def mid_index(arr):
    return int(len(arr) / 2)

def slice_q(arr, index):
    return arr[0:index], arr[index + 1:len(arr)]

def get_iqr(q1, q3):
    return q3[mid_index(q3)] - q1[mid_index(q1)]

def get_outlier_bounds(q1, q3):
    iqr = get_iqr(q1, q3)
    left = q1[mid_index(q1)] - (1.5 * iqr)
    right = q3[mid_index(q3)] + (1.5 * iqr)

    return left, right

def get_outliers(data):
    q1, q3 = slice_q(data, mid_index(data))
    left, right = get_outlier_bounds(q1, q3)

    left_outliers = q1[0:numpy.searchsorted(q1, [left], side='left', sorter=None)[0]]
    right_outliers = q3[numpy.searchsorted(q3, [right], side='left', sorter=None)[0]:len(data)]

    return left_outliers, right_outliers

arr = [10.2, 14.1, 14.4, 14.4, 14.4, 14.5, 14.5, 14.6, 14.7, 14.7, 14.7, 14.9, 15.1, 15.9, 16.4]
l, r = get_outliers(arr)
print(f"Outliers: {l}, {r}")
