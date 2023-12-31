from numpy import *
from matplotlib.pyplot import *

# load train set data minus targets and normalize

converter = lambda s: float(s) / 255
converters = {i: converter for i in range(64 * 64 + 1)}
all_data = loadtxt(
    open("../datasets/medical_mnist.csv", "rb"),
    delimiter=",",
    usecols=range(1, 64 * 64 + 1),
    converters=converters,
)

all_targets_data = loadtxt(
    open("../datasets/medical_mnist.csv", "rb"),
    delimiter=",",
    usecols=range(0, 1),
    dtype=int64,
)


def test_accuracy(data, targets):
    correct = 0
    # total number of data being tested
    total = shape(data)[0]

    for datum in range(shape(data)[0]):
        # feed forward

        # to hidden layer
        hj_prime = dot(weights_hidden, data[datum])

        # apply relu to all
        hj = vec_relu(hj_prime)

        # add bias
        hj = insert(hj, 0, 1)

        # hidden to output
        ok_prime = dot(weights_output, hj)
        ok = softmax(ok_prime)

        # get prediction
        prediction = argmax(ok)

        target = targets[datum]

        if prediction == target:
            correct += 1

    return correct / total


def confusion(data, targets):
    # initialize confusion matrix
    mat = zeros((6, 6), dtype=int)

    for datum in range(shape(data)[0]):
        # feed forward

        hj_prime = dot(weights_hidden, data[datum])
        hj = vec_relu(hj_prime)

        # bias
        hj = insert(hj, 0, 1)

        ok_prime = dot(weights_output, hj)
        ok = softmax(ok_prime)

        target = targets[datum]

        prediction = argmax(ok)

        for i in range(6):
            for j in range(6):
                if prediction == i and target == j:
                    mat[i, j] += 1

    return mat


def softmax(z):
    return exp(z) / sum(exp(z), axis=0)


def relu(z):
    return max(0, z)


def relu_derivative(z):
    if z > 0:
        return 1
    return 0


vec_relu = vectorize(relu)
vec_relu_derivative = vectorize(relu_derivative)


# for testing
def print_dim(arr, name):
    print(name + ": (" + str(shape(arr)[0]) + ", " + str(shape(arr)[1]) + ")")


prev_test_acc = 0

# hyperparams
epochs = 50
eta = 0.01
momentum = 0.1

print_dim(all_data, "all_data")

# initialize change index array
change = list(range(shape(all_data)[0]))
# randomize data
random.shuffle(change)
all_data = all_data[change, :]
all_targets_data = [all_targets_data[i] for i in change]

data_shape = shape(all_data)[0]
data_targets_shape = shape(all_targets_data)[0]

eighty_idx_data = int(shape(all_data)[0] * 0.8)
eighty_idx_targets_data = int(shape(all_targets_data)[0] * 0.8)

# split the data 80/20 between training and testing

data = all_data[0:eighty_idx_data]
print_dim(data, "data")
targets_data = all_targets_data[0:eighty_idx_targets_data]
print(shape(targets_data)[0])

test = all_data[eighty_idx_data:data_shape]
print_dim(test, "test")
targets_test = all_targets_data[eighty_idx_targets_data:data_targets_shape]
print(shape(targets_test)[0])

# add bias to data
data = concatenate((ones((shape(data)[0], 1)), data), axis=1)
test = concatenate((ones((shape(test)[0], 1)), test), axis=1)

change_data = list(range(shape(data)[0]))

for n_units in [100]:
    train_acc_arr = array([])
    train_epoch_arr = array([])

    test_acc_arr = array([])
    test_epoch_arr = array([])
    print("n_units:" + str(n_units))

    weights_hidden = random.rand(n_units, shape(data)[1]) * 0.1 - 0.05

    weights_output = random.rand(6, n_units + 1) * 0.1 - 0.05

    for epoch in range(0, epochs):
        print("epoch:" + str(epoch))

        # gather accuracy data on test and training set
        train_acc = test_accuracy(data, targets_data)
        test_acc = test_accuracy(test, targets_test)

        train_acc_arr = append(train_acc_arr, train_acc)
        train_epoch_arr = append(train_epoch_arr, epoch)

        test_acc_arr = append(test_acc_arr, test_acc)
        test_epoch_arr = append(test_epoch_arr, epoch)

        # initialize deltas to hold previous value
        deltaw_i2h = zeros((shape(weights_hidden)))
        deltaw_h2o = zeros((shape(weights_output)))

        for datum in range(shape(data)[0]):
            hj_prime = dot(weights_hidden, data[datum])

            hj = vec_relu(hj_prime)

            # add bias
            hj = insert(hj, 0, 1)

            # hidden to output

            ok_prime = dot(weights_output, hj)

            ok = softmax(ok_prime)

            # get target
            target = targets_data[datum]

            # one-hot encoding
            t = zeros(6)
            t[target] = 1

            # backpropagate

            delta_k = t - ok

            error_sum = 0
            for i in range(shape(weights_output)[0]):
                error_sum += weights_output[i] * delta_k[i]

            delta_j = vec_relu_derivative(hj) * error_sum

            deltaw_h2o = eta * outer(delta_k, hj) + momentum * deltaw_h2o

            deltaw_i2h = eta * outer(delta_j[1:], data[datum]) + momentum * deltaw_i2h
            # delta_j[1:] = ignore bias

            weights_output += deltaw_h2o
            weights_hidden += deltaw_i2h

        # randomize data
        random.shuffle(change_data)
        data = data[change_data, :]
        targets_data = [targets_data[i] for i in change_data]

    # generate plot
    plot(train_epoch_arr, train_acc_arr, "r-", label="train")
    plot(test_epoch_arr, test_acc_arr, "c-", label="test")
    legend()
    xlabel("Epoch")
    ylabel("Accuracy")
    title("n_units " + str(n_units))
    show()

mat = confusion(test, targets_test)
print(mat)
print("accuracy:")
print(test_accuracy(test, targets_test))

print(sum(mat, axis=0))
print(sum(mat, axis=1))
precision = diag(mat) / sum(mat, axis=0)

precision_sum = 0
for i in range(shape(precision)[0]):
    precision_sum += precision[i]

macro_precision = precision_sum / 6

recall = diag(mat) / sum(mat, axis=1)

recall_sum = 0
for i in range(shape(recall)[0]):
    recall_sum += recall[i]

macro_recall = recall_sum / 6

print("precision:")
print(precision)

print("macro average precision:")
print(macro_precision)

print("recall:")
print(recall)

print("macro average recall:")
print(macro_recall)
