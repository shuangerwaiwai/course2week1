import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils
import reg_utils
import gc_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)

def model(X, Y, learning_rate = 0.01, num_iteration=15000, print_cost = True, initialization="he", is_plot = True):
    grads={}
    costs=[]
    m = X.shape[1]
    layers_dim = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dim)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dim)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dim)
    else:
        print("error initialize!exit!")
        exit

    for i in range(0, num_iteration):
        a3, cache = init_utils.forward_propagation(X, parameters)

        cost = init_utils.compute_loss(a3, Y)

        grads = init_utils.backward_propagation(X, Y, cache)

        parameters = init_utils.update_parameters(parameters, grads, learning_rate)

        if i%1000 == 0:
            costs.append(cost)
            if print_cost:
                print(str(i) + "th iteration, cost is:" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('ietrations (per hundreds)')
        plt.title('learning rate = ' + str(learning_rate))
        plt.show()

    return parameters
def initialize_parameters_zeros(layers_dims):
    parameters={}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_zeros([3,2,1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

#parameters = model(train_X, train_Y, initialization= "zeros", is_plot=True)

# print("train_set:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
#
# print("test_set:")
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print("prediction_train = " + str(predictions_train))
# print("prediction_test = " + str(predictions_test))
#
# plt.title("Model with Zeros initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*10
        parameters["b" + str(l)] = np.random.randn(layers_dims[l], 1)

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_random([3, 2, 1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

# parameters = model(train_X, train_Y, initialization="random", is_plot=True)
# print("train_set:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print(predictions_train)
# print(predictions_test)
#
# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters={}
    L = len(layers_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])* np.sqrt(2/layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# parameters = initialize_parameters_random([2, 4, 1])
# print("W1=" + str(parameters["W1"]))
# print("b1=" + str(parameters["b1"]))
# print("W2=" + str(parameters["W2"]))
# print("b2=" + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization="he", is_plot=True)
print("train_set:")
prediction_train = init_utils.predict(train_X, train_Y, parameters)
print("test_set:")
prediction_test = init_utils.predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
