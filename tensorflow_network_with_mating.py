from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class ReluDense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([out_features, out_features]), name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([out_features, out_features]), name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b
    
class NeuralNet(Model):
    def __init__(self, X_in, X_out, optimizer, max_random_layers=None, learning_rate=0.9, dropout_prob=0.1):
        super(NeuralNet, self).__init__()
        self.number_of_relu_dense = 0
        self.number_of_vanilla_dense = 0
        self.relu_layers = []
        self.dense_layers = []
        if max_random_layers is None:
            for _ in range(4):
                self.relu_layers.append(ReluDense(X_in, X_out))
                self.number_of_relu_dense += 1
            for _ in range(4):
                self.dense_layers.append(Dense(X_in, X_out))
                self.number_of_vanilla_dense += 1
        else:
            for _ in range(1, random.randint(2, max_random_layers)):
                self.relu_layers.append(ReluDense(X_in, X_out))
                self.number_of_relu_dense += 1
            for _ in range(1, random.randint(2, max_random_layers)):
                self.dense_layers.append(Dense(X_in, X_out))
                self.number_of_vanilla_dense += 1
        self.out = Dense(X_in, X_out)
        self.optimizer = optimizer
        self.dropout_prob = dropout_prob
        self.tape = None
        self.learning_rate = learning_rate

    def dropout(self, X, drop_probability):
        keep_probability = 1 - drop_probability
        mask = np.random.uniform(0, 1.0) < keep_probability
        if keep_probability > 0.0:
            scale = (1/keep_probability)
        else:
            scale = 0.0
        return mask * X * scale

    def call(self, x, train=False):
        if train:
            for layer in self.relu_layers:
                x = layer(x)
                x = self.dropout(x, self.dropout_prob)
            for layer in self.dense_layers:
                x = layer(x)
                x = self.dropout(x, self.dropout_prob)
        else:
            for layer in self.relu_layers:
                x = layer(x)
            for layer in self.dense_layers:
                x = layer(x)
            x = self.out(x)
        return tf.reduce_mean(x, axis=1)

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.call(x, train=True)
            loss = mse(pred, y)
            self.tape = tape
        gradients = self.tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

# naive strategy - could also average weights and bias term instead
class EnsembleNet(Model):
    def __init__(self, net_one, net_two, weight):
        super(EnsembleNet, self).__init__()
        self.net_one = net_one
        self.net_two = net_two
        # weight is two numbers between 0 and 1
        # e.g. [0.5, 0.5]
        # this is the weight on the result of each network
        self.weight = weight
        
    def call(self, x, train=False):
        x_one = self.net_one(x, train=train)
        x_two = self.net_two(x, train=train)
        return x_one * self.weight[0] + x_two * self.weight[1]

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        pred = self.call(x, train=True)
        loss = mse(pred, y)

        gradients_one = self.net_one.tape.gradient(loss, self.net_one.trainable_variables)
        self.net_one.optimizer.apply_gradients(zip(gradients_one, self.net_one.trainable_variables))
        gradients_two = self.net_two.tape.gradient(loss, self.net_two.trainable_variables)
        self.net_two.optimizer.apply_gradients(zip(gradients_two, self.net_two.trainable_variables))

class MateNet(Model):
    def __init__(self, x, y, net_one, net_two, optimizer, weight=None):
        super(MateNet, self).__init__()
        self.net_one = net_one
        self.net_two = net_two
        self.optimizer = optimizer
        if weight is None:
            self.set_weights(x, y)
        else:
            # weight is two numbers between 0 and 1
            # e.g. [0.5, 0.5]
            # this is the weight on the result of each network
            self.weight = weight
        self.combine_nets(x)
        
    def set_weights(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        one_pred = self.net_one.call(x)
        two_pred = self.net_two.call(x)
        one_loss = mse(one_pred, y)
        two_loss = mse(two_pred, y)
        denominator = one_loss + two_loss
        self.weight = [
            tf.cast(one_loss/denominator, tf.float32),
            tf.cast(two_loss/denominator, tf.float32)
        ]

    def combine_nets(self, x):
        optimizer = tf.optimizers.Adam(self.net_one.learning_rate)
        combined_net = NeuralNet(
            x.shape[0], x.shape[1],
            optimizer,
            learning_rate=learning_rate,
            dropout_prob=0.0
        )
        # check to make sure we have the same number of layers in both networks
        # otherwise we cannot guarantee that the weighting scheme will be correct
        assert len(self.net_one.relu_layers) == len(self.net_two.relu_layers)
        assert len(self.net_one.dense_layers) == len(self.net_two.dense_layers)
        for index, relu_layer in enumerate(self.net_one.relu_layers):
            combined_relu_layer = ReluDense(x.shape[0], x.shape[1])
            combined_relu_layer.w = relu_layer.w * self.weight[0]
            combined_relu_layer.w += self.net_two.relu_layers[index].w * self.weight[1]
            combined_relu_layer.b = relu_layer.b * self.weight[0]
            combined_relu_layer.b += self.net_two.relu_layers[index].b * self.weight[1]
            combined_net.relu_layers.append(combined_relu_layer)
        for index, dense_layer in enumerate(self.net_one.dense_layers):
            combined_dense_layer = Dense(x.shape[0], x.shape[1])
            combined_dense_layer.w = dense_layer.w * self.weight[0]
            combined_dense_layer.w += self.net_two.dense_layers[index].w * self.weight[1]
            combined_dense_layer.b = dense_layer.b * self.weight[0]
            combined_dense_layer.b += self.net_two.dense_layers[index].b * self.weight[1]
            combined_net.dense_layers.append(combined_dense_layer)

        combined_net.out.w = self.net_one.out.w * self.weight[0]
        combined_net.out.w += self.net_two.out.w * self.weight[1]
        combined_net.out.b = self.net_one.out.b * self.weight[0]
        combined_net.out.b += self.net_two.out.b * self.weight[1]
        self.combined_net = combined_net
        
    def call(self, x, train=False):
        for layer in self.combined_net.relu_layers:
            x = layer(x)
        for layer in self.combined_net.dense_layers:
            x = layer(x)
        x = self.combined_net.out(x)
        return tf.reduce_mean(x, axis=1)

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.call(x, train=True)
            loss = mse(pred, y)
            self.tape = tape
        gradients = self.tape.gradient(loss, self.combined_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.combined_net.trainable_variables))
        
def mse(x, y):
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    return tf.metrics.MSE(x, y)

def maape(y, y_pred):
    normed_absolute_error = np.abs(y_pred - y) / y
    normed_arctan_abs_error = np.arctan(normed_absolute_error)
    return np.sum(normed_arctan_abs_error)

def train_nets(X, y, neural_nets, num_steps):
    stopped = []
    mse_losses = [[] for _ in range(len(neural_nets))]
    maape_losses = [[] for _ in range(len(neural_nets))]
    for step in range(num_steps):
        for index in range(len(neural_nets)):
            if index in stopped:
                continue
            nn = neural_nets[index]
            nn.step(X, y)
            pred = nn(X_val)
            loss_mse = mse(pred, y_val)
            loss_maape = maape(pred, y_val)
            is_nan = pd.isnull(loss_mse.numpy()) or pd.isnull(loss_maape)
            is_less_than_zero = loss_mse.numpy() < 0 or loss_maape < 0
            if is_nan or is_less_than_zero:
                stopped.append(index)
                continue
            mse_losses[index].append(loss_mse)
            maape_losses[index].append(loss_maape)

    neural_nets = [
        neural_nets[index]
        for index in range(len(neural_nets))
        if index not in stopped
    ]
    mse_losses = [
        mse_losses[index]
        for index in range(len(mse_losses))
        if index not in stopped
    ]
    maape_losses = [
        maape_losses[index]
        for index in range(len(maape_losses))
        if index not in stopped
    ]
    return mse_losses, maape_losses, neural_nets

def select_best_k_networks(losses, k, networks):
    average_loss = []
    last_25_percent_index = int(len(losses[0]) * 0.75)
    for index, loss in enumerate(losses):
        last_25_percent = loss[last_25_percent_index:]
        average_loss.append(
            [index, np.mean(last_25_percent)]
        )
    average_loss = sorted(average_loss, key=lambda t: t[1])
    average_loss_index = [ave_loss[0] for ave_loss in average_loss[:k]]
    return [
        networks[index] for index in average_loss_index
    ]
    
        
        
if __name__ == '__main__':
    X = pd.read_csv("X.csv")
    y = np.load("y.npy")
    X = X.values
    X_val = pd.read_csv("X_val.csv")
    X_val = X_val.values
    y_val = np.load("y_val.npy")
    flip_maape = False
    mated_nets = []
    learning_rate = 0.9
    optimizer = tf.optimizers.Adam(learning_rate)
    num_steps = 2
    k_best = 10
    strategy = "average_weights"
    for _ in range(3):
        neural_nets = [NeuralNet(X.shape[0], X.shape[1], optimizer, learning_rate=learning_rate, dropout_prob=0.0)
                       for _ in range(50)]

        mse_losses, maape_losses, neural_nets = train_nets(X, y, neural_nets, num_steps)
        best_mse_nets = select_best_k_networks(mse_losses, k_best, neural_nets)
        best_maape_nets = select_best_k_networks(maape_losses, k_best, neural_nets)
        
        if flip_maape:
            best_maape_nets = best_maape_nets[::-1]
        if strategy == "average_predictions":
            for index in range(len(best_mse_nets)):
                ensemble = EnsembleNet(best_mse_nets[index], best_maape_nets[index], [0.5, 0.5])
                mated_nets.append(ensemble)
        elif strategy == "average_weights":
            for index in range(len(best_mse_nets)):
                ensemble = MateNet(X, y, best_mse_nets[index], best_maape_nets[index], optimizer)
                mated_nets.append(ensemble)

    mse_losses, maape_losses, neural_nets = train_nets(X, y, mated_nets, num_steps)    
    min_mses = []
    min_maapes = []
    for mse_loss in mse_losses:
        min_mses.append(min(mse_loss))
    for maape_loss in maape_losses:
        min_maapes.append(min(maape_loss))
    print("mse", min(min_mses))
    print("maape", min(min_maapes))
