from Red import *
from sklearn.datasets import fetch_mldata
import sklearn.metrics as met

a = multilayer_perceptron(784, 10, [16, 16], activation='sigmoid')
mnist = fetch_mldata('MNIST original', data_home='~/Documents/Universidad/GCOM/Pr4')
data = mnist.data
target = mnist.target
mis_digitos = np.hstack([data, target.reshape(data.shape[0], 1)])
np.random.shuffle(mis_digitos)
ochenta = int(np.rint(mis_digitos.shape[0] * 0.8))
train = mis_digitos[:ochenta, :]
train_x = train[:, :train.shape[1] - 1].T
train_t = train[:, train.shape[1] - 1:]
test = mis_digitos[ochenta:, :]
test_x = test[:, :test.shape[1] - 1].T
test_t = test[:, test.shape[1] - 1:]
a.train(train_x, train_t, 0.1)

res = []
for elem in test_x.T:
    m = a.classify(elem)
    res.append(np.argmax(m))

print(met.precision_score(test_t, res, average='micro'))
print(met.f1_score(test_t, res, average='micro'))
print(met.recall_score(test_t, res, average='micro'))
print(met.accuracy_score(test_t, res))