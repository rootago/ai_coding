import numpy as np
import struct
import matplotlib.pyplot as plt

def load_labels(file):
    with open(file,"rb") as f :
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)
def load_images(file):
    with open(file,"rb") as f :
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack('>iiii', data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items,-1)
def make_one_hot(labels,class_nums=10):
    result = np.zeros((len(labels), class_nums))
    for row,col in enumerate(labels):
        result[row][col] = 1
    return result
def get_datas():
    train_datas = load_images("Mnist_data\\train-images.idx3-ubyte") / 255 
    train_label = make_one_hot(load_labels("Mnist_data\\train-labels.idx1-ubyte"))
    test_datas = load_images("Mnist_data\\t10k-images.idx3-ubyte") / 255
    test_label = load_labels("Mnist_data\\t10k-labels.idx1-ubyte")
    return train_datas, train_label, test_datas, test_label
def sigmoid(x):
    return 1/(1+np.exp(-x)) 
def sotfmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1,keepdims=True)
    return ex/sum_ex

class Sigmoid:
    def forward(self, x):
        self.r = sigmoid(x)
        return self.r
    def backward(self, da):
        return da * (self.r * (1-self.r))

class Softmax:
    def forward(self, x):
        self.pre = sotfmax(x)
        return sotfmax(x)
    def backward(self,batch_label):
        return (self.pre-batch_label) / batch_label.shape[0]

class Linear:
    def __init__(self,in_num,out_num):
        self.weight = np.random.normal(0,1,size=(in_num,out_num))
        self.bias = np.zeros(out_num)
    def forward(self,x):
        self.x = x
        return x @ self.weight + self.bias
    def backward(self,G):
        dW = self.x.T @ G
        dX = G @ self.weight.T
        db = np.sum(G,axis=0)
        self.weight = self.weight - alpha * dW
        self.bias = self.bias - alpha * db
        return dX
    
if __name__ == '__main__':
    train_datas, train_label, test_datas, test_label= get_datas()
    epoch = 100
    batch_size = 300
    alpha = 0.05
    hidden_layer_size = 256
    layers = [ # 不用起名字了
        Linear(784, hidden_layer_size),
        Sigmoid(),
        Linear(hidden_layer_size, 10),
        #Sigmoid(),  # 加一层
        #Linear(10, 10), # 加一层  循环里的代码不变
        # ...
        Softmax()
    ]
    batch_time = int(np.ceil(len(train_datas) /batch_size)) 

    for e in range(epoch):
        for batch_index in range(batch_time):
            batch_x = train_datas[batch_index*batch_size:(batch_index+1)*batch_size]
            batch_label = train_label[batch_index*batch_size:(batch_index+1)*batch_size]
            
            x = batch_x
            for layer in layers:
                x = layer.forward(x)

            loss = -np.mean(batch_label * np.log(x))

            G = batch_label
            for layer in layers[::-1]:
                G = layer.backward(G)

        x = test_datas
        for layer in layers:
            x = layer.forward(x)

        pre = np.argmax(x, axis=1)
        accuracy = np.sum(pre == test_label) / len(test_label) 
        print(f"accuracy:{accuracy:.4f}")
        #    input("stop!")
