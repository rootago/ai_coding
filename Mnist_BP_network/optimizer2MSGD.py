import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
#import typing_extensions
import torch.optim as opt # SGD,Adam,Adamax,AdamW,Adagrad...

print(torch.__version__)
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
    x = np.clip(x, -500, 10000000) # 防止溢出,数值剪枝，防止梯度爆炸
    return 1/(1+np.exp(-x))  # 对于e^(-x) x 的上限随便，下限绝对值应较小
def sotfmax(x):
    # x = np.clip(x, -1000000, 500) # 防止溢出,数值剪枝，防止梯度爆炸，但对于softmax来说这个方法很不好
    # 所有元素同时“负数标准化”， 减去最大值
    # x为矩阵,应对每行数据做“负数标准化”，注意keepdims保持维度，否则会降维，要用求出的max补充
    ex = np.exp(x - np.max(x,axis=1,keepdims=True)) # 对于e^(-x) x 的下限随便，上限绝对值应较小
    sum_ex = np.sum(ex, axis=1,keepdims=True)
    result = ex / sum_ex
    result = np.clip(result, 1e-10, 1) # 防止log(0)下溢
    return result

class Sigmoid:
    def forward(self, x):
        self.r = sigmoid(x)
        return self.r
    
    def __call__(self,x):
        return self.forward(x)
    def backward(self, da):
        return da * (self.r * (1-self.r))

class Softmax:
    def forward(self, x):
        self.pre = sotfmax(x)
        return sotfmax(x)
    def __call__(self,x):
        return self.forward(x)
    def backward(self,batch_label):
        return (self.pre-batch_label) / batch_label.shape[0]

class Linear:
    def __init__(self,in_num,out_num):
        self.weight = np.random.normal(0,1,size=(in_num,out_num))
        self.bias = np.zeros(out_num)
        self.mu = 0.95
        self.Vt = 0
        self.Vb = 0
    def forward(self,x):
        self.x = x
        return x @ self.weight + self.bias
    def __call__(self,x):
        return self.forward(x)
    def backward(self,G):
        dW = self.x.T @ G
        dX = G @ self.weight.T
        db = np.sum(G,axis=0)
        # 这就是optimizer，随机梯度下降优化器(SGD)
        # self.weight = self.weight - alpha * dW
        # self.bias = self.bias - alpha * db
        # MSGD 
        self.Vt = self.mu * self.Vt - alpha * dW # V(t) -> V(t+1)
        self.Vb = self.mu * self.Vb - alpha * db
        
        self.weight = self.weight + self.Vt
        self.bias = self.bias + self.Vb

        return dX

class MyModel:
    def __init__(self, layers):
        self.layers = layers
    def forward(self,x,label = None):
        for layer in self.layers:
            x = layer(x)
        self.x = x
        if label is not None:
            self.label = label
            loss = -np.mean(label * np.log(x))  # 如果x太小，可能被认为是0，导致 log(x) 值为负无穷大，下溢
            # x太小是softmax输出太小导致的

            return loss
    
    def __call__(self,x,label=None):
        return self.forward(x,label)

    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)
        return G
    
if __name__ == '__main__':
    train_datas, train_label, test_datas, test_label= get_datas()
    epoch = 100
    batch_size = 300
    alpha = 0.05
    hidden_layer_size = 256
    model = MyModel(
        [
            Linear(784, hidden_layer_size),
            Sigmoid(),
            Linear(hidden_layer_size, 10),
            Softmax()
        ]
    )
    batch_time = int(np.ceil(len(train_datas) /batch_size)) 

    for e in range(epoch):
        for batch_index in range(batch_time):
            batch_x = train_datas[batch_index*batch_size:(batch_index+1)*batch_size]
            batch_label = train_label[batch_index*batch_size:(batch_index+1)*batch_size]
            
            x = batch_x
            G = batch_label

            loss = model(x,G)

            G = model.backward() 

        x = test_datas
        model(x)

        pre = np.argmax(model.x, axis=1)
        accuracy = np.sum(pre == test_label) / len(test_label) 

        print(f"epoch:{e+1}, loss:{loss:.4f}")
        print(f"accuracy:{accuracy:.4f}")
        print("--------------------------------------------------")
        #    input("stop!")
        # alpha = 0.05
        #  SGD : epoch =10 accu = 0.821
        # MSGD : epoch =10 accu = 0.848 mu = 0.3
        # MSGD : epoch =10 accu = 0.838 mu = 0.2
        # MSGD : epoch =10 accu = 0.853 mu = 0.4

        # MSGD : epoch =10 accu = 0.927 mu = 1  
        # 损失函数过大，精度小幅度上下震荡，梯度爆炸，数据溢出，12轮时精度一下子从0.92降至0.098，程序真正崩溃
        # 0.098 约为0.1,也就是说模型开始随便乱猜，总有10%的概率猜对
        # 收敛速度过于快,碰到了softmax和sigmoid的溢出情况，导致了梯度爆炸现象，最后导致损失函数不可计量（NaN）
        # 其实是三个问题：sigmoid,softmax,loss都出现了溢出

        # MSGD : epoch =10 accu = 0.913 mu = 0.9（数据正常）
        # 尚可进一步二分调试，然意义不大

        # MSGD : epoch =10 accu = 0.927 mu = 1 （clip修剪优化sigmoid） 
        # 然而我们会发现修建结果会严重影响模型的精度
        #选的不好，要么剪裁掉 important feature，要么解决不了梯度爆炸问题，实际上你修改的是原始分布
        # 对于sigmoid来说，把很大的x改成100对sigmoid的输出结果几乎没有影响，可以参考其函数图像，最终100和很大值基本都逼近于1
        # 对于softmax来说，把很大的x改成100对softmax的输出结果有较大的影响，e^x会放大数据之间的差距，所以softmax的结果是把x向量各个数值拉开，然后就导致输出结果有较大的影响，影响到反向传播的参数调整，最终导致了模型的严重变形，导致了精度的严重下降

        # MSGD : epoch =10 accu = 0.903 mu = 1 （clip修剪优化sigmoid，负向标准化优化softmax）
        # loss 仍然溢出

        # MSGD : epoch =10 accu = 0.9224 mu = 1 （clip修剪优化sigmoid，负向标准化优化softmax，clip修剪softmax输出）
        # accuracy 震荡 

        # MSGD : epoch =10 accu = 0.9277 mu = 0.95 （clip修剪优化sigmoid，负向标准化优化softmax，clip修剪softmax输出）
        # 可以继续二分调参，但是意义不大，mu建议还是不要为1