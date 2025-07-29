import numpy as np
import struct
import matplotlib.pyplot as plt

# 加载数据
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
    result = np.zeros((len(labels), class_nums)) ## 里面是个元组哈
    for row,col in enumerate(labels):
        result[row][col] = 1
    return result

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sotfmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, # 按行求和
                     keepdims=True) # 保持原本形状不变
    return ex/sum_ex

if __name__ == '__main__':
    train_datas = load_images("Mnist_data\\train-images.idx3-ubyte") / 255 # 255 是归一化
    train_label = make_one_hot(load_labels("Mnist_data\\train-labels.idx1-ubyte"))

    test_datas = load_images("Mnist_data\\t10k-images.idx3-ubyte") / 255
    test_label = load_labels("Mnist_data\\t10k-labels.idx1-ubyte")
    print("")

    # a test : 
    # t = train_datas[1009]
    # plt.imshow(t.reshape(28,28))
    # plt.show()

    epoch = 100
    batch_size = 300 # 一次拿几张图
    alpha = 0.05
    
    hidden_layer_size = 256

    w1 = np.random.normal(0,1,size=(784,hidden_layer_size))
    b1 = np.zeros(hidden_layer_size)

    w2 = np.random.normal(0,1,size=(hidden_layer_size,10))
    b2 = np.zeros(10)

    batch_time = int(np.ceil(len(train_datas) /batch_size)) # 训练数据集的批次数，遍历完所有的照片
    for e in range(epoch):
        for batch_index in range(batch_time):
            batch_x = train_datas[batch_index*batch_size:(batch_index+1)*batch_size]  # n * m 
            batch_label = train_label[batch_index*batch_size:(batch_index+1)*batch_size] # n * k

            ## 前向传播
            h = batch_x @ w1 + b1
            a = sigmoid(h)
            p = a @ w2 + b2
            pre = sotfmax(p)
            loss = -np.mean(batch_label * np.log(pre))  # 多元交叉熵,只有前面 y * ln(y_hat)
            # print(loss)

            ## 反向传播
            G2 = (pre - batch_label) / batch_size  # 计算梯度，防止梯度爆炸
            dW2 = a.T @ G2 
            db2 = np.sum(G2, axis=0)
            da = G2 @ w2.T
            G1 = da * (a * (1 - a)) ## 不是矩阵乘法，而是哈达玛积
            dW1 = batch_x.T @ G1
            db1 = np.sum(G1, axis=0)

            w1 = w1 - alpha * dW1
            b1 = b1 - alpha * db1
            w2 = w2 - alpha * dW2
            b2 = b2 - alpha * db2

        h = test_datas @ w1 + b1  # 预测
        a = sigmoid(h)
        p = a @ w2 + b2
        pre = sotfmax(p)
        pre = np.argmax(pre, axis=1) # 取最大概率的索引
        accuracy = np.sum(pre == test_label) / len(test_label) 
        # 准确率，猜对了就是1，猜错了就是0，加起来就是猜对的数量
        # 并 除以总样本数，就是准确率
        print(f"accuracy:{accuracy:.4f}")
        #    input("stop!")







