import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas_datareader.stooq as stooq
import pennylane as qml

class StockCV:
    input = None
    output =None
    weights = []

    def __init__(self,sc) -> None:
        end = datetime.date.today()
        start = end - relativedelta(years=1)
        df = stooq.StooqDailyReader(f'{sc}.jp',start,end).read()[['High','Low']].mean(axis='columns')
        training = df[int(len(df)/2):]
        moving = self.gradient(training.rolling(5).mean()[5:])
        var = training.rolling(5).mean()[5:-1]
        self.input = np.zeros(2*len(moving)).reshape(2,len(moving))
        self.input[0,:] = moving
        self.input[1,:] = var
        self.output = self.gradient(training[:-5])

    def circuit(self, param,input):
        qml.DisplacementEmbedding(input,wires=[0,1])
        qml.CVNeuralNetLayers(*param,wires=[0,1])
        return qml.expval(qml.NumberOperator([0]))

    def cost(self,param):
        print(param)
        summation = 0
        circ = qml.QNode(self.circuit,qml.device("default.qubit",wires=2, shots=1000))
        for i in range(len(self.input)):
            summation += (circ(param,self.input[i]) - self.output[i])**2
        cost_val = summation / len(self.input)
        print(cost_val)
        return cost_val

    def train(self):
        shapes = qml.CVNeuralNetLayers.shape(n_layers=1,n_wires=2)
        print(shapes)
        self.weights = [np.random.random(shape) for shape in shapes]
        optimizer = qml.AdamOptimizer()
        for i in range(50):
            self.weights = optimizer.step(self.cost,self.weights)

    def gradient(self,input):
        return (input[:-1].values - input[1:].values) / input[1:].values

