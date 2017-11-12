#from lstm import Lstm
from lstm_new import LstmCell, Lstm
import numpy as np
if __name__ == '__main__':
    #num_input = 10
    #num_hidden = 6
    #num_embeding = 4
    #num_layer = 1
    #num_in_dict = 10 
    #num_out_dict = 10    
    #lstm = Lstm(num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict)
    #X = [[np.random.random([10, 1]), np.random.random([10, 1]), 
          #np.random.random([10, 1]), np.random.random([10, 1]),
          #np.random.random([10, 1]), np.random.random([10, 1]),
          #np.random.random([10, 1]), np.random.random([10, 1])
          #]]
    #Y = [[1, 2, 3, 4, 5,6, 7, 8 ]]
    #X[0][0][1, 0] = 1
    ##X[0][1][2, 0] = 1

    #lstm.Train(X, Y)
    
    num_input = 10
    num_hidden = 10
    num_embeding = 4
    num_layer = 1
    num_in_dict = 10 
    num_out_dict = 10    
    cell0 = LstmCell(num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict)  
    cell1 = LstmCell(num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict)
    net = [cell0, cell1]
    X = [[np.random.random([10, 1]), np.random.random([10, 1])
        ]]
    Y = [[np.random.random([10, 1]), np.random.random([10, 1])
        ]]
    lstm = Lstm(net, X, Y)
    lstm.Train()
