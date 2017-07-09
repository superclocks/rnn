//
// Created by zhongchao03 on 2017/6/30.
//

#include "math.h"

MatrixXd Math::Sigmoid(MatrixXd& data) {
    return  1.0 / (1.0 + data.array().array().exp());
}

MatrixXd Math::DerSigmoid(MatrixXd& data) {
    MatrixXd sigmoid =  Sigmoid(data);
    return sigmoid.array().array() * (1.0 - sigmoid.array().array());
}

MatrixXd Math::Tanh(MatrixXd& data) {
    MatrixXd a = data.array().array().exp();
    MatrixXd b = 1.0 / a.array().array();
    MatrixXd c = (a - b);
    MatrixXd d = (a + b);
    return c.array().array() / d.array().array();
}

MatrixXd Math::DerTanh(MatrixXd& data) {
    MatrixXd tanh = Tanh(data);
    return 1.0 - tanh.array().array() * tanh.array().array();
}

MatrixXd Math::Softmax(MatrixXd& data) {
    MatrixXd a = data.array().array().exp();
    double v = a.sum();
    return a.array().array() / v;
}

MatrixXd Math::DerSoftmax(MatrixXd& data) {
    MatrixXd softmax = Softmax(data);
    return softmax.array().array() * (1.0 - softmax.array().array());
}