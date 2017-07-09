//
// Created by zhongchao03 on 2017/6/30.
//

#ifndef RNN_MATH_H
#define RNN_MATH_H

#include "../../include/eigen-3.3.4/Eigen/Dense"
#include "../../include/eigen-3.3.4/unsupported/Eigen/MatrixFunctions"
using namespace Eigen;
class Math{
public:
    static MatrixXd Sigmoid(MatrixXd& data);
    static MatrixXd DerSigmoid(MatrixXd& data);

    static MatrixXd Tanh(MatrixXd& data);
    static MatrixXd DerTanh(MatrixXd& data);

    static MatrixXd Softmax(MatrixXd& data);
    static MatrixXd DerSoftmax(MatrixXd& data);

};
#endif //RNN_MATH_H
