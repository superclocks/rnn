//
// Created by zhongchao03 on 2017/6/29.
//

#ifndef RNN_LSTM_PARAS_H
#define RNN_LSTM_PARAS_H

#include "Eigen/Dense"
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;


class LstmParas{
public:
    //g gate paras
    MatrixXd Wgx_;
    MatrixXd Wgh_;
    MatrixXd bg_;
    //i gate paras
    MatrixXd Wix_;
    MatrixXd Wih_;
    MatrixXd bi_;
    //f gate paras
    MatrixXd Wfx_;
    MatrixXd Wfh_;
    MatrixXd bf_;
    //o gate paras
    MatrixXd Wox_;
    MatrixXd Woh_;
    MatrixXd bo_;
    //output layer paras
    MatrixXd Why_;
    MatrixXd bh_;

    //
    int m_; //x(t) h(t) hidden dim
    int n_; //x(t) input dim
    int q_; //h(t) input dim
    int p_; //output dim, dict size
public:
    LstmParas(int m, int n, int q, int p);
    ~LstmParas();
};
#endif //RNN_LSTM_PARAS_H
