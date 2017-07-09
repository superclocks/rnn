//
// Created by zhongchao03 on 2017/6/29.
//

#ifndef RNN_LSTM_H
#define RNN_LSTM_H

#include "lstm_paras.h"
#include <stdio.h>
#include <math.h>

class Lstm{
private:
    LstmParas* lstm_paras_;
    MatrixXd h_;
    MatrixXd s_;

    vector<MatrixXd> g_vec;
    vector<MatrixXd> der_g_vec;

    vector<MatrixXd> i_vec;
    vector<MatrixXd> der_i_vec;

    vector<MatrixXd> f_vec;
    vector<MatrixXd> der_f_vec;

    vector<MatrixXd> o_vec;
    vector<MatrixXd> der_o_vec;

    vector<MatrixXd> s_vec;
    vector<MatrixXd> h_vec;
    vector<MatrixXd> der_h_vec;

    vector<MatrixXd> y_vec;
    vector<MatrixXd> der_y_vec;
public:
    Lstm(int m, int n, int q, int p);
    void Train(char* path);
    void BatchTrain(double* data, long* label, int n = 1);
    void Forward(MatrixXd& x, MatrixXd& y);
    void Backward(MatrixXd& x, MatrixXd& y);
    void Clean();
    void Evaluate();

    void DerCheck();

    double LogLoss(MatrixXd& x, MatrixXd& y);
    ~Lstm();
};

#endif //RNN_LSTM_H
