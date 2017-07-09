//
// Created by zhongchao03 on 2017/6/29.
//
#include "lstm_paras.h"

LstmParas::LstmParas(int m, int n, int q, int p){
    this->m_ = m;
    this->n_ = n;
    this->q_ = q;
    this->p_ = p;

    Wgx_ = MatrixXd::Random(m_, n_);
    Wgh_ = MatrixXd::Random(m_, q_);
    bg_ = MatrixXd::Random(m_, 1);

    Wix_ = MatrixXd::Random(m_, n_);
    Wih_ = MatrixXd::Random(m_, q_);
    bi_ = MatrixXd::Random(m_, 1);

    Wfx_ = MatrixXd::Random(m_, n_);
    Wfh_ = MatrixXd::Random(m_, q_);
    bf_ = MatrixXd::Random(m_, 1);

    Wox_ = MatrixXd::Random(m_, n_);
    Woh_ = MatrixXd::Random(m_, q_);
    bo_ = MatrixXd::Random(m_, 1);

    Why_ = MatrixXd::Random(p_, m_);
    bh_ = MatrixXd::Random(p_, 1);
    //cout<<"Wox:\n" <<Wox_ << endl;
    //cout<<"Why:\n" <<Why_ << endl;
}

LstmParas::~LstmParas(){

}
