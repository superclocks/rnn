//
// Created by zhongchao03 on 2017/6/29.
//

#include "lstm.h"
#include "../utils/math.h"

Lstm::Lstm(int m, int n, int q, int p) {
    lstm_paras_ = new LstmParas(m, n, q, p);
    h_ = MatrixXd::Zero(m, 1);
    s_ = MatrixXd::Zero(m, 1);
}
Lstm::~Lstm() {
    delete lstm_paras_;
}

void Lstm::Train(char* path){

}

void Lstm::BatchTrain(double* data, long* label, int n){
    MatrixXd* x = new MatrixXd(1, n);
    MatrixXd* y = new MatrixXd(1, n);
    for(int i = 0; i < n; i++){
        (*x)(0, i) = data[i];
        (*y)(0, i) = label[i];
    }
    Forward(*x, *y);
    Backward(*x, *y);
    cout<<"der Wgx:\n" <<lstm_paras_->Wgx_ <<endl;
}
void Lstm::Forward(MatrixXd& x, MatrixXd& y){
    MatrixXd zero = MatrixXd::Zero(lstm_paras_->m_, 1);
    s_vec.push_back(zero);
    h_vec.push_back(zero);
    for(int t = 0; t < x.size(); t++){
        int idx = x(0, t);
        MatrixXd g_t =  lstm_paras_->Wgx_.col(idx) + (lstm_paras_->Wgh_) * h_ + (lstm_paras_->bg_);
        MatrixXd g = Math::Tanh(g_t);
        MatrixXd der_g = Math::DerTanh(g_t);
        g_vec.push_back(g);
        der_g_vec.push_back(der_g);

        MatrixXd i_t = lstm_paras_->Wix_.col(idx) + lstm_paras_->Wih_ * h_ + lstm_paras_->bi_;
        MatrixXd i = Math::Sigmoid(i_t);
        MatrixXd der_i = Math::DerSigmoid(i_t);
        i_vec.push_back(i);
        der_i_vec.push_back(der_i);

        MatrixXd f_t = lstm_paras_->Wfx_.col(idx) + lstm_paras_->Wfx_ * h_ + lstm_paras_->bf_;
        MatrixXd f = Math::Sigmoid(f_t);
        MatrixXd der_f = Math::DerSigmoid(f_t);
        f_vec.push_back(f);
        der_f_vec.push_back(der_f);

        MatrixXd o_t = lstm_paras_->Wox_.col(idx) + lstm_paras_->Wox_ * h_ + lstm_paras_->bo_;
        MatrixXd o = Math::Sigmoid(o_t);
        MatrixXd der_o = Math::DerSigmoid(o_t);
        o_vec.push_back(o);
        der_o_vec.push_back(der_o);

        MatrixXd s = g.array().array() * i.array().array() + s_.array().array() * f.array();
        s_vec.push_back(s);

        MatrixXd h_t = s.array().array() * o.array().array();
        MatrixXd h = Math::Sigmoid(h_t);
        MatrixXd der_h = Math::DerSigmoid(h_t);
        h_vec.push_back(h);
        der_h_vec.push_back(der_h);

        MatrixXd y_t =  lstm_paras_->Why_ * h + lstm_paras_->bh_;
        MatrixXd y = Math::Softmax(y_t);
        MatrixXd der_y = Math::DerSoftmax(y_t);
        y_vec.push_back(y);
        der_y_vec.push_back(der_y);
        //cout<<"t: " <<t <<" " <<y << endl;
        s_ = s;
        h_ = h;
    }

}
void Lstm::Backward(MatrixXd& x, MatrixXd& y){
    long T = x.size() - 1;
    MatrixXd button_ds_t = MatrixXd::Zero(lstm_paras_->m_, 1);
    MatrixXd button_dh_t = MatrixXd::Zero(lstm_paras_->m_, 1);
    for(int t = T; t >=0; t--){
        int label = y(0, T);
        MatrixXd top_dh_t = (-1.0 / y_vec[T](T, 0) * der_y_vec[T](label, 0) * lstm_paras_->Why_.row(label)).transpose() + button_dh_t;
        MatrixXd top_ds_t = MatrixXd::Zero(top_dh_t.size(), 1) + button_ds_t;

        MatrixXd ds_t = top_dh_t.array().array() * der_h_vec[t].array().array() * o_vec[t].array().array() + top_ds_t.array().array();
        MatrixXd do_t = top_dh_t.array().array() * der_h_vec[t].array().array() * s_vec[t].array().array();
        MatrixXd di_t = g_vec[t].array().array() * ds_t.array().array();
        MatrixXd dg_t = i_vec[t].array().array() * ds_t.array().array();
        MatrixXd df_t = s_vec[t].array().array() * ds_t.array().array();

        MatrixXd di_input_t = der_i_vec[t].array().array() * di_t.array().array();
        MatrixXd df_input_t = der_f_vec[t].array().array() * df_t.array().array();
        MatrixXd do_input_t = der_o_vec[t].array().array() * do_t.array().array();
        MatrixXd dg_input_t = der_g_vec[t].array().array() * dg_t.array().array();

        //求解各个门上的导数
        int xc = x.col(t).value();
        lstm_paras_->Wix_.col(xc).array() += di_input_t.array();
        lstm_paras_->Wih_.array().array() += (di_input_t * h_vec[t].transpose()).array().array();
        lstm_paras_->bi_.array().array() += di_input_t.array().array();

        lstm_paras_->Wfx_.col(xc).array().array() += df_input_t.array().array();
        lstm_paras_->Wfh_.array().array() += (df_input_t * h_vec[t].transpose()).array().array();
        lstm_paras_->bf_.array().array() += df_input_t.array().array();

        lstm_paras_->Wox_.col(xc).array().array() += do_input_t.array().array();
        lstm_paras_->Woh_.array().array() += (do_input_t * h_vec[t].transpose()).array().array();
        lstm_paras_->bo_.array().array() += do_input_t.array().array();

        lstm_paras_->Wgx_.col(xc).array().array() += dg_input_t.array().array();
        lstm_paras_->Wgh_.array().array() += (dg_input_t * h_vec[t].transpose()).array().array();
        lstm_paras_->bg_.array().array() += dg_input_t.array().array();

        //求解输出层导数
        y_vec[t](xc, 0) -= 1.0;
        lstm_paras_->Why_.array().array() += (y_vec[t] * h_vec[t + 1].transpose()).array().array();
        lstm_paras_->bh_.array().array() += y_vec[t].array().array();

        //
        MatrixXd dxh = MatrixXd::Zero(lstm_paras_->m_, 1);
        dxh.array().array() += (lstm_paras_->Wih_ * di_input_t).array().array();
        dxh.array().array() += (lstm_paras_->Wfh_ * df_input_t).array().array();
        dxh.array().array() += (lstm_paras_->Woh_ * df_input_t).array().array();
        dxh.array().array() += (lstm_paras_->Wgh_ * df_input_t).array().array();


        button_ds_t = top_ds_t.array().array() * f_vec[t].array().array();
        button_dh_t = dxh;

    }
}
void Lstm::Clean() {
    g_vec.clear();
    der_g_vec.clear();

    i_vec.clear();
    der_i_vec.clear();

    f_vec.clear();
    der_f_vec.clear();

    o_vec.clear();
    der_o_vec.clear();

    s_vec.clear();
    h_vec.clear();
    der_h_vec.clear();

    y_vec.clear();
    der_y_vec.clear();
}
void Lstm::Evaluate(){

}
double Lstm::LogLoss(MatrixXd &x, MatrixXd &y) {
    int l = y.cols();
    double loss = 0.0;
    for(int i = 0; i < l; i++){
        int label = y(0, i);
        double p = y_vec[i](label, 0);
        //cout<<p << endl;
        loss += log(p);
    }
    return -loss;
}
void Lstm::DerCheck() {
    //Wgx数值梯度
    int l = 4;
    MatrixXd* x = new MatrixXd(1, l);
    MatrixXd* y = new MatrixXd(1, l);
    for(int i = 0; i < l; i++){
        (*x)(0, i) = i;
        (*y)(0, i) = i + 1;
    }


    int m = lstm_paras_->Wgx_.rows();
    int n = lstm_paras_->Wgx_.cols();

    double delta = 0.000001;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            //cout<<"before Wgx:\n" <<lstm_paras_->Wgx_<< endl;
            Forward(*x, *y);
            double logloss_before = LogLoss(*x, *y);
            Clean();
            lstm_paras_->Wgx_(i, j) += delta;
            //cout<<"after Wgx:\n" <<lstm_paras_->Wgx_<< endl;
            Forward(*x, *y);
            double logloss_after = LogLoss(*x, *y);
            lstm_paras_->Wgx_(i, j) -= delta;
            cout<< (logloss_after - logloss_before) / delta << " ";
        }
        cout<< endl;
    }
}