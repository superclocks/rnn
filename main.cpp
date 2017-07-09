#include "./src/lstm/lstm.h"

#include <iostream>

int main() {
    int seq_len = 4;
    Lstm* lstm = new Lstm(seq_len, seq_len, seq_len, 5);
    double* x = new double[seq_len];
    long* y = new long[seq_len];
    for(int i = 0; i < seq_len; i++){
        x[i] = i;
        y[i] = i + 1;
    }
    lstm->BatchTrain(x, y, 4);
    lstm->DerCheck();
    std::cout << "Hello, World!" << std::endl;
    return 0;
}