import numpy as np
import matplotlib.pyplot as plt
import copy
from active_func import *

class LstmCell:
    def __init__(self, num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict):
        # control paras
        self.num_input_ = num_input
        self.num_hidden_ = num_hidden
        self.num_embeding_ = num_embeding
        self.num_layer_ = num_layer
        self.num_in_dict_ = num_in_dict
        self.num_out_dict_ = num_out_dict
        # model paras
        self.paras_dict = {}
        self.paras_dict['Wgx_'] = np.random.random([self.num_hidden_, self.num_input_])
        self.paras_dict['Wix_'] = np.random.random([self.num_hidden_, self.num_input_])
        self.paras_dict['Wfx_'] = np.random.random([self.num_hidden_, self.num_input_])
        self.paras_dict['Wox_'] = np.random.random([self.num_hidden_, self.num_input_])
    
        self.paras_dict['Wgh_'] = np.random.random([self.num_hidden_, self.num_hidden_])
        self.paras_dict['Wih_'] = np.random.random([self.num_hidden_, self.num_hidden_])
        self.paras_dict['Wfh_'] = np.random.random([self.num_hidden_, self.num_hidden_])
        self.paras_dict['Woh_'] = np.random.random([self.num_hidden_, self.num_hidden_])
    
        self.paras_dict['bg_'] = np.random.random([self.num_hidden_, 1])
        self.paras_dict['bi_'] = np.random.random([self.num_hidden_, 1])
        self.paras_dict['bf_'] = np.random.random([self.num_hidden_, 1])
        self.paras_dict['bo_'] = np.random.random([self.num_hidden_, 1])
    
        self.paras_dict['Why_'] = np.random.random([self.num_out_dict_, self.num_hidden_])
        self.paras_dict['by_'] = np.random.random([self.num_out_dict_, 1])
    def CleanDerInfos(self):
        # der
        for key in self.paras_dict.keys():
            self.paras_dict[key][:] = 0.0
    def CleanForwardInfos(self):
        # forward infos
        self.gt_ = [[]]
        self.dgt_ = [[]]
        self.it_ = [[]]
        self.dit_ = [[]]
        self.ft_ = [[]]
        self.dft_ = [[]]
        self.ot_ = [[]]
        self.dot_ = [[]]
        self.ht_ = [[]]
        self.dht_ = [[]]
        self.yt_ = [[]]
        self.dyt_ = [[]]
        # 
        self.st_ = [np.zeros([self.num_hidden_, 1])]
        self.ht_ = [np.zeros([self.num_hidden_, 1])]
        # backward infos
        self.dLh_ = []
        self.dLh_next_ = []
        self.dLs_ = []
        self.dLs_next_ = []
    def Calculator(self, Wx ,xt, Wh, ht, b):
        r = np.dot(Wx, xt) + np.dot(Wh, ht) + b
        return r
    def Train(self, X, Y):
        for i in xrange(len(X)):
            xi = X[i]
            yi = Y[i]
            self.Forward(xi)
            self.Backward(xi, yi)
            self.GradChecker(xi, yi)
    def Forward(self, xs):
        self.CleanForwardInfos()
        n = len(xs)
        for t in range(1, n + 1):
            xt = xs[t - 1]
            t1 = self.Calculator(self.Wgx_, xt, self.Wgh_, self.ht_[t - 1], self.bg_)
            self.gt_.append(Tanh(t1))
            self.dgt_.append(DerTanh(t1))
            
            t2 = self.Calculator(self.Wix_, xt, self.Wih_, self.ht_[t - 1], self.bi_)
            self.it_.append(Sigmoid(t2))
            self.dit_.append(DerSigmoid(t2))
            
            t3 = self.Calculator(self.Wfx_, xt, self.Wfh_, self.ht_[t - 1], self.bf_)
            self.ft_.append(Sigmoid(t3))
            self.dft_.append(DerSigmoid(t3))
            
            t4 = self.Calculator(self.Wox_, xt, self.Woh_, self.ht_[t - 1], self.bo_)
            self.ot_.append(Sigmoid(t4))
            self.dot_.append(DerSigmoid(t4))
            
            self.st_.append(self.gt_[t] * self.it_[t] + self.st_[t - 1] * self.ft_[t])
            
            self.ht_.append(Tanh(self.st_[t] * self.ot_[t]))
            self.dht_.append(DerTanh(self.st_[t] * self.ot_[t]))
            v = np.dot(self.Why_, self.ht_[t]) + self.by_
            self.yt_.append(Softmax(v))
            #self.dyt_.append(self.DerSoftmax(v))
    # compute time T dLh
    def DLhT(self, yt, t):
        # compute time T
        y_pred = copy.deepcopy(self.yt_[t])
        y_pred[yt] -= 1.0
        dLht = np.dot(self.Why_.T, y_pred)
        self.dLh_.append(dLht)
        # compute for time T - 1
        
        return dLht.reshape([dLht.shape[0], 1])
    # compute time T dLs
    def DLsT(self, yt):
        s_T = self.st_[-1] 
        o_T = self.ot_[-1]
        dLst = self.DLhT(yt) * self.DerSigmoid(s_T * o_T) * o_T
        return dLst
    def StepUpdate(self, t, xt, yt, up_s, up_h):
        dhs = self.dht_[t] * self.ot_[t]
        dho = self.dht_[t] * self.st_[t]
        dLs = up_h * dhs  + up_s
        
        dsWgx = self.it_[t] * self.dgt_[t]
        dsWgh = self.it_[t] * self.dgt_[t] #* self.ht_[t - 1]
        dsbg = self.it_[t] * self.dgt_[t]
        
        dsWix = self.gt_[t] * self.dit_[t]
        dsWih = self.gt_[t] * self.dit_[t] #* self.ht_[t - 1]
        dsbi = self.gt_[t] * self.dit_[t]
        
        dsWfx = self.st_[t - 1] * self.dft_[t]
        dsWfh = self.st_[t - 1] * self.dft_[t] #* self.ht_[t - 1]
        dsbf = self.st_[t - 1] * self.dft_[t]
        
        # computer der
        # g gate paras
        self.dLWgx_ += np.outer(dLs * dsWgx, xt)
        self.dLWgh_ += np.outer(dLs * dsWgh, self.ht_[t - 1])
        self.dLbg_ += dLs * dsbg
        # i gate paras
        self.dLWix_ += np.outer(dLs * dsWix, xt)
        self.dLWih_ += np.outer(dLs * dsWih, self.ht_[t - 1])
        self.dLbi_ += dLs * dsbi
        # f gate paras
        self.dLWfx_ += np.outer(dLs * dsWfx, xt)
        self.dLWfh_ += np.outer(dLs * dsWfh, self.ht_[t - 1])
        self.dLbf_ += dLs * dsbf
        # o gate paras
        dLo = up_h * dho
        self.dLWox_ += np.outer(dLo * self.dot_[t], xt)
        self.dLWoh_ += np.outer(dLo * self.dot_[t], self.ht_[t - 1])
        self.dLbo_ += dLo * self.dot_[t]
        # output paras
        y_pred = copy.deepcopy(self.yt_[t])
        y_pred[yt] -= 1.0
        self.dLWhy_ += np.outer(y_pred, self.ht_[t])
        self.dLby_ += y_pred
        
        # computer next state
        dgh1 = np.dot(self.Wgh_.T, self.dgt_[t])
        dih1 = np.dot(self.Wih_.T, self.dit_[t])
        dfh1 = np.dot(self.Wfh_.T, self.dft_[t])
        doh1 = np.dot(self.Woh_.T, self.dot_[t])
        
        r = self.st_[t] * doh1 + (self.gt_[t] * dih1 + self.it_[t] * dgh1 + self.st_[t - 1] * dfh1) * self.ot_[t]
        self.down_s = dLs * self.ft_[t]
        self.down_h = up_h * self.dht_[t] * r 
        
    def Backward(self, xs, ys):
        self.CleanDerInfos()
        t = type(ys)
        if(not(t is list)):
            ys = list(ys)
        
        up_h = self.DLhT(ys[-1], -1)
        up_s = np.zeros([up_h.shape[0], 1])
        
        n = len(ys)
        xt = xs[n - 1]
        yt = ys[n - 1]
        t = n
        self.StepUpdate(t, xt, yt, up_s, up_h)
        
        for t in range(n - 1, 0, -1):
            xt = xs[t - 1]
            yt = ys[t - 1]
            dlh = self.DLhT(yt, t)
            up_h = self.down_h + dlh
            up_s = self.down_s
            self.StepUpdate(t, xt, yt, up_s, up_h)
    def Update(self):
        pass
    def OneModelParaChecker(self, bptt_w, xs, ys):
        numeric_w = np.zeros_like(bptt_w)
        m, n = numeric_w.shape
        delta = 1e-6
        t = len(xs)
        for i in xrange(m):
            for j in xrange(n):
                self.Forward(xs)
                y0_list = []
                for ti in xrange(1, t + 1):
                    yt = ys[ti - 1]
                    y_0 = np.log(self.yt_[ti][yt])
                    y0_list.append(y_0)
                    
                self.CleanForwardInfos()
                bptt_w[i, j] += delta
                self.Forward(xs)
                y1_list = []
                for ti in xrange(1, t + 1):
                    yt = ys[ti - 1]
                    y_1 = np.log(self.yt_[ti][yt])
                    y1_list.append(y_1)
                numeric_w[i, j] = sum((np.array(y0_list) - np.array(y1_list)) / delta)
                bptt_w[i, j] -= delta
        return numeric_w
    def GradChecker(self, xs, ys):
        def Plot(numeric_w, bptt_w, name):
            fig = plt.figure()
            plt.plot(numeric_w.ravel(), 'r')
            plt.plot(bptt_w.ravel(), 'og')
            plt.title(name)
            plt.legend(['numeric grad', 'bptt grad'])
            plt.savefig('./fig/' + name + '.jpg')
        dLWgx = self.OneModelParaChecker(self.Wgx_, xs, ys)
        Plot(dLWgx, self.dLWgx_, 'dLWgx')
        dLWix = self.OneModelParaChecker(self.Wix_, xs, ys)
        Plot(dLWix, self.dLWix_, 'dLWix')
        dLWfx = self.OneModelParaChecker(self.Wfx_, xs, ys)
        Plot(dLWfx, self.dLWfx_, 'dLWfx')
        dLWox = self.OneModelParaChecker(self.Wox_, xs, ys)
        Plot(dLWox, self.dLWox_, 'dLWox')
        
        dLWgh = self.OneModelParaChecker(self.Wgh_, xs, ys)
        Plot(dLWgh, self.dLWgh_, 'dLWgh')
        dLWih = self.OneModelParaChecker(self.Wih_, xs, ys)
        Plot(dLWih, self.dLWih_, 'dLWih')
        dLWfh = self.OneModelParaChecker(self.Wfh_, xs, ys)
        Plot(dLWfh, self.dLWfh_, 'dLWfh')
        dLWoh = self.OneModelParaChecker(self.Woh_, xs, ys)
        Plot(dLWoh, self.dLWoh_, 'dLWoh')        
        
        dLbg = self.OneModelParaChecker(self.bg_, xs, ys)
        Plot(dLbg, self.dLbg_, 'dLbg')
        dLbi = self.OneModelParaChecker(self.bi_, xs, ys)
        Plot(dLbi, self.dLbi_, 'dLbi')
        dLbf = self.OneModelParaChecker(self.bf_, xs, ys)
        Plot(dLbf, self.dLbf_, 'dLbf')
        dLbo = self.OneModelParaChecker(self.bo_, xs, ys)
        Plot(dLbo, self.dLbo_, 'dLbo')                
        
        dLWhy = self.OneModelParaChecker(self.Why_, xs, ys)
        Plot(dLWhy, self.dLWhy_, 'dLWhy')
        dLby = self.OneModelParaChecker(self.by_, xs, ys)
        Plot(dLby, self.dLby_, 'dLby')
        
       
class Lstm:
    def __init__(self, num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict):
        self.init(num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict)
    
    def init(self, num_input, num_hidden, num_embeding, num_layer, num_in_dict, num_out_dict):
        # control paras
        self.num_input_ = num_input
        self.num_hidden_ = num_hidden
        self.num_embeding_ = num_embeding
        self.num_layer_ = num_layer
        self.num_in_dict_ = num_in_dict
        self.num_out_dict_ = num_out_dict
        # model paras
        self.Wgx_ = np.random.random([self.num_hidden_, self.num_input_])
        self.Wix_ = np.random.random([self.num_hidden_, self.num_input_])
        self.Wfx_ = np.random.random([self.num_hidden_, self.num_input_])
        self.Wox_ = np.random.random([self.num_hidden_, self.num_input_])
    
        self.Wgh_ = np.random.random([self.num_hidden_, self.num_hidden_])
        self.Wih_ = np.random.random([self.num_hidden_, self.num_hidden_])
        self.Wfh_ = np.random.random([self.num_hidden_, self.num_hidden_])
        self.Woh_ = np.random.random([self.num_hidden_, self.num_hidden_])
    
        self.bg_ = np.random.random([self.num_hidden_, 1])
        self.bi_ = np.random.random([self.num_hidden_, 1])
        self.bf_ = np.random.random([self.num_hidden_, 1])
        self.bo_ = np.random.random([self.num_hidden_, 1])
    
        self.Why_ = np.random.random([self.num_out_dict_, self.num_hidden_])
        self.by_ = np.random.random([self.num_out_dict_, 1])         
        # temp paras
        self.CleanDerInfos()
        self.CleanForwardInfos()
    def CleanDerInfos(self):
        # der
        self.dLWgx_ = np.zeros_like(self.Wgx_)
        self.dLWix_ = np.zeros_like(self.Wix_)
        self.dLWfx_ = np.zeros_like(self.Wfx_)
        self.dLWox_ = np.zeros_like(self.Wox_)
    
        self.dLWgh_ = np.zeros_like(self.Wgh_)
        self.dLWih_ = np.zeros_like(self.Wih_)
        self.dLWfh_ = np.zeros_like(self.Wfh_)
        self.dLWoh_ = np.zeros_like(self.Woh_)
    
        self.dLbg_ = np.zeros_like(self.bg_)
        self.dLbi_ = np.zeros_like(self.bi_)
        self.dLbf_ = np.zeros_like(self.bf_)
        self.dLbo_ = np.zeros_like(self.bo_)
    
        self.dLWhy_ = np.zeros_like(self.Why_)
        self.dLby_ = np.zeros_like(self.by_)        
    def CleanForwardInfos(self):
        
        # forward infos
        self.gt_ = [[]]
        self.dgt_ = [[]]
        self.it_ = [[]]
        self.dit_ = [[]]
        self.ft_ = [[]]
        self.dft_ = [[]]
        self.ot_ = [[]]
        self.dot_ = [[]]
        self.ht_ = [[]]
        self.dht_ = [[]]
        self.yt_ = [[]]
        self.dyt_ = [[]]
        # 
        self.st_ = [np.zeros([self.num_hidden_, 1])]
        self.ht_ = [np.zeros([self.num_hidden_, 1])]
        # backward infos
        self.dLh_ = []
        self.dLh_next_ = []
        self.dLs_ = []
        self.dLs_next_ = []
    
    def Calculator(self, Wx ,xt, Wh, ht, b):
        r = np.dot(Wx, xt) + np.dot(Wh, ht) + b
        return r
    def Train(self, X, Y):
        for i in xrange(len(X)):
            xi = X[i]
            yi = Y[i]
            self.Forward(xi)
            self.Backward(xi, yi)
            self.GradChecker(xi, yi)
    def Forward(self, xs):
        self.CleanForwardInfos()
        n = len(xs)
        for t in range(1, n + 1):
            xt = xs[t - 1]
            t1 = self.Calculator(self.Wgx_, xt, self.Wgh_, self.ht_[t - 1], self.bg_)
            self.gt_.append(Tanh(t1))
            self.dgt_.append(DerTanh(t1))
            
            t2 = self.Calculator(self.Wix_, xt, self.Wih_, self.ht_[t - 1], self.bi_)
            self.it_.append(Sigmoid(t2))
            self.dit_.append(DerSigmoid(t2))
            
            t3 = self.Calculator(self.Wfx_, xt, self.Wfh_, self.ht_[t - 1], self.bf_)
            self.ft_.append(Sigmoid(t3))
            self.dft_.append(DerSigmoid(t3))
            
            t4 = self.Calculator(self.Wox_, xt, self.Woh_, self.ht_[t - 1], self.bo_)
            self.ot_.append(Sigmoid(t4))
            self.dot_.append(DerSigmoid(t4))
            
            self.st_.append(self.gt_[t] * self.it_[t] + self.st_[t - 1] * self.ft_[t])
            
            self.ht_.append(Tanh(self.st_[t] * self.ot_[t]))
            self.dht_.append(DerTanh(self.st_[t] * self.ot_[t]))
            v = np.dot(self.Why_, self.ht_[t]) + self.by_
            self.yt_.append(Softmax(v))
            #self.dyt_.append(self.DerSoftmax(v))
    # compute time T dLh
    def DLhT(self, yt, t):
        # compute time T
        y_pred = copy.deepcopy(self.yt_[t])
        y_pred[yt] -= 1.0
        dLht = np.dot(self.Why_.T, y_pred)
        self.dLh_.append(dLht)
        # compute for time T - 1
        
        return dLht.reshape([dLht.shape[0], 1])
    # compute time T dLs
    def DLsT(self, yt):
        s_T = self.st_[-1] 
        o_T = self.ot_[-1]
        dLst = self.DLhT(yt) * self.DerSigmoid(s_T * o_T) * o_T
        return dLst
    def StepUpdate(self, t, xt, yt, up_s, up_h):
        dhs = self.dht_[t] * self.ot_[t]
        dho = self.dht_[t] * self.st_[t]
        dLs = up_h * dhs  + up_s
        
        dsWgx = self.it_[t] * self.dgt_[t]
        dsWgh = self.it_[t] * self.dgt_[t] #* self.ht_[t - 1]
        dsbg = self.it_[t] * self.dgt_[t]
        
        dsWix = self.gt_[t] * self.dit_[t]
        dsWih = self.gt_[t] * self.dit_[t] #* self.ht_[t - 1]
        dsbi = self.gt_[t] * self.dit_[t]
        
        dsWfx = self.st_[t - 1] * self.dft_[t]
        dsWfh = self.st_[t - 1] * self.dft_[t] #* self.ht_[t - 1]
        dsbf = self.st_[t - 1] * self.dft_[t]
        
        # computer der
        # g gate paras
        self.dLWgx_ += np.outer(dLs * dsWgx, xt)
        self.dLWgh_ += np.outer(dLs * dsWgh, self.ht_[t - 1])
        self.dLbg_ += dLs * dsbg
        # i gate paras
        self.dLWix_ += np.outer(dLs * dsWix, xt)
        self.dLWih_ += np.outer(dLs * dsWih, self.ht_[t - 1])
        self.dLbi_ += dLs * dsbi
        # f gate paras
        self.dLWfx_ += np.outer(dLs * dsWfx, xt)
        self.dLWfh_ += np.outer(dLs * dsWfh, self.ht_[t - 1])
        self.dLbf_ += dLs * dsbf
        # o gate paras
        dLo = up_h * dho
        self.dLWox_ += np.outer(dLo * self.dot_[t], xt)
        self.dLWoh_ += np.outer(dLo * self.dot_[t], self.ht_[t - 1])
        self.dLbo_ += dLo * self.dot_[t]
        # output paras
        y_pred = copy.deepcopy(self.yt_[t])
        y_pred[yt] -= 1.0
        self.dLWhy_ += np.outer(y_pred, self.ht_[t])
        self.dLby_ += y_pred
        
        # computer next state
        dgh1 = np.dot(self.Wgh_.T, self.dgt_[t])
        dih1 = np.dot(self.Wih_.T, self.dit_[t])
        dfh1 = np.dot(self.Wfh_.T, self.dft_[t])
        doh1 = np.dot(self.Woh_.T, self.dot_[t])
        
        r = self.st_[t] * doh1 + (self.gt_[t] * dih1 + self.it_[t] * dgh1 + self.st_[t - 1] * dfh1) * self.ot_[t]
        self.down_s = dLs * self.ft_[t]
        self.down_h = up_h * self.dht_[t] * r 
        
    def Backward(self, xs, ys):
        self.CleanDerInfos()
        t = type(ys)
        if(not(t is list)):
            ys = list(ys)
        
        up_h = self.DLhT(ys[-1], -1)
        up_s = np.zeros([up_h.shape[0], 1])
        
        n = len(ys)
        xt = xs[n - 1]
        yt = ys[n - 1]
        t = n
        self.StepUpdate(t, xt, yt, up_s, up_h)
        
        for t in range(n - 1, 0, -1):
            xt = xs[t - 1]
            yt = ys[t - 1]
            dlh = self.DLhT(yt, t)
            up_h = self.down_h + dlh
            up_s = self.down_s
            self.StepUpdate(t, xt, yt, up_s, up_h)
    def Update(self):
        pass
    def OneModelParaChecker(self, bptt_w, xs, ys):
        numeric_w = np.zeros_like(bptt_w)
        m, n = numeric_w.shape
        delta = 1e-6
        t = len(xs)
        for i in xrange(m):
            for j in xrange(n):
                self.Forward(xs)
                y0_list = []
                for ti in xrange(1, t + 1):
                    yt = ys[ti - 1]
                    y_0 = np.log(self.yt_[ti][yt])
                    y0_list.append(y_0)
                    
                self.CleanForwardInfos()
                bptt_w[i, j] += delta
                self.Forward(xs)
                y1_list = []
                for ti in xrange(1, t + 1):
                    yt = ys[ti - 1]
                    y_1 = np.log(self.yt_[ti][yt])
                    y1_list.append(y_1)
                numeric_w[i, j] = sum((np.array(y0_list) - np.array(y1_list)) / delta)
                bptt_w[i, j] -= delta
        return numeric_w
    def GradChecker(self, xs, ys):
        def Plot(numeric_w, bptt_w, name):
            fig = plt.figure()
            plt.plot(numeric_w.ravel(), 'r')
            plt.plot(bptt_w.ravel(), 'og')
            plt.title(name)
            plt.legend(['numeric grad', 'bptt grad'])
            plt.savefig('./fig/' + name + '.jpg')
        dLWgx = self.OneModelParaChecker(self.Wgx_, xs, ys)
        Plot(dLWgx, self.dLWgx_, 'dLWgx')
        dLWix = self.OneModelParaChecker(self.Wix_, xs, ys)
        Plot(dLWix, self.dLWix_, 'dLWix')
        dLWfx = self.OneModelParaChecker(self.Wfx_, xs, ys)
        Plot(dLWfx, self.dLWfx_, 'dLWfx')
        dLWox = self.OneModelParaChecker(self.Wox_, xs, ys)
        Plot(dLWox, self.dLWox_, 'dLWox')
        
        dLWgh = self.OneModelParaChecker(self.Wgh_, xs, ys)
        Plot(dLWgh, self.dLWgh_, 'dLWgh')
        dLWih = self.OneModelParaChecker(self.Wih_, xs, ys)
        Plot(dLWih, self.dLWih_, 'dLWih')
        dLWfh = self.OneModelParaChecker(self.Wfh_, xs, ys)
        Plot(dLWfh, self.dLWfh_, 'dLWfh')
        dLWoh = self.OneModelParaChecker(self.Woh_, xs, ys)
        Plot(dLWoh, self.dLWoh_, 'dLWoh')        
        
        dLbg = self.OneModelParaChecker(self.bg_, xs, ys)
        Plot(dLbg, self.dLbg_, 'dLbg')
        dLbi = self.OneModelParaChecker(self.bi_, xs, ys)
        Plot(dLbi, self.dLbi_, 'dLbi')
        dLbf = self.OneModelParaChecker(self.bf_, xs, ys)
        Plot(dLbf, self.dLbf_, 'dLbf')
        dLbo = self.OneModelParaChecker(self.bo_, xs, ys)
        Plot(dLbo, self.dLbo_, 'dLbo')                
        
        dLWhy = self.OneModelParaChecker(self.Why_, xs, ys)
        Plot(dLWhy, self.dLWhy_, 'dLWhy')
        dLby = self.OneModelParaChecker(self.by_, xs, ys)
        Plot(dLby, self.dLby_, 'dLby')
        
    def Sigmoid(self, x):
        return 1/(1 + np.exp(-x))   
    def DerSigmoid(self, x):
        return self.Sigmoid(x) * (1.0 - self.Sigmoid(x))
        
    def Tanh(self, x): 
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def DerTanh(self, x):
        return 1.0 - self.Tanh(x) * self.Tanh(x)
    
    def Softmax(self, x):
        r = np.exp(x)
        return  r / sum(r)
    def DerSoftmax(self, x):
        return self.Softmax(x) * (1.0 - self.Softmax(x))