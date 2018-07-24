# -*- coding: utf-8 -*-

def get_params_num(netparams):
    total_num = 0.
    for param_str in netparams:
        param_num = 1.
        param = netparams[param_str]
        param_shape = param.shape
        for i in param_shape:
            param_num *= i
        
        total_num += param_num
    return int(total_num)

def print_params_num(netparams):
    params_num = get_params_num(netparams)
    print '==='*30
    print 'The total number of parameters is %d'%params_num