# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
def zh(a):
    if a>0:
        a=a
    else:
        a=0
    return a    

def standRegres(xArr, yArr):
    '''
    Description：
        linear regression
    Args:
        xArr ：feature
        yArr ：label 
    Returns:
        ws：coeff
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lasso(xArr, yArr,lam):
    ws=standRegres(xArr, yArr)
    m = shape(xArr)[1]
    lassows=zeros(m)
    for i in range(m-1):
        lassows[i]=double(ws[i])*max(0,1-m*lam/double(ws[i]))
    return (lassows.T)

def lassopredict(x_test,x_train,y_train,lam):
    ws=lasso(x_train, y_train,lam)
    result=dot(x_test,ws)
    return result