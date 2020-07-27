'''
@Description: 感知机
@Author: 陈十一
@Date: 2020-07-27 09:28:13
@LastEditTime: 2020-07-27 10:12:40
@LastEditors: 陈十一
'''
import numpy as np

t = 0

# 感知机算法  学习率r默认为1
def Perceptron(X, Y, w=None, b=0, r=1):
  rows, columns = X.shape
  
  if w is not None:
    if len(w) != columns:
      print("w size error")
      return
  else:
    w = np.zeros(columns)
  
  wrongs = rows
  
  for i in range(rows):

    # 损失函数小于0,即被误分类,更新参数
    if (np.dot(w, X[i]) + b) * Y[i] <=0:
      global t 
      t = t + 1
      w = w + Y[i] * X[i] * r
      b = b + Y[i] * r
      print("iteration times:", t, "mis point X", i + 1, "w=", w, "b=", b)
    else:
      wrongs = wrongs - 1
  
  # 还有误分类点时继续训练
  if wrongs == 0:
    print("finished, w=", w, "b=", b)
  else:
    Perceptron(X, Y, w=w, b=b, r=r)

# 样例数据，来自《统计学习方法》例2.1
Xs = np.array([(3,3), (4,3), (1,1)])
Ys = [1, 1, -1]
Perceptron(Xs, Ys)


    
  

