#%%

import numpy as np

x1s=[-1,1,-1,-1,1,1,-1,1]
x2s=[-1,-1,1,-1,1,-1,1,1]
x3s=[-1,-1,-1,1,-1,1,1,1]
rets=[-1,-1,-1,1,1,1,1,-1]
def checkInput(x1, x2, x3, layer1, layer2):
    x4=sign(layer1[0][0]+layer1[0][1]*x1+layer1[0][2]*x2+layer1[0][3]*x3)
    x5=sign(layer1[1][0]+layer1[1][1]*x1+layer1[1][2]*x2+layer1[1][3]*x3)
    x6=sign(layer1[2][0]+layer1[2][1]*x1+layer1[2][2]*x2+layer1[2][3]*x3)
    x7=sign(layer1[3][0]+layer1[3][1]*x1+layer1[3][2]*x2+layer1[3][3]*x3)
    return sign(layer2[0]+layer2[1]*x4+layer2[2]*x5+layer2[3]*x6+layer2[4]*x7)

def sign(x):
    if(x>=0):
        return 1
    return -1

def checkNN(layer1, layer2):
    for i in range(8):
        if checkInput(x1s[i], x2s[i],x3s[i],layer1,layer2)!=rets[i]:
            print("failed at ", x1s[i],x2s[i],x3s[i])
            return False
    return True


#%%
# layer1=[[-2.5,1,1,-1],[-1.5,-1,0,1],[-1.5,0,-1,1]]
layer1=[[-1,1,1,0],[1,0,0,1],[1,-1,-1,0],[1,0,0,-1]]
# layer1=[[-2,1,1,-1],[-1,-1,-1,2]]
layer2=[-1,1,1,1,1]
print(checkNN(layer1,layer2))
# %%
