import numpy as np
import matplotlib.pyplot as ply
import seaborn as sns 



X_train=np.array([[2104,5,1,45],  # Training Set Number -1
                 [1416,3,2,40], # Training set Number -2 
                 [852,2,1,35]])  # Training set Number -3
 
Y_train=np.array([460,232,178])
#  We have to treat the Whole Operation Like Two-Dimensional Array i and j 
print(X_train[1,2]) # Operations of 2-Dimensional Arrays 




# Defining the compute_function 
def compute_function(x,w,b):
    f_wb=np.dot(x,w) +b
    return f_wb



# Defining the compute_cost
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        temp_i=((np.dot(x[i],w)+b)-y[i])**2
        cost += temp_i
    cost = cost/(2*m)
    return cost


#  Calculating the Derivatives Vector 
def gradient_cal(x,y,w,b):
# df_dw --> Will be a vector 
# df_db --> will be a list

# X_train= [[2104,5,1,45](i=0),[1416,3,2,40](i=1), [852,2,1,35](i=2)]
# m=3
# n=4
# w = [w1 w2 w3 w4]
# y = [y1 y2 y3 ]
    df_db=0.0 
    n=x.shape[1]
    m=x.shape[0]
    df_dw=np.zeros((n,))   # Initializing the List with required size to avoid the error

    for i in range(m):
        common_part = (compute_function(x[i],w,b) -y[i])
        for j in range(n):
            df_dw[j]+=common_part*x[i,j]  # This denotes an Element not any other thing
        df_db+=common_part
        
    df_dw=df_dw/m
    df_db=df_db/m

    return df_dw,df_db


# Implementing the Gradient Descent Algorithm
def descent_algo(x,y,learning_rate,iterations):

    m=x.shape[0]
    n=x.shape[1]
    w_init = np.array([0,0,0,0])
    b_init=0.0

    for i in range(iterations):
        dj_dw,dj_db=gradient_cal(x,y,w_init,b_init)
        temp_w=w_init- (learning_rate * dj_dw)
        temp_b=b_init- (learning_rate * dj_db)
        w_init=temp_w # Here we are updating a list and not any single variable 
        b_init=temp_b
   
    return w_init,b_init



print(f"The Optimized Values for W(weight): {descent_algo(X_train,Y_train,0.001,10000)[0]} The Optimised Value for b(bias) : {descent_algo(X_train,Y_train,0.01,10000)[1]}")

print(gradient_cal(X_train,Y_train,np.array([1,2,3,4]),34)[0])
print(descent_algo(X_train,Y_train,0.001,1000)[0])