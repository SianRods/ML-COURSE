import numpy as np

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    # Input : Z =[1,2,3,4,5]
    # return array of 'a' containing all the activation values 
    # Both the 'z' and the activation vector 'a' will have the same size
    ### START CODE HERE ### 
    deno=0
    a=np.zeros(len(z))
    for i in range(len(z)):
        deno+= np.exp(z[i])
    
    for j in range(len(z)):
        temp=np.exp(z[j])/deno
        a[j]=temp
    
    ### END CODE HERE ### 
    return a