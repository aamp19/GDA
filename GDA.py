import numpy as np
import util1
import math
import pandas as pd
import matplotlib.pyplot as plt


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util1.load_dataset(train_path, add_intercept=False)

#     # *** START CODE HERE ***

    GDA_model = GDA()
    
    GDA_model.fit(x_train, y_train)
    valid_x, valid_y = util1.load_dataset(valid_path, add_intercept=False)

    
    y = GDA_model.predict(valid_x)
#     print('y ',y.shape)
    np.savetxt(save_path,y)
    #plot(valid_x, y, GDA_model.theta,save_path)
    util1.plot(valid_x, valid_y, GDA_model.theta,save_path)
    
    #np.savetxt('validation.txt',y)
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n, dtype=np.float32)
        sum_0 = 1
        sum_1 = 1
        sigma = 0
        ones = np.sum(y)
        phi = ones/m
        zeros = m - ones
        for i in range(m):
            x[i] = x[i].reshape(1,len(x[i]))
            #print(x[i].shape)
            if y[i] == 0:
                
                sum_0 += x[i]
                
                
            elif y[i] == 1:
                
                sum_1 += x[i]
                
        
        mu_0 = sum_0 / zeros
        mu_1 = sum_1 / ones
        
        for j in range(m):
            x[j] = x[j].reshape(1,len(x[j]))
            #print(x[j].shape)
            if y[j] == 0:
                
                sigma += (np.dot(((x[j] - mu_0).reshape(len((x[j] - mu_1)),1)), (((x[j] - mu_0).reshape(len(x[j] - mu_0),1)).transpose())))/m
                #print((x[j] - mu_0).reshape(len((x[j] - mu_1)),1).shape)
            elif y[j] == 1:
               
                sigma += (np.dot((x[j] - mu_1), ((x[j] - mu_1).transpose())))/m
        
        
#         mu_0 = mu_0.reshape(len(mu_0),1)
#         mu_1 = mu_1.reshape(len(mu_1),1)
        #sigma = sigma.reshape(len(sigma),1)
#         print('sigma ',sigma)
#         print('sigma shape', sigma.shape)
#         print('mu_0 ',mu_0)
#         print('mu_1 ', mu_1)
#         print('x ', x.shape)
        
        
        #theta_0 = (((mu_0.transpose()*np.linalg.inv(sigma)*mu_0) - (mu_1.transpose()*np.linalg.inv(sigma)*mu_1))/2 - np.log((1-phi)/phi))/2
        #theta_0 = (((np.transpose(mu_0)*np.linalg.inv(sigma)*mu_0) - (np.transpose(mu_1)*np.linalg.inv(sigma)*mu_1))/2 - np.log((1-phi)/phi))/2
#         theta_0 = ((np.dot(np.dot(mu_0.transpose(),np.linalg.inv(sigma)),mu_0)) - (np.dot(np.dot(mu_1.transpose(),np.linalg.inv(sigma)),mu_1)))/2 - np.log((1-phi)/phi)
#         theta = -1*np.dot(np.linalg.inv(sigma),(mu_0 - mu_1))
#         self.theta[0] = theta_0
#  
#         self.theta = np.zeros(n + 1)
#         self.theta[0] = (np.dot(np.dot(mu_0.transpose(), np.linalg.inv(sigma)), mu_0) - np.dot(np.dot(mu_1.transpose(), np.linalg.inv(sigma)), mu_1))/2 - np.log(1-phi/(phi+0.00000005))
#         self.theta[1:] = -1*np.dot(np.linalg.inv(sigma),(mu_0 - mu_1))
       
        self.theta = np.zeros(n + 1)
        self.theta[0] = (np.dot(np.dot(mu_0.transpose(), np.linalg.inv(sigma)), mu_0) - np.dot(np.dot(mu_1.transpose(), np.linalg.inv(sigma)), mu_1))/2 - np.log((1-phi)/phi)
        self.theta[1:] = -1*np.dot(np.linalg.inv(sigma),(mu_0 - mu_1))
        
#         print('theta ', self.theta)
        
# #         self.theta_combine = np.array([theta_0, theta])
#         print('x ',x.shape)
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).
        
        Returns:
            Outputs of shape (n_examples,).
        """
        
        # *** START CODE HERE ***
        #x = x.reshape(len(x),2)
        #print('theta combine1 shape ',self.theta_combine.shape)
        #print('x ',x.shape)
        theta0 = self.theta[0]
        theta1 = self.theta[1]

        #y_hat = 1/ (1+ np.exp(-1*np.dot(theta1.T, x))+theta0)
#         print('theta combine0',self.theta_combine[0].shape)
#         print(self.theta_combine[0])
        #print("theta shape",self.theta.shape)
#         print('theta0 ',theta0)
#         print('theta1 ',theta1)
        #theta1 = theta1.reshape(len(theta1),1)
#         print("theta shape",self.theta.shape)
        #y_hat = (1/(1 + np.exp(-1*(np.dot((theta1).transpose(),x) + theta0))))
        y_hat = 1/ (1+ np.exp(-np.dot(theta1.transpose(), x)) - theta0)
#         y_hat = y_hat.reshape(y_hat.shape[1],)
            
#         print('y_hat ',y_hat.shape)

        return y_hat
    
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.png')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.png')
