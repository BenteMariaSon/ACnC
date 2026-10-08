import numpy as np 
from sklearn.base import TransformerMixin, BaseEstimator

class OPLS(BaseEstimator, TransformerMixin):
    """Class to calculate Orthogonalized Partial Least Squares (OPLS). 
    OPLS seperates predictive variation from orthogonal variation in the predictor matirix x relative to the response variable y
    This implementation extimates one predictive component and removes the orthogonal components iteratively.
    
    Parameters:
    n_components : int, default=5   - number of orthogonal components to remove
    scale : bool, default=True      - If True, perform autoscaling on X
    scale_y : bool, default=True    - If True, perform autoscaling on y

    Attributes:
    w : ndarray         - Predictive weight vector
    t : ndarray         - Predictive score vector
    p : ndarray         - Predictive loading vector
    q : ndarray         - Response loading vector
    T_o : ndarray       - Matrix of orthogonal score vecotrs
    P_o : ndarray       - Matrix of orthogonal loading vecotrs
    W_o : ndarray       - matrix of orthogonal weight vectors
    x_mean_ : ndarray   - mean of X used for centering
    x_std_ : ndarray    - Standard deviation of X used for scaling
    y_mean_ : ndarray   - mean of y used for centering
    y_std_ : ndarray    - Standard deviation of y used for scaling

    Functions:
    .fit(X, y)  - Fit the X and y data to the model 
    .project(X) - Project new X data onto a model and return the scores of the predictive and orthogonal components
    .predict(X) - Predict the y values of the given X matrix based on the predictive component 
    .predict_classes(X) - predict the classes of a given X matrix based on the predictive component (only work for binary data)

    """
    def __init__(self, n_components=5, scale=True, scale_y=False):
        self.n_components = n_components
        self.scale = scale
        self.scale_y = scale_y
        self.w = None
        self.t = None 
        self.p = None
        self.q = None
        self.T_o = None
        self.P_o = None
        self.W_o = None

    def fit(self, X, y):
        """Fit the OPLS model to data X and response y
        
        Parameters
        X : araay-like of shape (n_samples, n_features)       - Training data
        y : array-like of shape (n_samples,) or (n_samples,1) - Target values 
        """
        try:
            X=np.asarray(X, copy=True)
        except:
            pass
        try:
            y=np.asarray(y, copy=True, dtype=np.float64)  
        except:
            pass
        if y.ndim == 1:
            try:
                y = y.reshape(-1,1)
            except:
                pass

        # Apply mean centering and scaling if scale=True
        self.x_mean_ = X.mean(axis=0)
        X -= self.x_mean_ 
        if self.scale:
            self.x_std_ = X.std(axis=0)
            X /= self.x_std_
        else:
            self.x_std_ = np.ones(X.shape[1])

        self.y_mean_ = y.mean(axis=0)
        y -= self.y_mean_
        if self.scale_y:
            self.y_std_ = y.std(axis=0)
            y /= self.y_std_
        else: 
            self.y_std_ = np.ones(y.shape[1])

        T_o = []
        P_o = []
        W_o = []

        E = X.copy()

        # Predictive weight vector
        w = (np.linalg.inv(y.T @ y) @ y.T @ E).T # shape: (n_features, 1)
        w = w / np.sqrt(w.T @ w)

        for _ in range(self.n_components):
            t = (E @ w) / (w.T @ w) # shape: (n_samples, 1)
            p = (t.T @ E) / (t.T @ t) # shape: (1, n_features)
            p = p.T # shape: (n_features, 1)

            w_proj = ((w.T @ p) / (w.T @ w)) * w # Project p onto w
            w_o = p - w_proj
            w_o = w_o / np.linalg.norm(w_o)
            
            t_o = (E @ w_o) / (w_o.T @ w_o)
            p_o = (t_o.T @ E) / (t_o.T @ t_o)
            p_o = p_o.T

            E = E - t_o @ p_o.T

            T_o.append(t_o)
            P_o.append(p_o)
            W_o.append(w_o)

        # Convert list of arrays to matrices
        T_o = np.hstack(T_o)
        P_o = np.hstack(P_o)
        W_o = np.hstack(W_o)

        # Final predictive components
        w = ((y.T @ E) / (y.T @ y)).T
        w = w / np.linalg.norm(w)
        t = (E @ w) / (w.T @ w)
        p = ((t.T @ E) / (t.T @ t)).T
        q = ((t.T @ y) / (t.T @ t)).T

        self.w = w
        self.t = t 
        self.p = p
        self.q = q
        self.T_o = T_o
        self.P_o = P_o
        self.W_o = W_o

    def project(self, X):
        '''
        Projects new X data into the exsiting OPLS space and returns the scores of the predictive
        component and the score of the orthogonal components respectively'''
        
        Xc = X.copy()

        Xc -= self.x_mean_
        Xc /= self.x_std_

        # project X on the orthogonal part
        rotations_o = np.dot(self.W_o, np.linalg.pinv(np.dot(self.P_o.T, self.W_o)))
        t_o_proj = np.dot(Xc, rotations_o)

        # remove the orthogonal part from X
        x_o_hat = np.dot(t_o_proj, self.P_o.T)
        Xc -= x_o_hat

        # project X on the predictive part 
        rotations = np.dot(self.w, np.linalg.pinv(np.dot(self.p.T, self.w)))
        t_proj = np.dot(Xc, rotations)

        return t_proj, t_o_proj

    def reconstruct_with_predicitve(self, scores):
        """Reconstruct X from only the predicitve scores (returned first by .project)"""
        return (scores @ self.p.T) * self.x_std_ + self.x_mean_

    def reconstruct_with_orthogonal(self, scores):
        """Reconstruct X from only the orthogonal scores (returned second by .project)"""
        return (scores @ self.P_o.T) * self.x_std_ + self.x_mean_

    def predict(self, X):
        """Predict the y-values of a new X matrix"""
        t_proj, t_o_proj = self.project(X)
        y_pred = np.dot(t_proj, self.q.T) * self.y_std_ + self.y_mean_
        return y_pred.T
    
    def predict_classes(self, X):
        """Predict the y-classes of a new X matrix"""
        y_pred = self.predict(X)
        return np.where(y_pred>0.5, 1, 0)[0]