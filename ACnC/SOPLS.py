import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from itertools import product
from RSSchemometrics.Plotting import rucolors

class MeanCentering(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MeanCentering'
      self.mean=None
      self.fitted_ = False

    def fit(self,X,y=None):
      self.fitted_ = True
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
      return self
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.asarray(X)-np.asarray(self.mean))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 


def cross_validate(model, data_blocks, y, splitter, scoring, groups=None):
    scores = []
    for train_idx, test_idx in splitter.split(data_blocks[0], y, groups=groups):
        X_train = [block[train_idx] for block in data_blocks]
        X_test = [block[test_idx] for block in data_blocks]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores.append(scoring(y_test, y_pred))
    return np.asarray(scores)


def neg_RMSE(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return -(np.sqrt(np.average((y_true - y_pred)**2)))


class SOPLS(TransformerMixin, ClassifierMixin, RegressorMixin, BaseEstimator):
    """Class to calculate Sequential and Orthogonalized Partial Least Squares (SO-PLS).
    SO-PLS is a multi-block regression model method based on PLS, designed to handle multiple
    blocks of predictor data in a structures and interpretable way. 

    Parameters:
    n_components : int or array-like, default=5   - number of components to be extracted in each block, 
                                                    can be either a scalar (same number of each block) 
                                                    or an array of shape (n_blocks,1)
    x_pp : string or string-array of shape (n_blocks, 1), default='mean'  - Desired pretreatments for each X-block
                                                    Can take 'none', 'mean', or 'scale'. If ppro is a single string the same
                                                    pre-processing will be applied to all X blocks.
    y_pp : string, default='none'                 - Desired pretreatment for the y-array, can take 'none', 'mean', or 'scale'
                                        
    """
    def __init__(self, n_components=5, x_pp='mean', y_pp='none', pp_per_block=None):
        self.n_components = n_components
        self.x_pp = x_pp
        self.y_pp = y_pp
        self.pp_per_block = pp_per_block

        self.pls_models = []
        self.x_means_ = []
        self.x_stds_ = []
        self.y_mean_ = None 
        self.y_std_ = None

    def orthogonalize(self, X, T_prev):
        """Remove the part of X that is explained by the previous scores (T_prev)"""
        if T_prev.shape[1] == 0:
            return X
        P = T_prev @ np.linalg.pinv(T_prev.T @ T_prev) @ T_prev.T
        # P = T_prev @ np.linalg.pinv(T_prev) # Old orthogonalization as done in OPLS function
        return X - P @ X

    def fit(self, X_blocks, y):
        """Fit the SO-PLS model to the data blocks in X and response y

        Parameters:
        X_blocks : List containing X-blocks (of length n_blocks) where each block must have the same anount or rows (n_samples)
        y : Target block
        """

        y = np.asarray(y, copy=True)
        
        if y.ndim == 1:
            y = y.reshape(-1,1)

        self.n_blocks_ = len(X_blocks)
        self.pls_models_ = []
        self.x_means_ = []
        self.x_stds_ = []

        if isinstance(self.n_components, int): # if n_components is scalar we make sure it is repeated for each block
            self.n_components_ = [self.n_components] * self.n_blocks_
        else:
            self.n_components_ = self.n_components

        if isinstance(self.x_pp, str): # if x_pp is scalar we make sure it is repeated for each block
            self.x_pp_ = [self.x_pp] * self.n_blocks_
        else:
            self.x_pp_ = self.x_pp
            
        if isinstance(self.pp_per_block, Pipeline): # if pp_per_block contains a single Pipeline, repeat it for every block
            self.pp_per_block_ = [self.pp_per_block] * self.n_blocks_
        elif self.pp_per_block is None: # repeat the None for ever block as well
            self.pp_per_block_ = [self.pp_per_block] * self.n_blocks_
        else:
            self.pp_per_block_ = self.pp_per_block

        # y preprocessing
        if self.y_pp == 'none':
            self.y_std_ = np.ones(y.shape[1])
            self.y_mean_ = np.zeros(y.shape[1])
        else:
            self.y_mean_ = y.mean(axis=0)
            y -= self.y_mean_
            if self.y_pp == 'scale':
                self.y_std_ = y.std(axis=0)
                y /= self.y_std_
            else:
                self.y_std_ = np.ones(y.shape[1])

        y_res = y.copy().astype(float)
        T_concat = np.zeros((y.shape[0], 0))

        # Create extractable instances for X_ortho and y_res
        self.X_orthos_ = np.zeros(len(X_blocks), dtype=object)
        self.y_ress_ = np.empty(len(X_blocks), dtype=object)
        for i, X in enumerate(X_blocks):
            self.y_ress_[i] = y_res.copy()

            X = np.asarray(X).copy()
            # preprocess X
            if self.x_pp_[i] != 'none':
                mean = X.mean(axis=0)
                X -= mean
                self.x_means_.append(mean)
            else:
                self.x_means_.append(np.zeros(X.shape[1]))

            if self.x_pp_[i] == 'scale':
                std = X.std(axis=0)
                std[std == 0] = 1
                X /= std
                self.x_stds_.append(std)
            else:
                self.x_stds_.append(np.ones(X.shape[1]))

            # orthogonalize
            X_ortho = self.orthogonalize(X, T_concat)
            self.X_orthos_[i] = X_ortho
            if self.pp_per_block_[i] is not None:
                X_ortho_pp = np.asarray(self.pp_per_block_[i].fit_transform(X_ortho))
            else:
                X_ortho_pp = X_ortho

            # PLS fit
            pls = PLSRegression(n_components=self.n_components_[i], scale=False)
            pls.fit(X_ortho_pp, y_res)
            self.pls_models_.append(pls)

            # update residual y
            y_res -= pls.predict(X_ortho_pp)

            # update score matrix
            T_concat = np.hstack((T_concat, pls.x_scores_))
        return self

    def predict(self, X_blocks):
        """Predict using the SO-PLS model
        
        Parameters
        X_blocks : List containing X-blocks (of length n_blocks) where each block must have the same anount or rows (n_samples)
        Returns  : predicted Y (n_samples, 1)"""

        T_concat = np.zeros((X_blocks[0].shape[0], 0))
        y_pred = 0
        for i, X in enumerate(X_blocks):
            X = np.asarray(X).copy()
            X -= self.x_means_[i]
            X /= self.x_stds_[i]
            X_ortho = self.orthogonalize(X, T_concat)
            if self.pp_per_block_[i] is not None:
                X_ortho_pp = np.asarray(self.pp_per_block_[i].transform(X_ortho))
            else:
                X_ortho_pp = X_ortho
            pls = self.pls_models_[i]
            y_block = pls.predict(X_ortho_pp)
            y_pred += y_block

            T_block = pls.transform(X_ortho_pp)
            T_concat = np.hstack((T_concat, T_block))

        y_pred *= self.y_std_
        y_pred += self.y_mean_
        return y_pred.flatten()

#========================================================================================================================================================================

class SOPLS_SeqLVopt(TransformerMixin, ClassifierMixin, BaseEstimator):
    """Class to optimize a Sequential and Orthogonalized Partial Least Squares (SO-PLS) model.
    SO-PLS is a multi-block regression model method based on PLS, designed to handle multiple
    blocks of predictor data in a structures and interpretable way.

    Parameters:
    x_pp : string or string-array of shape (n_blocks, 1), default='mean'  - Desired pretreatments for each X-block
                                                    Can take 'none', 'mean', or 'scale'. If ppro is a single string the same
                                                    pre-processing will be applied to all X blocks.
    y_pp : string, default='none'                 - Desired pretreatment for the y-array, can take 'none', 'mean', or 'scale'
    score_metric: string                          - The sklearn score metric to use in optimisation, the default score metric negative RMSE
    CV_Scheme: SKlearn model selection method     - Method to use for data splitting on the optimisation
    epsilon: float, default=1e-4                  - Early stopping cirterium
    max_LVs: array-like or scalar, default=30     - list containing the maximum number of LVs allowed for each datablock, if scalar, the same
                                                    maximum number of LVs is used for each block.
                                       
    Attributes:
    opt_LVs: List containing the optimum number of LVs to use for each block.
    pls_models: List containing the sklearn pls models used for each block.

    """
    def __init__(self, x_pp='mean', y_pp='none', pp_per_block=None, score_metric=neg_RMSE, CV_scheme=KFold(n_splits=5, shuffle=True, random_state=37), epsilon=1e-4, max_LVs=20, groups=None): #, forced_lvs=None):
        self.x_pp = x_pp
        self.y_pp = y_pp
        self.pp_per_block = pp_per_block
        self.CV_Scheme = CV_scheme
        self.epsilon = epsilon
        self.max_LVs = max_LVs
        self.score_metric = score_metric
        self.groups = groups

        self.opt_LVs = []
        self.pls_models = []

    def fit(self, X_blocks, y):
        y = np.asarray(y, copy=True)
    
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_blocks = len(X_blocks)
        self.pls_models = []
        self.x_means = []
        self.x_stds = []

        # We can increase the calculation speed by preprocessing the first block in advance (instead within every cross-validaiton loop)
        if isinstance(self.pp_per_block, Pipeline): # if pp_per_block contains a single Pipeline
            self.pp_per_block_ = [self.pp_per_block] * self.n_blocks
        else:
            self.pp_per_block_ = self.pp_per_block
        pp_per_block_rm0 = self.pp_per_block_.copy() if self.pp_per_block_ is not None else self.pp_per_block_
        x_blocks_pp0 = X_blocks.copy()
        if self.pp_per_block is not None:
            x_blocks_pp0[0] = np.asarray(self.pp_per_block_[0].fit_transform(x_blocks_pp0[0]))
            pp_per_block_rm0[0] = Pipeline([('mc', MeanCentering())]) # replace the pp method with MeanCentering (which effectivly does nothing because the data is already automatically meancentered)
        
        if isinstance(self.max_LVs, int): # if n_components is scalar we make sure it is repeated for each block
            self.max_LVs = np.repeat(self.max_LVs, self.n_blocks)
        elif self.max_LVs is None:
            self.max_LVs = []
            for i in range(self.n_blocks):
                self.max_LVs.append(min(X_blocks[i].shape[0], X_blocks[i].shape[1]))        

        # change max_LV if it is set to a number that is too high:
        for i in range(self.n_blocks):
            self.max_LVs[i] = min(X_blocks[i].shape[0], X_blocks[i].shape[1], self.max_LVs[i])

        current_blocks = []
        self.opt_LVs = []

        for block_num in range(self.n_blocks): # loop over the blocks
            last_score = -np.inf
            current_blocks.append(x_blocks_pp0[block_num]) # Start out with only the first block and add the new block after the first has been optimized
            
            cv_scores = np.full(self.max_LVs[block_num]+1, -np.inf)
            
            for i in range(0, self.max_LVs[block_num]): # loop over the number of LVsssssssss
                current_LVs = self.opt_LVs.copy()
                current_LVs.append(i+1)
                model = SOPLS(n_components=current_LVs, x_pp=self.x_pp, y_pp=self.y_pp, pp_per_block=pp_per_block_rm0)
                # model.fit(current_blocks, y_model)
                cv_score = cross_validate(model, current_blocks, y, splitter=self.CV_Scheme, groups=self.groups, scoring=self.score_metric).mean()
                # print(f"LV {i+1}, score={cv_score}")
                cv_scores[i] = cv_score

                # if cv_score - last_score < self.epsilon:
                #     break
                last_score = cv_score
            self.opt_LVs.append(np.argmax(cv_scores[:i+1])+1)

        self.pls_models = SOPLS(n_components=self.opt_LVs, x_pp=self.x_pp, y_pp=self.y_pp, pp_per_block=self.pp_per_block_)
        self.pls_models.fit(X_blocks, y)
        self.opt_LVs = np.asarray(self.opt_LVs, dtype=int)

    def predict(self, X_blocks):
        X_blocks = X_blocks.copy()
        return self.pls_models.predict(X_blocks)

#========================================================================================================================================================================

class SOPLS_GlobLVopt(TransformerMixin, ClassifierMixin, BaseEstimator):
    """Class to optimize a Sequential and Orthogonalized Partial Least Squares (SO-PLS) model.
    SO-PLS is a multi-block regression model method based on PLS, designed to handle multiple
    blocks of predictor data in a structures and interpretable way.

    Parameters:
    x_pp : string or string-array of shape (n_blocks, 1), default='mean'  - Desired pretreatments for each X-block
                                                    Can take 'none', 'mean', or 'scale'. If ppro is a single string the same
                                                    pre-processing will be applied to all X blocks.
    y_pp : string, default='none'                 - Desired pretreatment for the y-array, can take 'none', 'mean', or 'scale'
    score_metric: string                          - The sklearn score metric to use in optimisation, the default score metric negative RMSE
    CV_Scheme: SKlearn model selection method     - Method to use for data splitting on the optimisation
    epsilon: float, default=1e-4                  - Early stopping cirterium
    max_LVs: array-like or scalar, default=30     - list containing the maximum number of LVs allowed for each datablock, if scalar, the same
                                                    maximum number of LVs is used for each block.
                                       
    Attributes:
    opt_LVs: List containing the optimum number of LVs to use for each block.
    pls_models: List containing the sklearn pls models used for each block.

    """
    def __init__(self, x_pp='mean', y_pp='none', pp_per_block=None, score_metric=neg_RMSE, CV_Scheme=KFold(n_splits=5, shuffle=True, random_state=37), max_LVs=20, groups=None): #, forced_lvs=None):
        self.x_pp = x_pp
        self.y_pp = y_pp
        self.pp_per_block = pp_per_block
        self.CV_Scheme = CV_Scheme
        self.max_LVs = max_LVs
        self.score_metric = score_metric
        self.groups = groups
        self.is_fitted_ = False

        self.opt_LVs = []
        self.pls_models = []

    def fit(self, X_blocks, y):
        y = np.asarray(y, copy=True)
        self.is_fitted_ = True
    
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_blocks = len(X_blocks)
        self.pls_models = []
        self.x_means = []
        self.x_stds = []
        
        # We can increase the calculation speed by preprocessing the first block in advance (instead within every cross-validaiton loop)
        if isinstance(self.pp_per_block, Pipeline): # if pp_per_block contains a single Pipeline
            self.pp_per_block_ = [self.pp_per_block] * self.n_blocks
        else:
            self.pp_per_block_ = self.pp_per_block
        pp_per_block_rm0 = self.pp_per_block_.copy() if self.pp_per_block_ is not None else self.pp_per_block_
        x_blocks_pp0 = X_blocks.copy()
        if self.pp_per_block is not None:
            x_blocks_pp0[0] = np.asarray(self.pp_per_block_[0].fit_transform(x_blocks_pp0[0]))
            pp_per_block_rm0[0] = Pipeline([('mc', MeanCentering())]) # replace the pp method with MeanCentering (which effectivly does nothing because the data is already automaticall meancentered)
        

        if isinstance(self.max_LVs, int): # if n_components is scalar we make sure it is repeated for each block
            self.max_LVs = np.repeat(self.max_LVs, self.n_blocks)
        elif self.max_LVs is None:
            self.max_LVs = []
            for i in range(self.n_blocks):
                self.max_LVs.append(min(X_blocks[i].shape[0], X_blocks[i].shape[1]))        

        # change max_LV if it is set to a number that is too high:
        for i in range(self.n_blocks):
            self.max_LVs[i] = min(X_blocks[i].shape[0], X_blocks[i].shape[1], self.max_LVs[i])

        # Determine all possible combinations of LVs
        ranges = [range(1, max_lv+1) for max_lv in self.max_LVs]
        self.LV_combinations = [lv_combination for lv_combination in product(*ranges)]
        
        # Loop over all LV combinations and save the calculated RMSE values
        self.CV_scores = np.zeros(len(self.LV_combinations), dtype=float)
        for i, lv_combination in enumerate(self.LV_combinations):
            model = SOPLS(n_components=lv_combination, x_pp=self.x_pp, y_pp=self.y_pp, pp_per_block=pp_per_block_rm0)
            cv_score = cross_validate(model, x_blocks_pp0, y, splitter=self.CV_Scheme, groups=self.groups, scoring=self.score_metric).mean()
            self.CV_scores[i] = cv_score

        self.opt_LVs = self.LV_combinations[np.argmax(self.CV_scores)]

        self.pls_models = SOPLS(n_components=self.opt_LVs, x_pp=self.x_pp, y_pp=self.y_pp, pp_per_block=self.pp_per_block_)
        self.pls_models.fit(X_blocks, y)
        self.opt_LVs = np.asarray(self.opt_LVs, dtype=int)

    def make_Mage_plot(self, neg_scores=True, only_label_pareto=True, point_color='lightgray', pareto_color=rucolors.red, title=None, x_label=None, y_label=None, ax_in=None):
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first")
        
        model_complexities = np.asarray([np.sum(lv_combination) for lv_combination in self.LV_combinations], dtype=int)
        LV_combinations_str = np.asarray([(str(lv_combination)) for lv_combination in self.LV_combinations], dtype=str)
        if neg_scores:
            CV_scores_plt = np.abs(self.CV_scores)
        else:
            CV_scores_plt = self.CV_scores
            
        if ax_in is None:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            ax = ax_in
            fig = ax_in.figure
        
        ax.scatter(model_complexities, CV_scores_plt, color=point_color, marker='.')
        
        # Pareto front 
        points = list(zip(model_complexities, CV_scores_plt))
        pareto = []
        pareto_labels = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i != j:
                    if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                        dominated = True
                        break
            if not dominated:
                pareto.append(p)
                pareto_labels.append(LV_combinations_str[i])
        pareto = np.asarray(pareto)
        pareto = pareto[np.argsort(pareto[:,0])]
        plt.plot(pareto[:, 0], pareto[:, 1], color=pareto_color, linewidth=2, label='Pareto Front')
        
        # text labels next to points
        if only_label_pareto:
            for i in range(len(pareto_labels)):
                ax.text(pareto[i, 0], pareto[i, 1], pareto_labels[i])
        else:
            for i in range(len(model_complexities)):
                ax.text(model_complexities[i], CV_scores_plt[i], LV_combinations_str[i])
                
        ax.set_title(title if title else "måge plot")
        ax.set_xlabel(x_label if x_label else "Total number of components")
        ax.set_ylabel(y_label if y_label else "Prediction error")
        ax.legend()
        
        if ax_in is None:
            # fig.tight_layout()
            plt.show()
        else:
            return fig, ax
        
    def predict(self, X_blocks):
        X_blocks = X_blocks.copy()
        return self.pls_models.predict(X_blocks)


#=================================================================================

# # Simulated data blocks
# np.random.seed(1)

# # regression testing
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# X, y = make_regression(
#     n_samples=300,
#     n_features=50,
#     n_informative=10,
#     noise=0.1,
#     random_state=42
# )
# X_blocks = [
#     X[:, :20],
#     X[:, 20:35],
#     X[:, 35:]
# ]

# indices = np.arange(len(y))
# train_idx, test_idx = train_test_split(
#     indices,
#     test_size=0.3,
#     random_state=42
# )
# X_train_blocks = [block[train_idx] for block in X_blocks]
# X_test_blocks  = [block[test_idx] for block in X_blocks]
# y_train = y[train_idx]
# y_test  = y[test_idx]

#=================================================================================
# # test SOPLS
# sopls = SOPLS(n_components=[1,1,2], x_pp='mean')
# sopls.fit(X_train_blocks, y_train)
# y_pred = sopls.predict(X_test_blocks)
# print(f"RMSE: {np.sqrt(np.mean((y_pred.flatten()-y_test)**2))}")

#=================================================================================
# test Sequential LV optimisation
# sopls_cv = SOPLS_SeqLVopt(x_pp='mean')
# sopls_cv.fit(X_train_blocks, y_train)
# y_pred = sopls_cv.predict(X_test_blocks)
# print(f"Optimal number of LVs: {sopls_cv.opt_LVs}")
# print(f"RMSE: {np.sqrt(np.mean((y_pred.flatten()-y_test)**2))}")

#=================================================================================
# # test Global LV optimisation 
# sopls_cv_glob = SOPLS_GlobLVopt(x_pp='mean', max_LVs=5)
# sopls_cv_glob.fit(X_train_blocks, y_train)
# print(f"Optimal number of LVs: {sopls_cv_glob.opt_LVs}")
# sopls_cv_glob.make_Mage_plot()

#=================================================================================

# # Check if a single block SOPLS is equal to what we do with normal PLS
# sopls = SOPLS(n_components=[3], x_pp='none', y_pp='none')
# # print([X[:, :20].shape[1])
# sopls.fit([X[:, :20]], y)

# # print(sopls.pls_models_[0].x_scores_)

# pls = PLSRegression(n_components=3, scale=False) 
# pls.fit(X[:, :20], y)
# # print(pls.x_scores)
# print(pls.x_scores_[:5])
# print(sopls.pls_models_[0].x_scores_[:5])
# # print(((pls.x_scores_ == sopls.pls_models_[0].x_scores_)))
