import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MCR_ALS:
    """Class to perform MCR-ALS
    """
    def __init__(self, X=None, n_comps=None, init_C=None, init_S=None, init_method=None, n_experiments=None, verbose=False):
        """Initialize class

        Args:
            X (numpy, optional): Spectral data. Defaults to None.
            n_comps (int, optional): number of spectral components to use. Defaults to None.
            init_C (numpy, optional): initial concentration components. Defaults to None.
            init_S (numpy, optional): initial spectral components. Defaults to None.
            init_method (string, optional): 'random' / 'pureS' / 'pureC'. Defaults to None.
            n_experiments (int, optional): number of experiments (in case of trilinear data). Defaults to None.
            verbose (bool, optional): additional info. Defaults to False.
        """
        if verbose:
            print(f'Initiated MCR-ALS\nToolbox by Mickey Lukkien, Radboud University Nijmegen\nAdapted from Mahdiyeh Ghaffari, Radboud University Nijmegen')
            print("\n"
                    "MCR-ALS Steps:\n"
                    "1. Determine number of components:\n"
                    "- SVD Analysis\n"
                    "2. Estimate initial components:\n"
                    "- Random Guess ('random')\n"
                    "- Pure (Simplified version of SIMPLSMA) ('pureS')\n"
                    "- Pure rows (Simplified version of SIMPLSMA) ('pureC')\n"
                    "3. Optimize with constraints:\n"
                    "- Closure/Normalization (Concentration and spectral components) ('Closure')\n"
                    "- Non-negativity (Concentration and spectral components) ('Nneg')\n"
                    "- Trilinearity (Concentration Components) ('Trilinear')\n"
                    "4. Visualization:\n"
                    "- Concentration components\n"
                    "- Spectral components\n"
                    "- Lack of fit history\n"
                    "- Residuals history\n")
        self.X = X
        self.n_comps = n_comps
        self.init_C = init_C
        self.init_S = init_S
        self.init_method = init_method
        self.n_experiments = n_experiments
        self.mcr_results = None

    def estimate_ncomps(self, X=None, method='svd', max_ncomps=20, ax=None, verbose=False):
        """Generates variance plot to estimate number of components

        Args:
            X (numpy, optional): spectral data. Defaults to None.
            method (str, optional): 'svd' - no other options implemented. Defaults to 'svd'.
            max_ncomps (int, optional): maximum amount of components. Defaults to 20.
            ax (matplotlib ax, optional): ax to plot on. Defaults to None.
            verbose (bool, optional): additional info. Defaults to False.

        Raises:
            KeyError: Must specify X during initiation or calling this function

        Returns:
            matplotlib ax: variance plot
        """
        if method == 'svd':
            if X is None:
                if self.X is not None:
                    X = self.X
                    if verbose: print("X was not specified. Stored X is used.")
                else: raise KeyError("X has not been specified")
            else: 
                if verbose: print("Specified X has been stored in self") 
                self.X = X
            if max_ncomps > X.shape[0]:
                max_ncomps = X.shape[0]
                if verbose: print(f'Reduced max_ncomps to {max_ncomps} due to size of X')
            #Get variances with svd
            Xm = X - np.mean(X, axis=0)
            U, D, Vt = np.linalg.svd(Xm, full_matrices=False)
            explained_variance_fraction = D**2 / np.sum(D**2)
            cumulative_variance_fraction = np.cumsum(explained_variance_fraction)

            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(np.arange(1,max_ncomps+1), explained_variance_fraction[:max_ncomps], color='red', label='Fraction Explained Variance')
            ax.plot(np.arange(1,max_ncomps+1), cumulative_variance_fraction[:max_ncomps], color='black', label='Fraction Cumulative Explained Variance')
            ax.set_title("Explained variance vs. number of components")
            ax.set_xlabel("Component Number")
            ax.set_ylabel("Fraction explained variance")
            ax.set_xticks(np.arange(1,max_ncomps+1))
            ax.legend()
            ax.grid(True)
            return ax
    
    def estimate_initialcomps(self, X=None, n_comps=None, method='random', seed=None, verbose=False, visualize=False):
        """Estimate the initial components

        Args:
            X (numpy array, optional): Spectral data. Defaults to None.
            n_comps (int, optional): number of components to use. Defaults to None.
            method (str, optional): 'random' / 'pureC' / 'pureS': random initiation or pure spectral/concentration profiles. Defaults to 'random'.
            seed (int, optional): for consistency use the same number. Defaults to None.
            verbose (bool, optional): additional information. Defaults to False.
            visualize (bool, optional): visualize initial components. Defaults to False.

        Raises:
            KeyError: Must specify X during initiation or calling this function
            KeyError: Must specify n_comps during initiation or calling this function
        """
        
        self.init_method = method
        #Get required paramers from function call or from self
        #X
        if X is None:
            if self.X is not None:
                X = self.X
                if verbose: print("X was not specified. Stored X is used.")
            else: raise KeyError("X has not been specified")
        else:
            if verbose: print("Specified X has been stored in self") 
            self.X = X
        #n_comps
        if n_comps is None:
            if self.n_comps is not None:
                n_comps = self.n_comps
                if verbose: print(f'n_comps was not specified. Stored n_comps ({n_comps}) is used.')
            else: raise KeyError("n_comps has not been specified")
        else: 
            if verbose: print('Specified n_comps has been stored in self')
            self.n_comps = n_comps
        if n_comps > X.shape[0]:
            n_comps = X.shape[0]
            self.n_comps = n_comps
            if verbose: print(f'Reduced n_comps to {n_comps} due to size of X')
        #Shape of X
        n_rows, n_columns = X.shape

        if method == 'random':
            #Calculates random concentration profiles
            if verbose: print('Calculating random concentration profiles')
            if seed is not None:
                np.random.seed(seed)
                if verbose: print(f'Using a specified seed ({seed}). Results will be consistent.')
            else: 
                if verbose: print(f'Not using a specified seed. Results will be inconsistent.')
            self.init_C = np.abs(np.random.rand(n_rows, n_comps))
            if verbose: print(f'Initial random concentration profiles with {n_comps} has been stored in self.')
            if visualize:
                fig, axs = plt.subplots(1,1)
                axs.plot(self.init_C)
                axs.set_title("Initial concentration component")

        elif method == 'pureC':
        #Finds column with purest concentration profile
            if verbose: print(f'Calculating {n_comps} pure concentration profiles')
            vars = np.var(X, axis=0)
            idxs = np.argsort(vars)[-n_comps:]
            self.init_C = X[:,idxs]
            if verbose: print(f'Initial pure concentration profiles with {n_comps} has been stored in self.')
            if visualize:
                fig, axs = plt.subplots(1,1)
                axs.plot(self.init_C)
                axs.set_title("Initial concentration component")

        elif method == 'pureS':
            #Finds row with purest spectral profile
            if verbose: print(f'Calculating {n_comps} pure spectral profiles')
            vars = np.var(X, axis=1)
            idxs = np.argsort(vars)[-n_comps:]
            self.init_S = X[idxs, :]
            if verbose: print(f'Initial pure spectral profiles with {n_comps} has been stored in self.')
            if visualize:
                fig, axs = plt.subplots(1,1)
                axs.plot(self.init_S.T)
                axs.set_title("Initial spectral component")
        
        


        
    def fit(self, X=None, n_comps=None, n_iter=50, C_constraints = [], S_constraints = [], n_experiments=None, concentration=None, spectrum=None, X_calibration=None, Y_calibration=None, Correlation_Column = [0], early_stopping=True, early_stopping_method='residual', tol=1e-3, verbose=False):
        """Optimize components

        Args:
            X (numpy array, optional): spectra. Defaults to None.
            n_comps (int, optional): number of components to use. Defaults to None.
            n_iter (int, optional): max number of iterations for optimization. Defaults to 50.
            C_constraints (list, optional): list of concentration constraints to use. options: ["Nneg", "Trilinear", "Closure"]. Defaults to [].
            S_constraints (list, optional): list of spectral constraints to use. options: ["Nneg", "Closure"]. Defaults to [].
            n_experiments (int, optional): number of experiments in data, only use if data is trilinear. Defaults to None.
            concentration (_type_, optional): not used. Defaults to None.
            spectrum (_type_, optional): not used. Defaults to None.
            X_calibration (_type_, optional): _description_. Defaults to None.
            Y_calibration (_type_, optional): not used. Defaults to None.
            Correlation_Column (list, optional): not used. Defaults to [0].
            early_stopping (bool, optional): not used. Defaults to True.
            early_stopping_method (str, optional): not used. Defaults to 'residual'.
            tol (_type_, optional): not used. Defaults to 1e-3.
            verbose (bool, optional): additional info. Defaults to False.

        Returns:
            dict: mcr results
        """
        #Get required paramers from function call or from self
        #X
        if X is None:
            if self.X is not None:
                X = self.X
                if verbose: print("X was not specified. Stored X is used.")
            else: raise KeyError("X has not been specified")
        else:
            if verbose: print("Specified X has been stored in self") 
            self.X = X
        #Add calibration spectra if used:
        if X_calibration is not None:
            self.X = np.vstack([X_calibration, self.X])
            if verbose: print("Calibration spectra have been added to the data matrix")
        #n_comps
        if n_comps is None:
            if self.n_comps is not None:
                n_comps = self.n_comps
                if verbose: print(f'n_comps was not specified. Stored n_comps ({n_comps}) is used.')
            else: raise KeyError("n_comps has not been specified")
        else: 
            if verbose: print('Specified n_comps has been stored in self')
            self.n_comps = n_comps
        #n_experiments
        if n_experiments is None:
            if self.n_experiments is not None:
                n_experiments = self.n_experiments
                if verbose: print(f'n_experiments was not specified. Stored n_experiments ({n_experiments}) is used.')
                else: raise KeyError("n_experiments has not been specified")
        else: 
            if verbose: print('Specified n_experiments has been stored in self')
            self.n_experiments = n_experiments

        #Check if there are initial estimates, and calculate them if they are absent
        if self.init_C is None and self.init_S is None:
            print("!!! WARNING:There are no initial estimates for the concentration or spectral profiles. A random estimate will be used for the concentration profiles")
            self.estimate_initialcomps(X, n_comps)

        if verbose: 
            print(f'MCR-ALS will perform max. {n_iter} iterations. Number of components to model: {n_comps}')
            print(f'Selected constraints on concentration profiles: {C_constraints}')
            print(f'Selected constraints on spectral profiles: {S_constraints}')

        #Initialize C(oncentration) and S(pectral) profiles
        if self.init_method in ["pureC", "random"]:
            #Initial estimate is concentration profile
            C = self.init_C
            S = np.zeros((n_comps, X.shape[1]))
            startS = True
        elif self.init_method in ["pureS"]:
            #Initial estimate is spectral profile
            C = np.zeros((X.shape[0], n_comps))
            S = self.init_S
            startS=False

        #ALS Loop
        residuals = []
        lofs = []
        for i in tqdm(range(n_iter)):
            #Solve for S
            if startS:
                S = np.linalg.lstsq(C, X, rcond=None)[0]
                S = S[:n_comps, :]
                for constraint in S_constraints:
                    if constraint == "Nneg":
                        S = self.Constraint_NNeg(S)
                    elif constraint == "Closure":
                        S = self.Constraint_Closure(S, axis=1)
                    # elif constraint == "ImposeS":
                    #     if spectrum is None:
                    #         raise KeyError("Must provide a spectrum profile to impose on S")
                    #     S = self.ImposeS(S, spectrum)
                    else: raise KeyError("Specified S_constraint is not supported.")
            startS = True
            #Solve for C
            C = np.linalg.lstsq(S.T, X.T, rcond=None)[0].T
            C = C[:,:n_comps]
            for constraint in C_constraints:
                if constraint == "Nneg":
                    C = self.Constraint_NNeg(C)
                elif constraint == "Closure":
                    C = self.Constraint_Closure(C, axis=1)
                elif constraint == "Trilinear":
                    if n_experiments is None: raise KeyError("Specify n_experiments when using trilinearity")
                    C = self.Constraint_Trilinear(C, n_experiments)
                # elif constraint == "ImposeC":
                #     if concentration is None:
                #         raise KeyError("Must provide a concentration profile to impose on C")
                #     C = self.ImposeC(C, concentration)
                # elif constraint == "Correlation":
                #     if (X_calibration is None) | (Y_calibration is None):
                #         raise KeyError("Must provide calibration data with references (X_calibration and Y_calibration)")
                #     C = self.Constraint_Correlation(Y_calibration=Y_calibration, Correlation_Column=Correlation_Column, C=C)

                else: raise KeyError("Specified C_constraint is not supported.")
            #Calculate metrics
            X_recon = C @ S
            error = X - X_recon
            residual = np.linalg.norm(error)
            lof = 100*residual / np.linalg.norm(X)
            residuals.append(residual)
            lofs.append(lof)
            #Early stopping - not implemented
        
        #Remove calibration data from C
        if X_calibration is not None:
            num_cal = X_calibration.shape[0]
            C = C[num_cal:, :]

        self.mcr_results = {
            'C':C,
            'S':S,
            'lof':lofs[-1],
            'residual':residuals[-1],
            'lofs':lofs,
            'residuals':residuals
        }
        return self.mcr_results

    def Constraint_NNeg(self, X):
        """Applies nonnegativity

        Args:
            X (np array): component

        Returns:
            np array: non-negative component
        """
        return np.where(X < 0, 0, X)
    
    def Constraint_Closure(self, X, axis):
        """Applies closure

        Args:
            X (np array): component
            axis (int): 0 or 1: for normalizing spectra or concentration

        Returns:
            np array: normalized component
        """
        #Axis=1 normalizes rows (For normalizing total concentration)
        #Axis=0 normalizes columns (For normalizing spectral wavelengths)
        return X / np.sum(X, axis=axis, keepdims=True)
    
    # def Constraint_Correlation(self, Y_calibration, Correlation_Column, C):
    #     #NOTE: Correlation is automatically applied to the first column of C, unless otherwise specified. This matches the same profile used for imposing a spectral profile.
    #     #When using multiple columns for correlation, one must check which profile should use which calibration profile. 
    #     if Y_calibration.ndim ==1:
    #         Y_calibration == Y_calibration.reshape(-1,1)
    #     num_cal = Y_calibration.shape[0]
    #     #Loop for all concentration columns
    #     for i in range(Y_calibration.shape[1]):
    #         Y = Y_calibration[:,i].reshape(-1,1)
    #         C_cal = C[:num_cal, Correlation_Column[i]].reshape(-1,1)
    #         LR = LinearRegression()
    #         LR.fit(C_cal, Y)
    #         C[:, Correlation_Column[i]] = LR.predict(C[:, Correlation_Column[i]].reshape(-1,1)).flatten()
    #     return C        
    
    def Constraint_Trilinear(self, X, n_experiments):
        """Applies trilinear constraints

        Args:
            X (np array): C component
            n_experiments (int): number of experiments in data

        Returns:
            np array: trilinear component
        """
        n_total, n_components = X.shape
        n_samples = n_total // n_experiments
        X_new = X.copy()

        for comp in range(n_components):
            profiles = X[:, comp].reshape(n_experiments, n_samples).T
            Y = profiles.copy()

            ymax = np.max(profiles, axis=0)
            imax = np.argmax(profiles, axis=0)
            valid = ymax > 0
            ymin = np.min(imax[valid])
            imin = np.where(imax == ymin)[0][0]

            for j in range(n_experiments):
                if j != imin and ymax[j] > 0:
                    ishift = imax[j] - ymin
                    if ishift > 0:
                        Y[:-ishift, j] = profiles[ishift:, j]
                        Y[-ishift:, j] = 0
                    else:
                        Y[:, j] = profiles[:, j]

            U, S, Vt = np.linalg.svd(Y, full_matrices=False)
            t = -S[0] * Vt[0, :]
            cp2 = -np.outer(U[:, 0], t)

            for j in range(n_experiments):
                if j != imin and ymax[j] > 0:
                    ishift = imax[j] - ymin
                    if ishift > 0:
                        Y[:ishift, j] = 0
                        Y[ishift:, j] = cp2[:-ishift, j]
                    else:
                        Y[:, j] = cp2[:, j]
                else:
                    Y[:, j] = cp2[:, j]

            X_new[:, comp] = Y.T.flatten()

        return X_new
    
    # def ImposeS(self, X, spectra):
    #     num_spec = spectra.shape[0]
    #     for i in range(num_spec):
    #         X[i,:] = spectra[i,:]
    #     return X
    
    # def ImposeC(self, X, concentration):
    #     num_concentration = concentration.shape[0]
    #     for i in range(num_concentration):
    #         X[:,i] = concentration[:,1]
    #     return X

    
    def vis_C(self, ax=None, title="Concentration Components", xlabel="Sample number", ylabel="Concentration (a.u.)"):
        """Visualize concentration component

        Args:
            ax (matplotlib ax, optional): ax to plot on. Defaults to None.
            title (str, optional): -. Defaults to "Concentration Components".
            xlabel (str, optional): -. Defaults to "Sample number".
            ylabel (str, optional): -. Defaults to "Concentration (a.u.)".

        Returns:
            matplotlib ax: plot
        """
        if self.mcr_results is None:
            raise KeyError("MCR-ALS model has not yet been fitted.")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        cmap = plt.get_cmap('viridis')
        try: colors = [cmap(i / (self.n_comps - 1)) for i in range(self.n_comps)]
        except: colors=['yellow']
        for i in range(self.n_comps):
            ax.plot(self.mcr_results["C"][:,i], label=f'Component {i+1}', color=colors[i])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return ax
    
    def vis_S(self, ax=None, title="Spectral Components", xlabel="Variable number", ylabel="Intensity (a.u.)", xaxis=None):
        """Visualize spectral component

        Args:
            ax (matplotlib ax, optional): ax to plot on. Defaults to None.
            title (str, optional): -. Defaults to "Spectral Components".
            xlabel (str, optional): -. Defaults to "Variable number".
            ylabel (str, optional): -. Defaults to "Intensity (a.u.)".
            xaxis (np array, optional): wavelengths x axis. Defaults to None.

        Returns:
            matplotlib ax: plot
        """
        if self.mcr_results is None:
            raise KeyError("MCR-ALS model has not yet been fitted.")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        cmap = plt.get_cmap('viridis')
        try: colors = [cmap(i / (self.n_comps - 1)) for i in range(self.n_comps)]
        except: colors=['yellow']
        for i in range(self.n_comps):
            if xaxis is None:
                ax.plot(self.mcr_results["S"][i,:], label=f'Component {i+1}', color=colors[i])
            else: ax.plot(xaxis, self.mcr_results["S"][i,:], label=f'Component {i+1}', color=colors[i])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return ax
    
    def vis_lofs(self, ax=None, title="LOF '%' during fitting", xlabel="Iteration", ylabel="LOF %", round=True):
        """Visualize lack of fit over iterations

        Args:
            ax (matplotlib ax, optional): ax to plot on. Defaults to None.
            title (str, optional): -. Defaults to "LOF '%' during fitting".
            xlabel (str, optional): -. Defaults to "Iteration".
            ylabel (str, optional): -. Defaults to "LOF %".
            round (bool, optional): rounds down numbers. Defaults to True.=

        Returns:
            matplotlib ax: plot
        """
        if self.mcr_results is None:
            raise KeyError("MCR-ALS model has not yet been fitted.")
        
        if ax is None:
            fig, ax = plt.subplots()
        if round:
            lofs = np.round(self.mcr_results["lofs"], 3)
        else:
            lofs = self.mcr_results["lofs"]
        ax.plot(lofs)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def vis_residuals(self, ax=None, title="Residuals during fitting", xlabel="Iteration", ylabel="Residual (a.u.)", round=True):
        """Visualizes sum of residuals

        Args:
            ax (matplotlib ax, optional): ax to plot on. Defaults to None.
            title (str, optional): -. Defaults to "Residuals during fitting".
            xlabel (str, optional): -. Defaults to "Iteration".
            ylabel (str, optional): -. Defaults to "Residual (a.u.)".
            round (bool, optional): rounds down numbers. Defaults to True.

        Returns:
            matplotlib ax: plot
        """
        if self.mcr_results is None:
            raise KeyError("MCR-ALS model has not yet been fitted.")
        
        if ax is None:
            fig, ax = plt.subplots()
        if round:
            residuals = np.round(self.mcr_results["residuals"], 3)
        else:
            residuals = self.mcr_results["residuals"]
        ax.plot(residuals)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    
    def analyze(self, S_xaxis=None, round=True):
        """Analyze mcr results

        Args:
            S_xaxis (np array, optional): wavelengths x axis. Defaults to None.
            round (bool, optional): rounds down numbers. Defaults to True.
        """
        fig, axs = plt.subplots(2,2, figsize=(10,10))
        self.vis_lofs(ax=axs[0,0], round=round)
        self.vis_residuals(ax=axs[0,1], round=round)
        self.vis_C(ax=axs[1,0])
        self.vis_S(ax=axs[1,1], xaxis=S_xaxis)

    def __test__(self):
        """Testing code
        """
        #Simulate data
        num_samples = 50
        num_wavelengths = 100
        wavelengths = np.linspace(200, 400, num_wavelengths)
        component1 = np.exp(-0.5 * ((wavelengths - 250) / 10)**2)  
        component2 = np.exp(-0.5 * ((wavelengths - 320) / 15)**2)  
        concentration1 = np.linspace(1, 0, num_samples)       
        concentration2 = np.linspace(0, 1, num_samples)           
        C = np.vstack([concentration1, concentration2]).T  
        S = np.vstack([component1, component2])           
        D = C @ S  
        noise_level = 0.02
        D = D + noise_level * np.random.normal(size=D.shape)
        fig, ax = plt.subplots()
        ax.plot(D.T)
        plt.show()

        #Test MCR functionality
        self.estimate_ncomps(D, verbose=True)
        self.estimate_initialcomps(n_comps=2, verbose=True, method='pureC')
        self.fit(X=None, n_comps=2, n_iter=50, C_constraints = ["Nneg"], S_constraints = [], n_experiments=None, early_stopping=True, early_stopping_method='residual', tol=1e-3, verbose=True)
        self.analyze()
