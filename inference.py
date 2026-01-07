"""
Inference methods for Bayesian active learning.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from config import DEVICE, PRIOR_VARIANCE, NOISE_VARIANCE


class AnalyticGaussianInference:
    """Analytical Gaussian inference with closed-form posterior."""
    
    def __init__(self, prior_var=PRIOR_VARIANCE, noise_var=NOISE_VARIANCE, device=DEVICE):
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.device = device
        self.W_mean = None
        self.W_cov = None
    
    def fit(self, X, Y):
        # Convert to numpy for matrix operations
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        Y_np = Y.cpu().numpy() if torch.is_tensor(Y) else Y
        
        N, D = X_np.shape
        
        # Prior precision
        prior_precision = (1.0 / self.prior_var) * np.eye(D)
        
        # Compute posterior parameters
        XTX = X_np.T @ X_np
        posterior_cov = np.linalg.inv((1.0 / self.noise_var) * XTX + prior_precision)
        XTY = X_np.T @ Y_np
        posterior_mean = posterior_cov @ ((1.0 / self.noise_var) * XTY)
        
        # Store as tensors
        self.W_mean = torch.from_numpy(posterior_mean).float().to(self.device)
        self.W_cov = torch.from_numpy(posterior_cov).float().to(self.device)
    
    def predict(self, X, n_samples=None):
        if self.W_mean is None:
            raise ValueError("Model not fitted. Call fit() before predict().")
        
        X_tensor = X.to(self.device) if torch.is_tensor(X) else torch.from_numpy(X).float().to(self.device)
        
        # Predictive distribution
        pred_mean = X_tensor @ self.W_mean
        pred_var = self.noise_var + torch.sum((X_tensor @ self.W_cov) * X_tensor, dim=1)
        
        return pred_mean, pred_var
    
    def rmse(self, X, Y):
        pred_mean, _ = self.predict(X)
        mse = torch.mean((pred_mean - Y) ** 2)
        return torch.sqrt(mse).item()


class MFVIDiagonalInference:
    """Mean Field Variational Inference with diagonal covariance."""
    
    def __init__(self, feature_dim=128, output_dim=10, prior_var=PRIOR_VARIANCE, 
                 noise_var=NOISE_VARIANCE, device=DEVICE):
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.device = device
        
        # Variational parameters
        self.m = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        self.log_s2 = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        
        # Best parameters tracking
        self.best_m = None
        self.best_log_s2 = None
    
    def sample_weights(self):
        epsilon = torch.randn_like(self.m)
        return self.m + torch.exp(0.5 * self.log_s2) * epsilon
    
    def elbo(self, X, Y, n_samples=3):
        batch_size = X.size(0)
        total_elbo = 0.0
        
        for _ in range(n_samples):
            W = self.sample_weights()
            pred = X @ W
            const_term = -0.5 * batch_size * np.log(2 * np.pi * self.noise_var)
            data_term = -0.5 * torch.sum((Y - pred) ** 2) / self.noise_var
            total_elbo += (const_term + data_term) / n_samples
        
        # KL divergence
        s2 = torch.exp(self.log_s2)
        kl_per_param = 0.5 * (s2 / self.prior_var + self.m ** 2 / self.prior_var - 1 - self.log_s2 + np.log(self.prior_var))
        kl_div = torch.sum(kl_per_param)
        
        return total_elbo - kl_div
    
    def fit(self, X, Y, n_iter=30, lr=0.01):
        optimizer = optim.Adam([self.m, self.log_s2], lr=lr)
        best_elbo = -float('inf')
        
        for i in range(n_iter):
            optimizer.zero_grad()
            elbo_val = self.elbo(X, Y, n_samples=3)
            loss = -elbo_val
            loss.backward()
            optimizer.step()
            
            if elbo_val > best_elbo:
                best_elbo = elbo_val
                self.best_m = self.m.detach().clone()
                self.best_log_s2 = self.log_s2.detach().clone()
        
        # Restore best parameters
        if self.best_m is not None:
            self.m.data = self.best_m
            self.log_s2.data = self.best_log_s2
    
    def predict(self, X, n_samples=10):
        pred_means = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                W = self.sample_weights()
                pred = X @ W
                pred_means.append(pred.unsqueeze(0))
        
        pred_means_all = torch.cat(pred_means, dim=0)
        pred_mean = pred_means_all.mean(dim=0)
        pred_var = pred_means_all.var(dim=0).mean(dim=1)
        
        return pred_mean, pred_var
    
    def rmse(self, X, Y):
        pred_mean, _ = self.predict(X, n_samples=5)
        mse = torch.mean((pred_mean - Y) ** 2)
        return torch.sqrt(mse).item()


class MFVIFullCovarianceInference:
    """MFVI with full covariance matrix."""
    
    def __init__(self, feature_dim=128, output_dim=10, prior_var=PRIOR_VARIANCE, 
                 noise_var=NOISE_VARIANCE, device=DEVICE):
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # Flattened variational parameters
        param_dim = feature_dim * output_dim
        self.m = nn.Parameter(torch.zeros(param_dim, device=device))
        self.L_tril = nn.Parameter(torch.eye(param_dim, device=device) * 0.1)
        
        # Best parameters tracking
        self.best_m = None
        self.best_L_tril = None
    
    def get_covariance_matrix(self):
        L = torch.tril(self.L_tril)
        return L @ L.T + 1e-6 * torch.eye(self.feature_dim * self.output_dim, device=self.device)
    
    def sample_weights(self, n_samples=1):
        L = torch.tril(self.L_tril)
        epsilon = torch.randn(n_samples, self.feature_dim * self.output_dim, device=self.device)
        samples = self.m + epsilon @ L.T
        return samples.view(n_samples, self.feature_dim, self.output_dim)
    
    def elbo(self, X, Y, n_samples=3):
        batch_size = X.size(0)
        total_elbo = 0.0
        
        for _ in range(n_samples):
            W = self.sample_weights(1).squeeze(0)
            pred = X @ W
            const_term = -0.5 * batch_size * self.output_dim * np.log(2 * np.pi * self.noise_var)
            data_term = -0.5 * torch.sum((Y - pred) ** 2) / self.noise_var
            total_elbo += (const_term + data_term) / n_samples
        
        # KL divergence (full covariance)
        Σ = self.get_covariance_matrix()
        k = self.feature_dim * self.output_dim
        term1 = torch.trace(Σ / self.prior_var)
        term2 = torch.sum(self.m ** 2) / self.prior_var
        term3 = k * np.log(self.prior_var) - torch.logdet(Σ)
        kl_div = 0.5 * (term1 + term2 - k + term3)
        
        return total_elbo - kl_div
    
    def fit(self, X, Y, n_iter=20, lr=0.005):
        optimizer = optim.Adam([self.m, self.L_tril], lr=lr)
        best_elbo = -float('inf')
        
        for i in range(n_iter):
            optimizer.zero_grad()
            elbo_val = self.elbo(X, Y, n_samples=2)
            loss = -elbo_val
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.m, self.L_tril], 1.0)
            optimizer.step()
            
            if elbo_val > best_elbo:
                best_elbo = elbo_val
                self.best_m = self.m.detach().clone()
                self.best_L_tril = self.L_tril.detach().clone()
        
        # Restore best parameters
        if self.best_m is not None:
            self.m.data = self.best_m
            self.L_tril.data = self.best_L_tril
    
    def predict(self, X, n_samples=10):
        pred_means = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                W = self.sample_weights(1).squeeze(0)
                pred = X @ W
                pred_means.append(pred.unsqueeze(0))
        
        pred_means_all = torch.cat(pred_means, dim=0)
        pred_mean = pred_means_all.mean(dim=0)
        pred_var = pred_means_all.var(dim=0).mean(dim=1)
        
        return pred_mean, pred_var
    
    def rmse(self, X, Y):
        pred_mean, _ = self.predict(X, n_samples=5)
        mse = torch.mean((pred_mean - Y) ** 2)
        return torch.sqrt(mse).item()


class HeteroscedasticLastLayer:
    """Heteroscedastic last layer with uncertainty decomposition."""
    
    def __init__(self, feature_dim=128, output_dim=10, prior_var=PRIOR_VARIANCE, device=DEVICE):
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.prior_var = prior_var
        self.device = device
        
        # Variational parameters for mean
        self.mean_m = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        self.mean_log_s2 = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        
        # Variational parameters for variance
        self.logvar_m = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        self.logvar_log_s2 = nn.Parameter(torch.zeros(feature_dim, output_dim, device=device))
        
        # Best parameters tracking
        self.best_params = None
    
    def sample_weights(self):
        epsilon_mean = torch.randn_like(self.mean_m)
        W_mean = self.mean_m + torch.exp(0.5 * self.mean_log_s2) * epsilon_mean
        
        epsilon_logvar = torch.randn_like(self.logvar_m)
        W_logvar = self.logvar_m + torch.exp(0.5 * self.logvar_log_s2) * epsilon_logvar
        
        return W_mean, W_logvar
    
    def elbo(self, X, Y, n_samples=3):
        total_elbo = 0.0
        
        for _ in range(n_samples):
            W_mean, W_logvar = self.sample_weights()
            pred_mean = X @ W_mean
            pred_logvar = X @ W_logvar
            pred_var = torch.exp(pred_logvar)
            
            # Heteroscedastic Gaussian log likelihood
            log_lik = -0.5 * torch.sum(
                np.log(2 * np.pi) + pred_logvar + (Y - pred_mean)**2 / pred_var
            )
            
            # KL divergence for mean parameters
            s2_mean = torch.exp(self.mean_log_s2)
            kl_mean = 0.5 * torch.sum(
                s2_mean / self.prior_var + self.mean_m**2 / self.prior_var - 1 - self.mean_log_s2 + np.log(self.prior_var)
            )
            
            # KL divergence for variance parameters
            s2_logvar = torch.exp(self.logvar_log_s2)
            kl_logvar = 0.5 * torch.sum(
                s2_logvar / self.prior_var + self.logvar_m**2 / self.prior_var - 1 - self.logvar_log_s2 + np.log(self.prior_var)
            )
            
            total_elbo += (log_lik - kl_mean - kl_logvar) / n_samples
        
        return total_elbo
    
    def fit(self, X, Y, n_iter=30, lr=0.01):
        params = [self.mean_m, self.mean_log_s2, self.logvar_m, self.logvar_log_s2]
        optimizer = optim.Adam(params, lr=lr)
        best_elbo = -float('inf')
        
        for i in range(n_iter):
            optimizer.zero_grad()
            elbo_val = self.elbo(X, Y, n_samples=3)
            loss = -elbo_val
            loss.backward()
            optimizer.step()
            
            if elbo_val > best_elbo:
                best_elbo = elbo_val
                self.best_params = {
                    'mean_m': self.mean_m.detach().clone(),
                    'mean_log_s2': self.mean_log_s2.detach().clone(),
                    'logvar_m': self.logvar_m.detach().clone(),
                    'logvar_log_s2': self.logvar_log_s2.detach().clone()
                }
        
        # Restore best parameters
        if self.best_params is not None:
            self.mean_m.data = self.best_params['mean_m']
            self.mean_log_s2.data = self.best_params['mean_log_s2']
            self.logvar_m.data = self.best_params['logvar_m']
            self.logvar_log_s2.data = self.best_params['logvar_log_s2']
    
    def predict(self, X, n_samples=10):
        pred_means = []
        pred_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                W_mean, W_logvar = self.sample_weights()
                pred_mean_sample = X @ W_mean
                pred_logvar_sample = X @ W_logvar
                pred_var_sample = torch.exp(pred_logvar_sample)
                pred_means.append(pred_mean_sample.unsqueeze(0))
                pred_vars.append(pred_var_sample.unsqueeze(0))
        
        pred_means_all = torch.cat(pred_means, dim=0)
        pred_vars_all = torch.cat(pred_vars, dim=0)
        
        pred_mean = pred_means_all.mean(dim=0)
        epistemic_var = pred_means_all.var(dim=0).mean(dim=1)
        aleatoric_var = pred_vars_all.mean(dim=0).mean(dim=1)
        total_var = epistemic_var + aleatoric_var
        
        return pred_mean, total_var, aleatoric_var, epistemic_var
    
    def predictive_entropy(self, X, n_samples=10):
        _, total_var, _, _ = self.predict(X, n_samples=n_samples)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * total_var)
        return entropy
    
    def rmse(self, X, Y):
        pred_mean, _, _, _ = self.predict(X, n_samples=10)
        mse = torch.mean((pred_mean - Y) ** 2)
        return torch.sqrt(mse).item()