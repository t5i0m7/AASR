function [mu,sig,alpha,beta] = test_eeg_dist_revi(X,min_clean_fraction,max_dropout_fraction,quants,step_sizes,beta)
% Estimate the mean and standard deviation of clean EEG from contaminated data.
% [Mu,Sigma,Alpha,Beta] = fit_eeg_distribution(X,MinCleanFraction,MaxDropoutFraction,FitQuantiles,StepSizes,ShapeRange)
% 
% This function estimates the mean and standard deviation of clean EEG from a sample of amplitude
% values (that have preferably been computed over short windows) that may include a large fraction
% of contaminated samples. The clean EEG is assumed to represent a generalized Gaussian component in
% a mixture with near-arbitrary artifact components. By default, at least 25% (MinCleanFraction) of
% the data must be clean EEG, and the rest can be contaminated. No more than 10%
% (MaxDropoutFraction) of the data is allowed to come from contaminations that cause lower-than-EEG
% amplitudes (e.g., sensor unplugged). There are no restrictions on artifacts causing
% larger-than-EEG amplitudes, i.e., virtually anything is handled (with the exception of a very
% unlikely type of distribution that combines with the clean EEG samples into a larger symmetric
% generalized Gaussian peak and thereby "fools" the estimator). The default parameters should be
% fine for a wide range of settings but may be adapted to accomodate special circumstances.
% 
% The method works by fitting a truncated generalized Gaussian whose parameters are constrained by
% MinCleanFraction, MaxDropoutFraction, FitQuantiles, and ShapeRange. The alpha and beta parameters
% of the gen. Gaussian are also returned. The fit is performed by a grid search that always finds a
% close-to-optimal solution if the above assumptions are fulfilled.
% 
% In:
%   X : vector of amplitude values of EEG, possible containing artifacts
%       (coming from single samples or windowed averages)
% 
%   MinCleanFraction : Minimum fraction of values in X that needs to be clean
%                      (default: 0.25)
% 
%   MaxDropoutFraction : Maximum fraction of values in X that can be subject to
%                        signal dropouts (e.g., sensor unplugged) (default: 0.1)
% 
%   FitQuantiles : Quantile range [lower,upper] of the truncated generalized Gaussian distribution
%                  that shall be fit to the EEG contents (default: [0.022 0.6])
% 
%   StepSizes : Step size of the grid search; the first value is the stepping of the lower bound
%               (which essentially steps over any dropout samples), and the second value
%               is the stepping over possible scales (i.e., clean-data quantiles)
%               (default: [0.01 0.01])
% 
%   ShapeRange : Range that the clean EEG distribution's shape parameter beta may take (default:
%                1.7:0.15:3.5)
% 
% Out:
%   Mu : estimated mean of the clean EEG distribution
% 
%   Sigma : estimated standard deviation of the clean EEG distribution
% 
%   Alpha : estimated scale parameter of the generalized Gaussian clean EEG distribution (optional)
% 
%   Beta : estimated shape parameter of the generalized Gaussian clean EEG distribution (optional)

%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2013-08-15

% Copyright (C) Christian Kothe, SCCN, 2013, ckothe@ucsd.edu
%
% This program is free software; you can redistribute it and/or modify it under the terms of the GNU
% General Public License as published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
% even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program; if not,
% write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
% USA

% assign defaults
if ~exist('min_clean_fraction','var') || isempty(min_clean_fraction)
    min_clean_fraction = 0.25; end
if ~exist('max_dropout_fraction','var') || isempty(max_dropout_fraction)
    max_dropout_fraction = 0.1; end
if ~exist('quants','var') || isempty(quants)
    quants = [0.022 0.6]; end
if ~exist('step_sizes','var') || isempty(step_sizes)
    step_sizes = [0.01 0.01]; end
if ~exist('beta','var') || isempty(beta)
    beta = 1.7:0.15:3.5; end

% sanity checks
if ~isvector(quants) || numel(quants) > 2
    error('Fit quantiles needs to be a 2-element vector (support for matrices deprecated).'); end
if any(quants(:)<0) || any(quants(:)>1)
    error('Unreasonable fit quantiles.'); end
if any(step_sizes<0.0001) || any(step_sizes>0.1)
    error('Unreasonable step sizes.'); end
if any(beta>=7) || any(beta<=1)
    error('Unreasonable shape range.'); end

% sort data so we can access quantiles directly
X = double(sort(X(:)));
n = length(X);
zbounds =[];

% calc z bounds for the truncated standard generalized Gaussian pdf and pdf rescaler
for b=1:length(beta)    
    zbounds = cat(1,zbounds,sign(quants-1/2).*gammaincinv(sign(quants-1/2).*(2*quants-1),1/beta(b)).^(1/beta(b))); %#ok<*AGROW>
    rescale(b) = beta(b)/(2*gamma(1/beta(b)));
end
zbounds= zbounds';
% determine the quantile-dependent limits for the grid search
% quants = [0.022 0.6]
% lower_min = 0.22;max_width = 0.578
lower_min = min(quants);                    % we can generally skip the tail below the lower quantile
max_width = diff(quants);                   % maximum width is the fit interval if all data is clean
min_width = min_clean_fraction*max_width;   % minimum width of the fit interval, as fraction of data

% get matrix of shifted data ranges
% X: 1 x M
% (1:round(n*max_width))' => select max width of RMS's index
% round(n*(lower_min:step_sizes(1):lower_min+max_dropout_fraction))
% => select offset
X = X(bsxfun(@plus,(1:round(n*max_width))',round(n*(lower_min:step_sizes(1):lower_min+max_dropout_fraction))));
X1 = X(1,:); 
X = bsxfun(@minus,X,X1);
opt_val = inf;
% for each interval width...
% n*(0.578:-0.01:0.25*0.578)
for m = unique(round(n*(max_width:-step_sizes(2):min_width)))
    % scale and bin the data in the intervals
    nbins = round(3*log2(1+m/2));
    H = bsxfun(@times,X(1:m,:),nbins./X(m,:));
    logq = log(histc(H,[0:nbins-1,Inf]) + 0.01);
    
    
    b_t = zbounds;
    x_m = b_t(1,:)'+(0.5:(nbins-0.5))/nbins.*diff(b_t)';
    
    p_m = exp(-abs(x_m).^repmat(beta',[1 size(x_m,2)])).*rescale';
    p_m = p_m./sum(p_m,2);
    logq_m = repmat(reshape(logq(1:end-1,:),[1 size(logq(1:end-1,:),1) size(logq(1:end-1,:),2)]),[size(p_m,1) 1 1]);
    
    kl_m =sum(repmat(p_m,[1 1 size(logq,2)]).*(log(repmat(p_m,[1 1 size(logq,2)])) - logq_m),2)+ log(m);
    kl_m = reshape(kl_m,[size(kl_m,1) size(kl_m,3)]); 
    
    [min_val_t,idx] = min(kl_m,[],2);
    [min_val,idx_b] = min(min_val_t);
    
    if min_val > opt_val
        continue;
    end
    opt_val = min_val;
    opt_beta = beta(idx_b);
    opt_bounds = b_t(:,idx_b)';
    opt_lu = [X1(idx(idx_b)) X1(idx(idx_b))+X(m,idx(idx_b))];
    
end

% recover distribution parameters at optimum
alpha = (opt_lu(2)-opt_lu(1))/diff(opt_bounds);
mu = opt_lu(1)-opt_bounds(1)*alpha;
beta = opt_beta;

% calculate the distribution's standard deviation from alpha and beta
sig = sqrt((alpha^2)*gamma(3/beta)/gamma(1/beta));