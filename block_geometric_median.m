function y = block_geometric_median(X,blocksize,varargin)
% Calculate a blockwise geometric median (faster and less memory-intensive 
% than the regular geom_median function).
%
% This statistic is not robust to artifacts that persist over a duration that
% is significantly shorter than the blocksize.
%
% In:
%   X : the data (#observations x #variables)
%   blocksize : the number of successive samples over which a regular mean 
%               should be taken
%   tol : tolerance (default: 1.e-5)
%   y : initial value (default: median(X))
%   max_iter : max number of iterations (default: 500)
%
% Out:
%   g : geometric median over X
%
% Notes:
%   This function is noticably faster if the length of the data is divisible by the block size.
%   Uses the GPU if available.
% 

if nargin < 2 || isempty(blocksize)
    blocksize = 1; end

if blocksize > 1
    [o,v] = size(X);       % #observations & #variables
    r = mod(o,blocksize);  % #rest in last block
    b = (o-r)/blocksize;   % #blocks
    if r > 0
        X = [reshape(sum(reshape(X(1:(o-r),:),blocksize,b*v)),b,v); sum(X((o-r+1):end,:))*(blocksize/r)];
    else
        X = reshape(sum(reshape(X,blocksize,b*v)),b,v);
    end
end

try
    y = gather(geometric_median(gpuArray(X),varargin{:}))/blocksize;
catch
    y = geometric_median(X,varargin{:})/blocksize;
end



function y = geometric_median(X,tol,y,max_iter)
% Calculate the geometric median for a set of observations (mean under a Laplacian noise distribution)
% This is using Weiszfeld's algorithm.
%
% In:
%   X : the data, as in mean
%   tol : tolerance (default: 1.e-5)
%   y : initial value (default: median(X))
%   max_iter : max number of iterations (default: 500)
%
% Out:
%   g : geometric median over X

if ~exist('tol','var') || isempty(tol)
    tol = 1.e-5; end
if ~exist('y','var') || isempty(y)
    y = median(X); end
if ~exist('max_iter','var') || isempty(max_iter)
    max_iter = 500; end

for i=1:max_iter
    invnorms = 1./sqrt(sum(bsxfun(@minus,X,y).^2,2));
    [y,oldy] = deal(sum(bsxfun(@times,X,invnorms)) / sum(invnorms),y);
    if norm(y-oldy)/norm(y) < tol
        break; end
end