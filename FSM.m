classdef FSM < handle
    %{
    
    Parameters:
    ====================
    X             -- Numpy array of size d-by-n, where each column corresponds to one observation
    q             -- Dimension of PCA subspace to learn, must satisfy 1 <= q <= d
    tau           -- Learning rate scale parameter for M vs W (see Pehlevan et al.)
    Minv0         -- Initial guess for the inverse of the lateral weight matrix M, must be of size q-by-q
    W0            -- Initial guess for the forward weight matrix W, must be of size q-by-d
    Methods:
    ========
    fit_next()
    Output:
    ====================
    Minv -- Final iterate of the inverse lateral weight matrix, of size q-by-q (sometimes)
    W    -- Final iterate of the forward weight matrix, of size q-by-d (sometimes)
    
    %}
   
    
   properties
      lr
      t      
      k
      d
      tau
      Minv
      M
      W
      outer_W
      outer_Minv
      y
      
   end
   
   methods
      function obj = FSM(k, d, tau0, M, W0, learning_rate)
           obj.k = k;
          obj.d = d;
          if isempty(W0)   
              obj.W = randn(k,d)/d;
          else
              obj.W = W0;
          end
          if isempty(M)
              obj.Minv = eye(k);
          else
              obj.Minv = inv(M);
              obj.M = M;
          end
          if isempty(tau0)
              obj.tau = 0.5;
          else
              obj.tau = tau0;
          end
          
          if isempty(learning_rate)
              obj.lr =  0.2;
%               obj.lr = @(t_in) 1.0 / (2 * t_in + 5);
              
          else
              obj.lr = learning_rate;
          end
          obj.t = 0;
          obj.outer_W = zeros(size(obj.W));
          obj.outer_Minv = zeros(size(obj.Minv));
          self.y = [];
      end
                
      function y = fit_next(obj,x)
          % return the projection into the subspace 
          % x : d by n 
          y = obj.Minv * (obj.W * x);          
          %Plasticity, using gradient ascent/descent
          %W <- W + 2 eta(t) * (y*x' - W)
          
%           step = obj.lr(obj.t);
          step = obj.lr;          
          obj.outer_W = (2 * step * y) * x' / size(x,2);
          obj.W = (1 - 2 * step) * obj.W + obj.outer_W;          
          
          % M <- M + eta(self.t)/tau * (y*y' - M)
          step = step / obj.tau;
%           obj.Minv = obj.Minv / (1 - step);
%           z = obj.Minv * y;
%           c = step / (1 + step * z' * y);
%           obj.outer_Minv  = (c .* z) * z';
          obj.M = obj.M + step*(y*y'/size(x,2)- obj.M);

          obj.Minv = inv(obj.M);
%           
%           obj.lr = obj.lr * 0.6;
          
      end
                
      
      function components = get_components(obj, orthogonalize)       
        %{       
        Extract components from object
         Parameters
         ---------
         orthogonalize: bool
             whether to orthogonalize when computing the error
 
         Returns
         -------
         components: ndarray 
         %}
          
          if isempty(orthogonalize)
              orthogonalize = 1;
          end
              
          components = (obj.Minv*obj.W)';
          
          if orthogonalize
                [components,~ ] = qr(components,0);
                
          end  
          
      end
      
   end
end
