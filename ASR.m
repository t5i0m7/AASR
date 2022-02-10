classdef ASR
    %ASR Summary of this class goes here
    %   Detailed explanation goes here

    properties
        srate
        cutoff
        filter_A
        filter_B
        ASR_T
        ASR_M
        state
        fsm
    end

    methods
        function obj = ASR(cutoff,srate)
            %ASR Construct an instance of this class
            %   Detailed explanation goes here
            obj.srate = srate;
            obj.cutoff = cutoff;
            [B,A] = yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]);
            obj.filter_A = A;
            obj.filter_B = B;
            obj.state = struct('M',[],'T',[],'B',obj.filter_B,'A',obj.filter_A,'cov',[],'carry',[],'iir',[],'last_R',[],'last_trivial',true);
            obj.fsm = [];
        end

        function data_ASR = reconstruct(obj,data)
            % extrapolate last few samples of the signal
            % data , data + data- data(end-1:-1:end-0.5/2*500)
            availableRAM_GB=8;
            asr_windowlen = 0.5;
            sig = [data bsxfun(@minus,2*data(:,end),...
                data(:,(end-1):-1:end-round(asr_windowlen/2*obj.srate)))];

            asr_stepsize = [];

            maxdims = 0.66;

            [data_ASR,obj.state] = test_asr_process(sig,obj.srate,obj.state,asr_windowlen,asr_windowlen/2,asr_stepsize,maxdims,availableRAM_GB,[]);
            data_ASR(:,1:size(obj.state.carry,2)) = [];
        end

        function data_clean = findClean(obj,data)
            clwin_window_len = 1 ; % how many second you take in window
            window_overlap = 0.66 ; % overlap partition


            [C,S] = size(data);

            N = clwin_window_len*obj.srate;

            wnd = 0:N-1;

            offsets = round(1:N*(1-window_overlap):S-N);

            % for each channel...
            % c * segment  z scores

            min_clean_fraction = 0.25;

            max_dropout_fraction = 0.1;

            truncate_quant = [0.022 0.6];

            clwin_step_sizes = [0.01 0.01];

            shape_range = 1.7:0.15:3.5;

            for c = C:-1:1
                % compute RMS amplitude for each window...
                X = data(c,:).^2;
                % compute all RMS from each sub windows
                X = sqrt(sum(X(bsxfun(@plus,offsets,wnd')))/N);
                % robustly fit a distribution to the clean EEG part
                [mu,sig] = test_eeg_dist_revi(X, ...
                    min_clean_fraction, max_dropout_fraction, ...
                    truncate_quant, clwin_step_sizes,shape_range);
                % calculate z scores relative to that
                wz(c,:) = (X - mu)/sig;
            end


            %
            max_bad_channels = 0.2;

            max_bad_channels = round(size(data,1)*max_bad_channels);

            zthresholds = [-3.5 5];

            % sort z scores into quantiles
            swz = sort(wz);
            % determine which windows to remove
            remove_mask = false(1,size(swz,2));
            if max(zthresholds)>0
                remove_mask(swz(end-max_bad_channels,:) > max(zthresholds)) = true; end
            if min(zthresholds)<0
                remove_mask(swz(1+max_bad_channels,:) < min(zthresholds)) = true; end
            removed_windows = find(remove_mask);


            % find indices of samples to remove % remoived offset + 0:499
            removed_samples = repmat(offsets(removed_windows)',1,length(wnd))+repmat(wnd,length(removed_windows),1);

            % mask them out
            sample_mask = true(1,S);
            sample_mask(removed_samples(:)) = false;

            data_clean = data(:,sample_mask);
        end


        function  obj = subspace(obj,data)
            X = obj.findClean(data);

            [C,S] = size(X);
            
            window_len = 0.5;


            min_clean_fraction = 0.25;
            window_overlap=  0.66 ;
            max_dropout_fraction = 0.1;

            X(~isfinite(X(:))) = 0;
            [X,iirstate] = filter(obj.filter_B,obj.filter_A,double(X),[],2);
            uc_data = X;
            X = X';


           
            M = covInASR(obj,uc_data);
            
            obj.ASR_M = M;
            
            
            % window length for calculating thresholds
            N = round(window_len*obj.srate);

            [V,~] = eig(M); % #ok<NASGU>
            X = abs(X*V);

            for c = C:-1:1
                % compute RMS amplitude for each window...
                rms = X(:,c).^2;
                rms = sqrt(sum(rms(bsxfun(@plus,round(1:N*(1-window_overlap):S-N),(0:N-1)')))/N);
                % fit a distribution to the clean part
                [mu(c),sig(c)] = test_eeg_dist_revi(rms,min_clean_fraction,max_dropout_fraction);
            end
            T = diag(mu + obj.cutoff*sig)*V';

            obj.ASR_T = T;
            obj.state = struct('M',M,'T',T,'B',obj.filter_B,'A',obj.filter_A,'cov',[],'carry',[],'iir',iirstate,'last_R',[],'last_trivial',true);



            if isempty(obj.fsm)

%
                Y_0 = V' * (uc_data);
                M_0 = Y_0 * Y_0' * (1/length(Y_0));
                W_0 = Y_0 * uc_data' * (1/length(uc_data));
                obj.fsm = FSM(8, 8,0.8, M_0 ,W_0, 0.2);
            end


        end

        function obj = update(obj,data)

            if isempty(obj.state.M)
                obj = obj.subspace(data);
                return;
            end
            
             c_data = obj.findClean(data);
   
            [uc_data,~] = filter(obj.state.B, obj.state.A,double(c_data),[],2);


            obj.fsm.fit_next(uc_data);

            V_ = obj.fsm.get_components([]);

            % online state
            obj.ASR_M = covInASR(obj,uc_data);

            M = obj.ASR_M;


            % state calculation

            X = uc_data';
            X = abs(X*V_);
            N = round(0.1*obj.srate);

            min_clean_fraction = 0.25;
            window_overlap=  0.66 ;
            max_dropout_fraction = 0.1;


            for c = size(X,2):-1:1
                % compute RMS amplitude for each window...
                rms = X(:,c).^2;
                rms = sqrt(sum(rms(bsxfun(@plus,round(1:N*(1-window_overlap):size(X,1)-N),(0:N-1)')))/N);
                % fit a distribution to the clean part
                [mu(c),sig(c)] = test_eeg_dist_revi(rms,min_clean_fraction,max_dropout_fraction);
            end
            T = diag(mu + obj.cutoff*sig)*V_';

            obj.state.M = M;
            obj.state.T = T;

        end

        function M = covInASR(~,data)
            [C,S] = size(data);
            blocksize = 10 ;
            U = zeros(length(1:blocksize:S),C*C);
            X = data';
            for k=1:blocksize
                % S compare k:blocksize:(S+k-1) vector, take minimum
                range = min(S,k:blocksize:(S+k-1));
                % mixing matrix => index
                % U = 0
                % B = range' * range'
                % U = U + B
                U = U + reshape(bsxfun(@times,reshape(X(range,:),[],1,C),reshape(X(range,:),[],C,1)),size(U));
            end
            % get the mixing matrix M
            M = sqrtm(real(reshape(block_geometric_median(U/blocksize),C,C)));

%                M = sqrtm(cov(data'));

        end

    end

end
