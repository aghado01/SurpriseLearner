nrtrials = 1000;
tau = 1-10^-4; %probability of context changing

%experimentally controlled context at each time t
%ctxt = 1;
%context = ones(nrtrials,1)*ctxt; % the context from which cues will be drawn on each trial
context = [ones(1, round(nrtrials/3))*4 ones(1, round(nrtrials/3))*7 ones(1, round(nrtrials/3))*1 1]; %similar to current experimental structure

% eta instantiates discrimination difficulty
eta = [.2 .3 .4 .5 .6 .7 .8]; %probabilities of generating L for each stimulus, psychometric function
p_xtk_zt = @(zt,x_tk) ((1-eta(zt)).^(1-x_tk).*eta(zt).^x_tk); %generates observations from a given cue
theta = [.1 .9];

%cue probabilities
n = [0.05;0.05;0.05;0.05;0.05;0.05;0.05];
pcue = [.8 .2; .7 .3; .6 .4; .5 .5; .4 .6; .3 .7; .2 .8];
pcue = [pcue(:,1)-n/2 n pcue(:,2)-n/2];
h = length(n);

%needed to generate stimuli
p_zt_d_true = [1/3 1/3 1/3 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1/3 1/3 1/3];
p_dt_zt = [1 1 1 1/2 0 0 0; 0 0 0 1/2 1 1 1];

p_zt_dt(1,:) = p_dt_zt(1,:)/sum(p_dt_zt(1,:)); %should be a way to do this in one line
p_zt_dt(2,:) = p_dt_zt(2,:)/sum(p_dt_zt(2,:));

% monkey expects contexts to not change (self transition) with probability tau
% with uniform uncertainty over remaining contexts

p_yt_yt_2 = tau*eye(7);
tmp = ones(7)-eye(7);
p_yt_yt_2 = p_yt_yt_2+tmp*((1-tau)/(length(p_yt_yt_2)-1));
clear tmp

%monkey: likelyhood of having seen a direction
%experimenter: proportion of right and left outcomes
%this determines the distribution of feedback given
p_rt_yt_2 = [[pcue(:,1) + pcue(:,2)/2] [pcue(:,3) + pcue(:,2)/2]]; %this is also tbnonull, assumes normalized inputs

%initial belief about the context
p_hat = 1/7; %initially uniform expectations over all contexts
B_yt_R2 = [(1-p_hat)/(h-1) (1-p_hat)/(h-1) (1-p_hat)/(h-1) p_hat (1-p_hat)/(h-1) (1-p_hat)/(h-1) (1-p_hat)/(h-1)]';

%initial belief about direction
B_d_Rk2 = [.5 .5]'; %R
B_yt_R = [.5 .5]'; %R
B_d_Rk = B_yt_R; %R

TonicNE(1) = 1-max(B_yt_R); %analogous to yu and dayan 2005

dt = 0;

for t =2:nrtrials

	%generate a specific stimulus with direction and difficulty
	d_true(t) = find(mnrnd(1,pcue(context(t),:))); %1 = left, 2 = null, 3 = right
	z(t) = find(mnrnd(1,p_zt_d_true(d_true(t),:))); % 1 2 3 = right, 4 = null, 5 6 7 = left
	d(t) = [0 1]*mnrnd(1,p_dt_zt(:,z(t)))'; %0 = Left 1 = Right

	% initialize observer's beliefs for next trial
	% this implements the bias effects, so it may change
	B_d_Rk2(:,1)= p_rt_yt_2'*B_yt_R2(:,t-1);
	B_d_Rk2 = B_d_Rk2 / sum(B_d_Rk2);

	k = 1;
	B_d_R{t}(:,1)=B_d_Rk2(:,1); %include the initial starting point of the drift diffusion

	while B_d_Rk2(1,k) > theta(1) & B_d_Rk2(1,k) < theta(2)

		dt = dt +1;
		k = k+1;
		% generate observation for xt given the bernoulli draw
		x_tk_zt(k) = p_xtk_zt(z(t),1) > rand; %index 1 = R. (0 = LEFT) % consistent with a?
		%x_tk_zt(k) = binornd(1,eta(z(t)));

		% originally p_xtk_zt(z(t),1)
		% should this be based on belief about the context, or true context?
		% if the context was changing, 1 would change to

		% compute the likelyhood term p(xtk | dtk = i)
		% should p_xtk_dt values be symmetric given values of xtk 1 and 0?
		% should relate to psychometric function
		p_xtk_dt(1,k) = p_xtk_zt([1:7],x_tk_zt(k))*p_zt_dt(1,:)'; % p(x_tk|d_tk=1)
		p_xtk_dt(2,k) = p_xtk_zt([1:7],x_tk_zt(k))*p_zt_dt(2,:)'; % p(x_tk|d_tk=2)
		nf = p_xtk_dt(1,k) + p_xtk_dt(2,k); % normalization factor
		p_xtk_dt(:,k) = p_xtk_dt(:,k)./nf;

		% b(t_tk=i|{x_tk}k=1..k-1, R_t-1) = ?_j(d_tk=i | d_t,k-1=j) . b(d_tk-1=j | {x_tk}k=1..k-1, R_t-1 )
		% b(d_tk=i|d_k-1=j) = I : transition is identity therefore no dynamics
		% B(d_tk=i|{x_tk}k=1.k, R_t-1) = p(x_tk|d_tk=i) * b(d_tk=i|{x_tk}k=1..k-1, R_t-1) ; R_t-1 = (0L|1R);

		%compute the prior term b(dtk = i | xtk{k=1:k-1}, Rt-1, Rt-2..)
		B_d_Rk2(1,k) = p_xtk_dt(1,k) * B_d_Rk2(1,k-1); %belief in direction at the previous k timestep
		B_d_Rk2(2,k) = p_xtk_dt(2,k) * B_d_Rk2(2,k-1);
		nf = (B_d_Rk2(1,k) + B_d_Rk2(2,k)); %normalize
		B_d_Rk2(:,k) = B_d_Rk2(:,k)/nf;

		emissions{t}(k) = x_tk_zt(k);
		B_d_R{t}(:,k)=B_d_Rk2(:,k);
		NE{t}(k) = B_d_Rk2(2,k)/B_d_Rk2(2,1);

	end

	feedback(t) = d(t);
	difficulty(t) = z(t);
	RT(t)=k;
	choice(t) = (B_d_Rk2(1,k) < B_d_Rk2(2,k)); % 0 is left, 1 is right - estimate of direction D (d^), this is the response given

	%the likelyhood of having seen the stimulus that did not occur is zero, so we only update using
	%likelyhoods for the true stimulus presented.
	B_yt_R2(:,t) = p_rt_yt_2(:,d(t)+1).*(p_yt_yt_2*B_yt_R2(:,t-1)); %double check whether p_yt_Yt should be transposed
	B_yt_R2(:,t) = B_yt_R2(:,t)/sum(B_yt_R2(:,t));

	TonicNE(t) = 1 - max(B_yt_R2(:,t));

end

correcttrials = feedback == choice;
nulltrials = difficulty == 4;
performance = mean(correcttrials)
nulldirections = mean(feedback(nulltrials))
nullchoices = mean(choice(nulltrials))
nullperformance = mean(correcttrials(nulltrials))



