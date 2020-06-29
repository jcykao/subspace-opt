%%demo file for running optimization functions
%%authors: Xiyuan Jiang, Hemant Saggar
%%date: 27th June 2020

%%%%%%%%%%%%%%%%%%%%    COPYRIGHT AND LICENSE NOTICE    %%%%%%%%%%%%%%%%%%%
%    Copyright Xiyuan Jiang, Hemant Saggar, Jonathan Kao 2020
%    
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%mandatory import of manopt functions
run('manopt/importmanopt');

%load the covariance matrices of data from both conditions. Please use your
%data here. cov_marices.mat must contain two nxn matrices where n is the number
%of neurons observed. XcovObs- covariance matrix under Observation
%condition and XcovEx- covariance matrix under Execution or hand motion
%condition.
load('cov_matrices.mat');

currpath = pwd;
%this is where the optimization functions reside
addpath([currpath '/optFunctions']);
%% Optimization 1, Orth Subspace
d_Ex = 4;
d_Obs = 4;

[Q, ~, info, options] = orthogonal_subspaces(XcovEx,d_Ex,XcovObs,d_Obs);
P1 = [eye(d_Ex); zeros(d_Obs,d_Ex)];
P2 = [zeros(d_Ex, d_Obs); eye(d_Obs)];
dmax = max(d_Ex,d_Obs);
eigvals1 = eigs(XcovEx, dmax, 'la'); 
eigvals2 = eigs(XcovObs, dmax, 'la');

% variance explained for each condition in each subspace
Ex_on_Ex = var_proj(Q*P1,XcovEx,sum(eigvals1(1:d_Ex))); % var explained of Ex in Orth-Ex subsapce
Obs_on_Ex = var_proj(Q*P1,XcovObs,sum(eigvals2(1:d_Ex))); % var explained of Obs in Orth-Ex subsapce
Obs_on_Obs = var_proj(Q*P2,XcovObs,sum(eigvals2(1:d_Obs)));
Ex_on_Obs = var_proj(Q*P2,XcovEx,sum(eigvals1(1:d_Obs)));

figure();
bar([Ex_on_Ex, Obs_on_Ex, Obs_on_Obs, Ex_on_Obs]);
grid on;
ax= gca();
ax.XTickLabel = {'Ex on Ex', 'Obs on Ex', 'Obs on Obs', 'Ex on Obs'};
xlabel('Subspace projections');
ylabel('Fraction of variance captured');
title('Variance captured for orthogonal subspace hypothesis');
%% Optimization 2, Exclusive Subspace
d_Ex = 4;
alphaNullSpace = 0.01;
[QEx,flagEx] = exclusive_subspace(XcovEx,XcovObs,d_Ex,alphaNullSpace); 
 % if flagEx = 1 then it means no subspace meets the constraint
eigvals1 = eigs(XcovEx, d_Ex, 'la');
res2_Ex_single = var_proj(QEx,XcovEx,sum(eigvals1(1:d_Ex)));
%constraining var of Obs, find the maximum var proj of Ex

d_Obs = 4;
alphaNullSpace = 0.01;
[QObs,flagObs] = exclusive_subspace(XcovObs,XcovEx,d_Obs,alphaNullSpace);
eigvals2 = eigs(XcovObs, d_Obs, 'la');
res2_Obs_single = var_proj(QObs,XcovObs,sum(eigvals2(1:d_Obs)));

figure();
bar([res2_Ex_single, res2_Obs_single]);
grid on;
ax= gca();
ax.XTickLabel = {'Ex exclusive','Obs exclusive'};
xlabel('Exclusive subspace');
ylabel('Fraction of variance captured');
title('Variance captured in exclusive subspace');
%% Optimization 3, Shared Subspace
d_Ex_for_shared = 4;
d_Obs_for_shared = 4;
d_shared = 4;

[QEx,flagEx] = exclusive_subspace(XcovEx,XcovObs,d_Ex_for_shared,alphaNullSpace); 

[QObs,flagObs] = exclusive_subspace(XcovObs,XcovEx,d_Obs_for_shared,alphaNullSpace);

[Q1,Qshared, Qcost, info, options] = shared_subspace(QEx,d_Ex_for_shared,QObs,d_Obs_for_shared,XcovEx,XcovObs,d_shared);

dmax = max(max(d_Ex_for_shared,d_Obs_for_shared),d_shared);
eigvals1 = eigs(XcovEx, dmax, 'la');
eigvals2 = eigs(XcovObs, dmax, 'la');
eigvals2_total = eig(XcovObs);
% variance explained for each condition in the shared subspace
res3_Ex = var_proj(Qshared,XcovEx,sum(eigvals1(1:d_shared)));
res3_Obs = var_proj(Qshared,XcovObs,sum(eigvals2(1:d_shared)));

figure();
bar([res3_Ex, res3_Obs]);
grid on;
ax = gca();
ax.XTickLabel= {'Ex in shared', 'Obs in shared'};
xlabel('Task/Condition');
ylabel('Fraction of variance captured');
title('Variance captured in the shared subspace');