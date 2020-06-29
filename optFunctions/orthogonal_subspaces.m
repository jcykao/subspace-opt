function [Q, Qcost, info, options] = orthogonal_subspaces(C1, d1, C2, d2, options)
%manifold optimization to two orthogonal subspaces that maximize the sum of variance captured.
%This file is called by other files.
%C1 = covariance matrix 1 for PCA, 
%d1 = number of dimensions of C1 to sum in trace.
%C2 = covariance matrix 2 for PCA,
%d2 = number of dimensions of C2 to sum in trace.

% Written by Xiyuan Jiang and Hemant Saggar
% Date: 06/27/2020

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


assert(isequal(C1, C1'));
assert(isequal(C2, C2'));
assert(size(C1,1) == size(C2,1));
n = size(C1,1);
dmax = max(d1,d2);
%largest magnitude eigenvalues
eigvals1 = eigs(C1, dmax, 'la'); %hoping that these all eigs are positive, still only divinding by largest algebraic +ve values
eigvals2 = eigs(C2, dmax, 'la');
assert(~any(eigvals1<0), 'eigvals1 <0');
assert(~any(eigvals2<0), 'eigvals2 <0');
P1 = [eye(d1); zeros(d2,d1)];
P2 = [zeros(d1, d2); eye(d2)];

% Create the problem structure.
    manifold = stiefelfactory(n,d1+ d2);
    problem.M = manifold;
% Define the problem cost function and its Euclidean gradient.
    problem.cost  = @(Q) -0.5*trace((Q*P1)'*C1*(Q*P1))/sum(eigvals1(1:d1)) - 0.5*trace((Q*P2)'*C2*(Q*P2))/sum(eigvals2(1:d2));
    problem.egrad = @(Q) -C1*Q*(P1*P1')/sum(eigvals1(1:d1)) - C2*Q*(P2*P2')/sum(eigvals2(1:d2));

% Numerically check gradient consistency (optional).
%checkgradient(problem);
% checkgradient(problem);
options.verbosity = 0;
% Solve.
[Q, Qcost, info, options] = trustregions(problem,[],options);
 
% Display some statistics.
% figure;
% semilogy([info.iter], [info.gradnorm], '.-');
% xlabel('Iteration number');
% ylabel('Norm of the gradient of f');

end