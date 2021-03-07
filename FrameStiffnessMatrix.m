clear
clc
close all

% syms A E L I u1 v1 phi1 F1 S1 Q1 u2 v2 phi2 F2 S2 Q2 P real
% 
% t = 0.125;
% p = 4;
% E = 11.4e6;
% L = 24;
% 
% A = p*t;
% I = t*p^3 / 12;
% 
% u1 = 0.001;
% v1 = L - sqrt(L^2 - u1^2);
% phi1 = 0;
% u2 = 0;
% v2 = 0;
% phi2 = 0;
% 
% a = A*E/L;
% b = E*I/L;
% c = 6*E*I/L^2;
% d = 12*E*I/L^3;
% 
% K = [
%     a   0   0  -a   0   0;
%     0   d   c   0  -d   c;
%     0   c 4*b   0  -c 2*b;
%    -a   0   0   a   0   0;
%     0  -d  -c   0   d  -c;
%     0   c 2*b   0  -c 4*b
%     ];
% 
% x = [u1 v1 phi1 u2 v2 phi2]';
% F = [F1 S1 Q1 F2 S2 Q2]';
% 
% F==K*x

% V = 3.06;
V = linspace(1,12,100);
L = 3.8e-3;
Imax = 1.7;
f = 60;
steps = 200;

spd = @(V,L,I,f,steps) V*f / (2*L*Imax*steps);
P = @(V,I) V*I;
tq = @(rpm,P) 60*P./(2*pi*rpm);

rpm = spd(V,L,Imax,f,steps);
Pmax = P(V,Imax);
tq_max = tq(rpm,Pmax);

plot(V,tq_max)
