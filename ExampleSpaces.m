clear
clc
close all

s = 5;
FigSz = [s s];
FontSz = 14;

N = 100;

f_v = @(r) 4* pi * r.^3 / 3;
f_m = @(rho,r) rho .* f_v(r);

R_v = [10,20];
R_m = [30,75];

w_r = linspace(1,2,N);
w_rho = linspace(1,8,N);

[W_r,W_rho] = meshgrid(w_r,w_rho);

V = f_v(W_r);
M = f_m(W_rho,W_r);

idx_V = (V>min(R_v) & V<max(R_v));
idx_M = (M>min(R_m) & M<max(R_m));
pass = idx_V & idx_M;
fail = logical(abs(pass-1));

%%
%%%%%%%%%%%%%%% Solution Space Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize figure
f_sample = figure;
f_sample.Units = 'inches';
f_sample.Position = [7 3.5, FigSz];
f_sample.Color = 'w';
hold on

% Plotting commands
scatter(W_r(pass),W_rho(pass),'LineWidth',1.25)
scatter(W_r(fail),W_rho(fail),'LineWidth',1.25)

% Formatting
ax = gca;
ax.FontSize = FontSz;
ylabel("Density ($\rho$) [kg/m$^3$]",'FontSize',1.3*FontSz,'Interpreter','Latex')
xlabel("Radius ($r$) [m]",'FontSize',1.3*FontSz,'Interpreter','Latex')
legend({"pass","fail"},'FontSize',1.3*FontSz,'Interpreter','Latex')

% Save figure to file
saveas(gcf,'D:\Josh\GoogleRoot\School\Clemson\Thesis\Submissions\Journal_May2021\LaTex\images\ExSolnSpace.png')

%%
%%%%%%%%%%%%%%% Problem Space Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize figure
f_sample = figure;
f_sample.Units = 'inches';
f_sample.Position = [7 3.5, FigSz];
f_sample.Color = 'w';
hold on

% Plotting commands
ax = gca;
offset = 0.4;
Xoffset = offset*min(R_m);
Yoffset = offset*min(R_v);
ax.XLim = [min(R_m)-Xoffset max(R_m)+Xoffset];
ax.YLim = [min(R_v)-Yoffset max(R_v)+Yoffset];
area(R_m,[max(R_v) max(R_v)],min(R_v))
plot(ax.XLim, [min(R_v) min(R_v)] ,'r' , ax.XLim, [max(R_v) max(R_v)], 'r', 'LineWidth', 1.25)
plot([min(R_m) min(R_m)], ax.YLim ,'r' , [max(R_m) max(R_m)], ax.YLim, 'r', 'LineWidth', 1.25)
legend({'Problem Space','Constraints'})

% Formatting
ax = gca;
ax.FontSize = FontSz;
ylabel("Volume ($V$) [m$^3$]",'FontSize',1.3*FontSz,'Interpreter','Latex')
xlabel("Mass ($m$) [kg]",'FontSize',1.3*FontSz,'Interpreter','Latex')

% Save figure to file
saveas(gcf,'D:\Josh\GoogleRoot\School\Clemson\Thesis\Submissions\Journal_May2021\LaTex\images\ExProbSpace.png')