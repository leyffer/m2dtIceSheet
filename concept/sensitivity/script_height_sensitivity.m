tic;

mode = 2;
% 1: random perturbation everywhere (not implemented in this script yet)
% 2: one uniform parameter in multiplication
% 3: add one value everywhere (not implemented in this script yet)
% 4: given field (not implemented in this script yet)

% how many samples to take
n_para = 200;
% 200: approximately 217 min = 3.6 h

%% variations

switch mode
    case 2
        list_para = 0.05 * randn(1, n_para);
    otherwise
        fprintf("Mode not implemented")
end

friction = cell(n_para);
velocity = cell(n_para);
height = cell(n_para);

for i = 1:n_para
    fprintf("%i", i);
    friction{i} = my_inversion(list_para(i), mode);
    [vel, h] = my_transient(friction{i});
    velocity{i} = vel;
    height{i} = h;
end

%% ground truth

friction_0 = my_inversion(0, mode);
[vel_0, height_0] = my_transient(friction_0);

% question:
% friction_0, vel_0, height_0 are used below to as reference for computing
% the modes. Should they really be the solutions at the original height map
% or should these be replaced with the mean over the samples?

toc;

%% compute how much the friction changes for the different parameters

change_friction = [];
change_vel = [];
change_height = [];

for i = 1:n_para
    change_friction = [change_friction, norm(friction{i}-friction_0)];
    change_vel = [change_vel, norm(velocity{i}-vel_0)];
    change_height = [change_height, norm(height{i}-height_0)];
end

change_friction = 100 * change_friction / norm(friction_0);
change_vel = 100 * change_vel / norm(vel_0);
change_height = 100 * change_height / norm(height_0);

[ordered_para, order] = sort(list_para);
change_friction = change_friction(order);
change_vel = change_vel(order);
change_height = change_height(order);

%% plot friction change

figure(1);

plot(100*ordered_para, change_friction)
xlabel("ice thickness error [%]")
ylabel("relative error")
title("friction change [%] vs ice thickness error")
legend("||a-a0||/||a0||")


%% plot velocity change

figure(2);

plot(100*ordered_para, change_vel)
xlabel("ice thickness error [%]")
ylabel("relative error")
title("velocity change [%] vs ice thickness error")
legend("||v-v0||/||v0||")


%% plot height change

figure(3);

plot(100*ordered_para, change_height)
xlabel("initial ice thickness error [%]")
ylabel("relative error")
title("ice thickness change [%, after 10 years] vs ice thickness error")
legend("||h-h0||/||h0||")

%% find dominating modes

snapshots_friction = [];
snapshots_vel = [];
snapshots_height = [];

for i = 1:n_para
    snapshots_friction = [snapshots_friction friction{i}-friction_0];
    snapshots_vel = [snapshots_vel, velocity{i}-vel_0];
    snapshots_height = [snapshots_height, height{i}-height_0];
end
% question:
% should we substract friction_0 (the friction at the mean) or the mean of
% the samples?

gramian = snapshots_friction' * snapshots_friction;

[V_friction, D_friction] = eig(gramian);
D_friction = diag(D_friction);
D_friction = D_friction(end:-1:1); % largest eigenvalue on the left
V_friction = V_friction(:, end:-1:1);

[V_vel, D_vel] = eig(gramian);
D_vel = diag(D_vel);
D_vel = D_vel(end:-1:1); % largest eigenvalue on the left
V_vel = V_vel(:, end:-1:1);

[V_height, D_height] = eig(gramian);
D_height = diag(D_height);
D_height = D_height(end:-1:1); % largest eigenvalue on the left
V_height = V_height(:, end:-1:1);

%% eigenvalue decay: friction

figure(4);

semilogy([1:n_para], D_friction)

%title('gramian eigenvalue decay')
xlim([1, n_para])
xlabel("eigenvalue number")
ylabel("eigenvalue")

%% energy decay: friction

rel = sum(D_friction);
energy = zeros(n_para, 1);
for i = 1:n_para
    energy(i) = sum(D_friction(1:i)) / rel;
end

figure(5);
plot([1:n_para], energy)
xlim([1, n_para])
xlabel("number of modes")
ylabel("captured energy")
title('captured energy vs number of included modes (friction)')


%% plot modes: friction

md = loadmodel('./Models/Greenland.Parameterization');
modes = snapshots_friction * V_friction;

plotmodel(md, 'ncols', 3, 'nrows', 2, ...
    'data', modes(:, 1), 'title', 'friction, mode 1', ...
    'data', modes(:, 2), 'title', 'friction, mode 2', ...
    'data', modes(:, 3), 'title', 'friction, mode 3', ...
    'data', modes(:, 4), 'title', 'friction, mode 4', ...
    'data', modes(:, 5), 'title', 'friction, mode 5', ...
    'data', modes(:, 6), 'title', 'friction, mode 6', ...
    'caxis#all', [min(min(modes)), max(max(modes))])

%% energy decay: velocity

rel = sum(D_vel);
energy = zeros(n_para, 1);
for i = 1:n_para
    energy(i) = sum(D_vel(1:i)) / rel;
end

figure(5);
plot([1:n_para], energy)
xlim([1, n_para])
xlabel("number of modes")
ylabel("captured energy")
title("captured energy vs number of included modes (velocity)")

%% energy decay: thickness

rel = sum(D_height);
energy = zeros(n_para, 1);
for i = 1:n_para
    energy(i) = sum(D_height(1:i)) / rel;
end

figure(5);
plot([1:n_para], energy)
xlim([1, n_para])
xlabel("number of modes")
ylabel("captured energy")
title("captured energy vs number of included modes (thickness)")

%% active subspaces

% todo:
% look up how to construct active subspaces again. I think those are
% probably the way to go to find out which regions in the height map  
% (or at least in the friction field) are the most influential for the
% transient solution.

% todo:
% we should also decide if we are interested in the transient solution or
% just in the friction field.

