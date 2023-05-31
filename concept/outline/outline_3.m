%% INPUT: resolution

% The bedmachine thickness data has a resolution of 150m (on the mesh, true
% resolution is between 150m and 5km). However, other datasets for other
% quantity, e.g., surface velocity, may have a finer resolution. So
% I don't think there's a unique choice. I decided to make a couple to
% start with.

resolution = 5000;
%  90 m: 528 s = 8.8 min - GIMP surface altitude resolution (also has 30m)
% 150 m: 449 s = 7.5 min - BedMachine resolution
% 200 m: 322 s = 5.4 min - first try
% 250 m: 286 s = 4.8 min - MEaSUREs Multi-year velocity resolution

%% file paths

% which outline to coarsen
filename = 'outlines/step1/domain_step1.exp';

% how to save the new outline
filename_new = compose('outlines/step2/domain_coarsened%i.exp', resolution);
filename_new = filename_new{1};

%% coarsen outline

tic;
expcoarsen(filename_new, filename, resolution);
toc;

%% take a look

expdisp(filename_new)

% Step 3: coarsening
% The Greenland outline obtained with steps 1-2 likely has a lot of hanging
% nodes, as well as a lot of nodes that are increadibly close to each
% other. The ISSM mesh generation should be able to kick those out, but
% from my tests it hardly ever does that. Instead, the fine outline causes
% the created mesh to have an increadibly fine resolution near the outline.
% In this script, we use the 'expcoarsen' function to coarsen the outline
% to a given resolution. 