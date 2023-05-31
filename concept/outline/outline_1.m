% Step 1: Create fine resolution outline

% In this script we create an outline file for the bedmachine mask of
% Greenland. The approximate resolution of the original mask is 150m, so
% the outline will be very fine (too fine to work with on my workstation),
% so we will coarsen it in the upcoming steps.

%% Set data path

ncdata='/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc'; % Nicole's data path
% useful commands:
% ncread(file, var) to read out data
% ncdisp(file) to get an overview of the contained content
% e.g., ncdisp('/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc')

%% loading data

x           = double(ncread(ncdata,'x'));    % cartesian x-coordinate
y           = double(ncread(ncdata,'y'));    % cartesian y-coordinate
surface     = ncread(ncdata, 'surface')';    % ice surface elevation in meters (relative to geoid)
thickness   = ncread(ncdata, 'thickness')';  % ice thickness in meters
bed         = ncread(ncdata, 'bed')';        % bed topography (bedrock altitude) in meters (relative to geoid)

mask        = ncread(ncdata, 'mask')';
% 0 = ocean
% 1 = ice-free land
% 2 = grounded ice
% 3 = floating ice
% 4 = non-Greenland land

%% retrict to grounded ice only

mask2 = mask;

% only look at the grounded ice
mask2(mask2 ~= 2) = 0;

%% restrict based on ice height

% exclude all areas where ice height is below threshold
mask2(thickness < 1) = 0;
% minimum ice thickness gets typically set to 1m
% there's a good chance that the outline we have after coarsening will 
% include areas that this initial fine outline doesn't have, so I'm
% restricting the map here to an interior area.

%% coarsen the resolution? blur the mask?

% I played around with coarsening and blurring for a while, but it
% introduced quite a bit of bias at this stage. Specifically: There are
% many ice "islands" around the main Greenland ice sheet that are not
% connected to it on the original mask. However, after coarsening /
% blurring, their area can be included in the outline file.

% blurring
w = 5;
mask_2 = conv2(mask, ones(w)/w^2, 'same');

% coarsening:
% slicer = 1;
% mask2 = mask2(1:slicer:end, 1:slicer:end);
% x = x(1:slicer:end);
% y = y(1:slicer:end);

%% get contours

c = contourc(x, y, double(mask2), [1.9 1.9]);
% the mask has value 2 at grounded ice and 0 everywhere else.

%% save this mask

save('outlines/step1-blurred/step1_data.mat', 'mask2', 'x', 'y');
% if blurring, coarsening is used, it makes sense to also save the
% thickness data, slicer, etc.

%% create a file with all outlines
% using code by Mathieu Morlinhem at
% https://issm.ess.uci.edu/forum/d/284-create-mesh-with-bamg-without-augus-file

cutoff = 5000;
% the larger this number, the more outlines (e.g. from islands) get excluded

i=1; j=1;
while (i<length(c))
    num=c(2,i); i=i+1;
    s(j).x=c(1,i:(i+num-1)); s(j).y=c(2,i:(i+num-1)); s(j).v=c(1,i);
    i=i+num; j=j+1;
end

j=j-1;
A=struct();counter=1;

for i=1:j
    if (length(s(i).x)>cutoff)
       A(counter).x=s(i).x; A(counter).y=s(i).y;
       counter=counter+1;
    end
end

expwrite(A, 'outlines/step1-blurred/domains_all.exp');

%% create individual files for each outline

domains = expread('outlines/step1-blurred/domains_all.exp');

for i = 1:size(domains, 2)
    filename = compose('outlines/step1-blurred/domain_%i.exp', i);
    filename = filename{1};
    expwrite(domains(i), filename)
end

