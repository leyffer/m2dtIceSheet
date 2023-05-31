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

%% restrict the mask for coarser resolution

mask2 = mask;

% only look at the grounded ice
non_ice = find(mask2 ~= 2);
mask2(non_ice) = 0;

% exclude all areas where ice height is below 50 cm
min_h = find(thickness < 0.5);
mask2(min_h) = 0;

% coarsen the resolution
slicer = 20;
mask2 = mask2(1:slicer:end, 1:slicer:end);
x2 = x(1:slicer:end);
y2 = y(1:slicer:end);

% construct contour
c = contourc(x2, y2, double(mask2), [.5 .5]);

%% create an outline file
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

expwrite(A, 'outlines/domains_all.exp');

%% save each outline individually

domains = expread('outlines/domains_all.exp');

for i = 1:size(domains, 2)
    filename = compose('outlines/domain_%i.exp', i);
    filename = filename{1};
    expwrite(domains(i), filename)
end


%% plot individual outline

i = 1;
filename = compose('outlines/domain_%i.exp', i);
filename = filename{1};
md=triangle(model, filename, 5000);
plotmodel(md, 'data', 'mesh')

%% mesh refinement?
% typically, we would now call bamg to refine and coarsen the mesh in a way
% that some input field can be approximated with sufficient accuracy 
% (typically the measured surface velocity field I think, though I should 
% double check that - after putting so much effort into getting the height
% data, it seems weird that the grid does not account for the height).
% In our case I would probably refine to approximate the thickness data,
% but if the goal is to measure the thickness than that's not known a
% priori. Keeping the grid uniform though leads to large DoF. The mesh for
% slicer = 25 with triangle resolution 5000m has 156980 DoF already. I
% would test how long it takes but for that I need to decide on the
% parameterization and that seems like a sidetrack to get stuck on. For the
% moment I'll hence keep the mesh ~uniform (as generated with the triangle)
% method, and I'll come back to adaptive mesh refinement later.

% TODO: decide on parameterization (or carry over from other examples)
% TODO: test how long a stressbalance solve and an inverse solve take with
% the current mesh
% TODO: think about which fields we should use as basis for grid
% refinement, e.g. surface height data
% TODO: think about if it would make sense to refine the grid adaptively
% during the OED process.


% That's not too bad to work with, but also not great since we need the
% mesh in the inversion.



