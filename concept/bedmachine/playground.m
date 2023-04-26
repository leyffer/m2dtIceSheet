

ncdata='/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc'; % Nicole's data path
% useful commands:
% ncread(file, var) to read out data
% ncdisp(file) to get an overview of the contained content

%% loading data

x           = ncread(ncdata,'x');           % cartesian x-coordinate
y           = ncread(ncdata,'y');           % cartesian y-coordinate
surface     = ncread(ncdata, 'surface');    % ice surface elevation in meters (relative to geoid)
thickness   = ncread(ncdata, 'thickness');  % ice thickness in meters
bed         = ncread(ncdata, 'bed');        % bed topography (bedrock altitude) in meters (relative to geoid)

mask        = ncread(ncdata, 'mask');
% 0 = ocean
% 1 = ice-free land
% 2 = grounded ice
% 3 = floating ice
% 4 = non-Greenland land

errbed      = ncread(ncdata, 'errbed');
% bed topography / ice thickness error (in meters)
% this is probably what we need

source      = ncread(ncdata, 'source');
% 'data source (0 = none, 1 = gimpdem, 2 = Mass conservation, 
%               3 = synthetic, 4 = interpolation, 5 = hydrostatic equilibrium, 
%               6 = kriging, 7 = RTOPO-2, 8 = gravity inversion, 
%               9 = Millan et al. 2021, 10+ = bathymetry data)'

dataid      = ncread(ncdata, 'dataid');
% data ID:  1 = GIMPdem
%           2 = Radar
%           7 = seismic
%          10 = multibeam

geoid       = ncread(ncdata, 'geoid');
% geo-ID whatever that is???
% long_name     = 'EIGEN-6C4 Geoid - WGS84 Ellipsoid difference'
% standard_name = 'geoid_height_above_reference_ellipsoid'
% units         = 'meters'
% grid_mapping  = 'mapping'
% geoid         = 'eigen-6c4 (Forste et al 2014)'

%% tests:

% surface above bed ?
test = surface - bed;
if min(min(test)) < 0
    fprintf("TEST FAILED: Surface below bedrock")
end

%% plot: thickness

figure(1);
imagesc(thickness');
colorbar
title('thickness image')

%% plot: surface - bed

figure(2);
imagesc((surface-bed)')
colorbar
title('surface-bed')

%% plot: mask

figure(3);
imagesc(mask')

%% plot: errbed

figure(4);
imagesc(errbed')
colorbar
title('errbed')

%% get a mesh
% HOW?? 


