% 

%% data path

ncdata='/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc'; % Nicole's data path
% useful commands:
% ncread(file, var) to read out data
% ncdisp(file) to get an overview of the contained content
% e.g., ncdisp('/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc')

%% loading data

x           = double(ncread(ncdata,'x'));    % cartesian x-coordinate
y           = double(ncread(ncdata,'y'));    % cartesian y-coordinate
thickness   = ncread(ncdata, 'thickness')';  % ice thickness in meters

%% outline data from ISSM

md = triangle(model, '../sensitivity/DomainOutline.exp', 20000);
plotmodel(md, 'data', 'mesh')

%% coordinate transform to different ellipsoid

[md.mesh.lat,md.mesh.long]=xy2ll(md.mesh.x,md.mesh.y,+1,39,71);
[x_test, y_test] = CoordTransform(md.mesh.lat, md.mesh.long, 'EPSG:4326', 'EPSG:3413');

%% interpolate on mesh

h = InterpFromGridToMesh(x,flipud(y),flipud(thickness),x_test,y_test,10000);

%% plot

plotmodel(md, 'data', h)