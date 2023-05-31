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

%% load mesh

slicer = 20;
filename = compose('outlines/domain_slice%i.exp', slicer);
filename = filename{1};
md=triangle(model, filename, 5000);
plotmodel(md, 'data', 'mesh')

%% truth model

md_true = md;
md_true.geometry.surface = InterpFromGridToMesh(x,flipud(y),flipud(surface),md_true.mesh.x,md_true.mesh.y,0);
md_true.geometry.base = InterpFromGridToMesh(x,flipud(y),flipud(bed),md_true.mesh.x,md_true.mesh.y,0);
md_true.geometry.thickness = md_true.geometry.surface - md_true.geometry.base;

plotmodel(md_true, 'ncols', 2, ...
    'data', md_true.geometry.base, 'title', 'basal topography', ...
    'data', md_true.geometry.surface, 'title', 'surface topography')

%% prior knowledge

md_prior = md;
md_prior.geometry.surface = InterpFromGridToMesh(x,flipud(y),flipud(surface),md_true.mesh.x,md_true.mesh.y,0);
% surface altititude data can come from satellite data I'm pretty sure. So
% we should be ok assuming that the surface data is known.

% TODO: look for a dataset that gives the surface altitude data just
% constructed from satellite measurements

% TODO: find out if the airplanes measure ice depth or position of the
% bedrock. This will be important for our simulated measurements if we use
% surface elevation data from somewhere else. I believe it's ice depth, so
% I'll treat it as such below.

md_prior.geometry.base = zeros(size(md_prior.geometry.surface));
% I don't think this is a good prior. It says that at the edge of the
% domain the bedrock is at height 0, but we actually know the height there
% because the thickness gets close to zero and at almost all parts there is
% non-ice land around whose elevation we can measure with the satellites.

md_prior.geometry.thickness = md_prior.geometry.surface - md_prior.geometry.base;

%% uncertainty

% compute the distance between any point and the closest boundary node
boundary_vertices = [md.mesh.x(logical(md.mesh.vertexonboundary)), md.mesh.y(logical(md.mesh.vertexonboundary))];
dist_to_boundary = zeros(md.mesh.numberofvertices, 1);

for i=1:md.mesh.numberofvertices
    vertex = [md.mesh.x(i), md.mesh.y(i)];

    diff = boundary_vertices - vertex;
    dist = vecnorm(diff');
    dist_to_boundary(i) = min(dist);
end

% since we know the base altitude at the boundary (even if right now that's 
% not yet reflected in the prior), it makes sense to set uncertainty to
% zero at the boundary and increase it from there the further we get away
% from the boundary


%% plot the thickness

plotmodel(md, 'ncols', 3, ...
    'data', md_true.geometry.thickness, 'title', 'truth model thickness', ...
    'data', md_prior.geometry.thickness, 'title', 'prior thickness', ...
    'data', dist_to_boundary, 'title', 'assumed uncertainty')

%% prior covariance

% this is tricky.
% intuitively I'd put small uncertainty near the edge of the domain

%% take a flight

% to keep things simple right now, let's assume an airplane can only fly in
% a straight line. We can refine this later.

% x-values for this ellipsoid lie between -652925 and 879625
x_start = -200000;
x_stop = +200000;

% y-values for this ellipsoide lie between -3384425 and -632675
y_start = -2000000;
y_stop = y_start;%-2500000;

% how wide is the area observed by the airplane?
width = 5000; %70;
% I remember from somewhere that the area on the ground measured by the
% airplane is about 70m wide. That's on-ground distance and our mesh is
% distorted from flattening everything, but since 70m (or whichever value 
% it actually is) is relatively small compared to the rest of Greenland it
% should be ok to work with the same width everywhere.

% TODO: find out measurement width for airplanes

% however, our grid currently has only a resolution of 5000m and we need to
% account for that. So I'm setting the width wider for now

% TODO: think about how to balance airplane observation width and mesh
% accuracy. Probably this is the place where we need to incorporate mesh
% refinement

% describe trajectory on the 2D-plane
traj = [x_stop-x_start, y_stop-y_start];
traj = traj / norm(traj);
normal = [traj(2), -traj(1)];
M_dist = [traj; -normal]'; % matrix for computing the distance

% find all coordinates that could possibly be close to the flight path
loc_x_sub_1 = find(md.mesh.x > min(x_start, x_stop) - width/2);
loc_x_sub_2 = find(md.mesh.x < max(x_start, x_stop) + width/2);
loc_x_sub = intersect(loc_x_sub_1, loc_x_sub_2);

loc_y_sub_1 = find(md.mesh.y > min(y_start, y_stop) - width/2);
loc_y_sub_2 = find(md.mesh.y < max(y_start, y_stop) + width/2);
loc_y_sub = intersect(loc_y_sub_1, loc_y_sub_2);

loc_sub = intersect(loc_x_sub, loc_y_sub);

% restrict loc_sub to only those index pairs that are actually close to the
% line
j = 1;
loc_final = [];

for i=1:size(loc_sub, 1)
    % location of the vertex on the place
    px = md.mesh.x(loc_sub(i));
    py = md.mesh.y(loc_sub(i));
    rhs = [px - x_start; py - y_start];

    % find distance of point to line
    yolo = M_dist \ rhs;
    dist = yolo(2);

    if (abs(dist) < width / 2)
        loc_final(j) = loc_sub(i);
        % loc_final contains the indeces of all chosen vertices
        j = j + 1;
    end
end

% note:
% there's probably a more efficient way to implement this search. One thing
% that's good about this way at least is that by doing the box constraints
% first we know that we are not going to extend beyond the starting /
% ending points

%% take a flight

% to keep things simple right now, let's assume an airplane can only fly in
% a straight line. We can refine this later.

% x-values for this ellipsoid lie between -652925 and 879625
yolo = x_start;
x_start = x_stop;
x_stop = x_start;%yolo;%x_stop;

% y-values for this ellipsoide lie between -3384425 and -632675
yolo = y_start;
y_start = y_stop;
y_stop = -2500000;%yolo;

% how wide is the area observed by the airplane?
width = 5000; %70;
% I remember from somewhere that the area on the ground measured by the
% airplane is about 70m wide. That's on-ground distance and our mesh is
% distorted from flattening everything, but since 70m (or whichever value 
% it actually is) is relatively small compared to the rest of Greenland it
% should be ok to work with the same width everywhere.

% TODO: find out measurement width for airplanes

% however, our grid currently has only a resolution of 5000m and we need to
% account for that. So I'm setting the width wider for now

% TODO: think about how to balance airplane observation width and mesh
% accuracy. Probably this is the place where we need to incorporate mesh
% refinement

% describe trajectory on the 2D-plane
traj = [x_stop-x_start, y_stop-y_start];
traj = traj / norm(traj);
normal = [traj(2), -traj(1)];
M_dist = [traj; -normal]'; % matrix for computing the distance

% find all coordinates that could possibly be close to the flight path
loc_x_sub_1 = find(md.mesh.x > min(x_start, x_stop) - width/2);
loc_x_sub_2 = find(md.mesh.x < max(x_start, x_stop) + width/2);
loc_x_sub = intersect(loc_x_sub_1, loc_x_sub_2);

loc_y_sub_1 = find(md.mesh.y > min(y_start, y_stop) - width/2);
loc_y_sub_2 = find(md.mesh.y < max(y_start, y_stop) + width/2);
loc_y_sub = intersect(loc_y_sub_1, loc_y_sub_2);

loc_sub = intersect(loc_x_sub, loc_y_sub);

% restrict loc_sub to only those index pairs that are actually close to the
% line

for i=1:size(loc_sub, 1)
    % location of the vertex on the place
    px = md.mesh.x(loc_sub(i));
    py = md.mesh.y(loc_sub(i));
    rhs = [px - x_start; py - y_start];

    % find distance of point to line
    yolo = M_dist \ rhs;
    dist = yolo(2);

    if (abs(dist) < width / 2)
        loc_final(j) = loc_sub(i);
        % loc_final contains the indeces of all chosen vertices
        j = j + 1;
    end
end

% note:
% there's probably a more efficient way to implement this search. One thing
% that's good about this way at least is that by doing the box constraints
% first we know that we are not going to extend beyond the starting /
% ending points


%% take a look if it looks correct

md_test = md_prior;
md_test.geometry.thickness(loc_final) = max(md_true.geometry.thickness); 
%md_test.geometry.thickness(loc_final) =md_true.geometry.thickness(loc_final);
% note: I'm not imposing the value of md_true here for better visibility

plotmodel(md, 'ncols', 2, ...
    'data', md_true.geometry.thickness, 'title', 'truth model thickness', ...
    'data', md_test.geometry.thickness, 'title', 'prior thickness with update')

%% update to posterior

% compute the distance between any point and the closest boundary node
boundary_vertices_extended = [boundary_vertices; [md.mesh.x(loc_final), md.mesh.y(loc_final)]];

dist_to_boundary_extended = zeros(md.mesh.numberofvertices, 1);

for i=1:md.mesh.numberofvertices
    vertex = [md.mesh.x(i), md.mesh.y(i)];

    diff = boundary_vertices_extended - vertex;
    dist = vecnorm(diff');
    dist_to_boundary_extended(i) = min(dist);
end

%% 

plotmodel(md, 'ncols', 3, ...
    'data', md_true.geometry.thickness, 'title', 'truth model thickness', ...
    'data', md_test.geometry.thickness, 'title', 'prior thickness with update', ...
    'data', dist_to_boundary_extended, 'title', 'updated uncertainty')


