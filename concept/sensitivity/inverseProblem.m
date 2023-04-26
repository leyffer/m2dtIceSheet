
%Location of SeaRISE dataset
%ncdata='../Data/Greenland_5km_dev1.2.nc';  % original
ncdata='/workspace/ISSM-MATLAB/examples/Data/Greenland_5km_dev1.2.nc'; % Nicole's data path

%% discretization

%Generate initial uniform mesh (resolution = 20000 m)
md=triangle(model,'./DomainOutline.exp',20000);

% Get velocities (Note: You can use ncdisp('file') to see an ncdump)
x1   = ncread(ncdata,'x1');
y1   = ncread(ncdata,'y1');
velx = ncread(ncdata,'surfvelx');
vely = ncread(ncdata,'surfvely');
vx   = InterpFromGridToMesh(x1,y1,velx',md.mesh.x,md.mesh.y,0);
vy   = InterpFromGridToMesh(x1,y1,vely',md.mesh.x,md.mesh.y,0);
vel  = sqrt(vx.^2+vy.^2);

%Mesh Greenland
md=bamg(md,'hmax',400000,'hmin',5000,'gradation',1.7,'field',vel,'err',8);

%convert x,y coordinates (Polar stereo) to lat/lon
[md.mesh.lat,md.mesh.long]=xy2ll(md.mesh.x,md.mesh.y,+1,39,71);

save ./Models/Greenland.Mesh_generation md;

plotmodel (md,'data','mesh');

%% parameterization

md = loadmodel('./Models/Greenland.Mesh_generation');

md = setmask(md,'','');
md = parameterize(md,'./Greenland.par');
md = setflowequation(md,'SSA','all');

plotmodel(md, 'data', md.geometry.thickness)
%plotmodel(md, 'data', md.initialization.vel, 'caxis', [1e-1 1e4], 'log', 10)

save ./Models/Greenland.Parameterization md;

%% height adjustment

md = loadmodel('./Models/Greenland.Parameterization');

mult = 0.01 * randn(size(md.geometry.thickness, 1), 1);
thickness = md.geometry.thickness;
thickness_adjustment = thickness.*mult;
md.geometry.thickness = max(md.geometry.thickness + thickness_adjustment, 0);
md.geometry.base = md.geometry.surface-md.geometry.thickness;

%% inverse problem setup

%Control general
md.inversion.iscontrol=1;
md.inversion.nsteps=30;
md.inversion.step_threshold=0.99*ones(md.inversion.nsteps,1);
md.inversion.maxiter_per_step=5*ones(md.inversion.nsteps,1);

%Cost functions
md.inversion.cost_functions=[101 103 501];
md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,3);
md.inversion.cost_functions_coefficients(:,1)=350;
md.inversion.cost_functions_coefficients(:,2)=0.6;
md.inversion.cost_functions_coefficients(:,3)=2e-6;

%Controls
md.inversion.control_parameters={'FrictionCoefficient'};
md.inversion.gradient_scaling(1:md.inversion.nsteps)=50;
md.inversion.min_parameters=1*ones(md.mesh.numberofvertices,1);
md.inversion.max_parameters=200*ones(md.mesh.numberofvertices,1);

%Additional parameters
md.stressbalance.restol=0.01; md.stressbalance.reltol=0.1;
md.stressbalance.abstol=NaN;
md.toolkits=toolkits;

%Go solve
md.cluster=generic('name',oshostname,'np',2);
md.verbose=verbose('solution',true,'control',true);
md=solve(md,'Stressbalance');

%Update model friction fields accordingly
md.friction.coefficient=md.results.StressbalanceSolution.FrictionCoefficient;

save ./Models/Greenland.Control_drag_01 md;


