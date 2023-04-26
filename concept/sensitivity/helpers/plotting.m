md = loadmodel('../Models/Greenland.Control_drag');
md_5 = loadmodel('../Models/Greenland.Control_drag_01');

%% mesh
plotmodel(md,'data','mesh');

%% height

plotmodel(md,'data',md.geometry.thickness, 'title', '              thickness data');

%% basal friction

plotmodel(md,'data',md.friction.coefficient, 'title', '              basal friction');

%% velocity

plotmodel(md,'data',max(md.inversion.vel_obs, 1), 'title', '              surface velocity data', 'log#1', 10);

%% compare thickness data

plotmodel(md, 'nlines', 1, 'ncols', 3, ...
    'data', md.geometry.thickness, 'title', 'thickness, reference data', ...
    'data', md_5.geometry.thickness, 'title', 'thickness, 0.1% error', ...
    'data', md.geometry.thickness - md_5.geometry.thickness, 'title', 'difference')

%% compare friction data

plotmodel(md, 'nlines', 1, 'ncols', 4, ...
    'data', md.friction.coefficient, 'title', 'basal friction (from reference data)', ...
    'data', md_5.friction.coefficient, 'title', 'basal friction (0.1% error data)', ...
    'data', md.friction.coefficient - md_5.friction.coefficient, 'title', 'difference', ...
    'data', max(100 * abs(md.friction.coefficient - md_5.friction.coefficient)./md.friction.coefficient, 1e-12), 'title', 'relative difference (%)', 'log#4', 10, 'caxis#4', [1e-2, 1e+2])


%%
rel_error = 100 * abs(md.friction.coefficient - md_5.friction.coefficient)./md.friction.coefficient;
min(rel_error)  % 0
max(rel_error)  % 2.9e+3
mean(rel_error) % 8.9

%%