% Step 5: create mask and blur...
% ... and so on

% I could (should?) have divided this script into several more, but with
% the filenames tracking the settings and matlab being able to run sections
% at a time, it seemed ok to have more things at once happening here,
% especially since many of the steps here have already been covered
% previously.

%% load x and y coordinates from bedmachine 
% unless these are already in the workspace of course

ncdata='/workspace/ISSM-MATLAB/examples/1_kindergarten/IceMachine/BedMachineGreenland-v5.nc'; % Nicole's data path

x           = double(ncread(ncdata,'x'));    % cartesian x-coordinate
y           = double(ncread(ncdata,'y'));    % cartesian y-coordinate

%% choose outline file to work with

name_folder = 'outlines/step3-150m';
name_outline = 'domain_150m_trimmed2';

path_outline = join([name_folder, '/', name_outline, '.exp']);

%% create model for this outline

% we want to blur the mask associated to the trimmed outline. I haven't
% found a good way to get the mask, so here we are using a work around. For
% this trick we (unfortunately) need to create a grid for the trimmed
% outline first.

tic;
md_trimmed = triangle(model, path_outline, 20000);
toc;

%% create mask
% We now create a field that is plainly equal to 1 everywhere on the mesh.
% Then, we interpolate this field on the original coordinate system from
% BedMachine.

tic;
helper = ones(md_trimmed.mesh.numberofvertices, 1);
mask = InterpFromMeshToGrid(md_trimmed.mesh.elements, md_trimmed.mesh.x, md_trimmed.mesh.y, helper, x, y, 0);
toc;

path_save_mask = join(['outlines/step4/masks/', name_outline]);
save(path_save_mask, 'mask');

%% look at the mask if you want to 

imagesc(mask);

%% blur the mask
% This is the trickiest part. We want to blur the mask to smooth out the
% details of the domain boundary but not so much that we lose important
% features. I tried around a bit and decided to for a uniform filter with
% width w=5. However, there might be much better choices, in particular a
% Gaussian filter might be a bettic;
% md = triangle(model, 'outlines/step1/domain_step1.exp', 20000);
% toc;ter choice.

w = 5;
mask_blurred = conv2(mask, ones(w)/w^2, 'same');

%% save

yolo = compose('_w%i', w);
path_save_mask_blurred = join(['outlines/step4/blurred_masks/', name_outline, yolo{1}]);
save(path_save_mask_blurred, 'mask_blurred');

%% always nice to take a look at the mask

imagesc(mask_blurred)

%% create the contour
% this is another tricky part. We need to tell the function contourc which
% isoline we want. We can choose any value between 0 and 1 since these are
% the minimum and maximum on the blurred mask. If we choose 0 (or close to
% it) we are biasing the new outline towards outside of the original
% domain, if we choose a value close to 1 we are forcing the the new
% outline to be completely inward from the trimmed one. My issue with 0.5
% was that if the trimmed outline oscillerates the new one would just
% straight cut through it, i.e. alternate between areas that were
% originally outside or inside the domain. I found 0.66 to be a good
% compromise, but, again, there might be better choices.

val_c = 0.66;
% Note:There might be errors if choosing exactly 0 or 1, not sure.

c = contourc(x, y, double(mask_blurred), [val_c val_c]);

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

path_save_A = compose('outlines/step4/outlines/%s%s.exp', name_outline, yolo{1});
path_save_A = path_save_A{1};
expwrite(A, path_save_A);

%% save outlines individually

% in all my examples we only found a single outline above, so this part is
% commented out. If A actually contains several outlines, we need to save
% them individually and look through them again, just as we did in the
% script outline_5.

% domains = expread(path_save_A);
% 
% for i = 1:size(domains, 2)
%     filename = compose('outlines/step4/outlines/%s%s_%i.exp', name_outline, yolo{1} i);
%     filename = filename{1};
%     expwrite(domains(i), filename)
% end

%% take a look at the outline and how well it fits to the trimmed mesh

plotmodel(md_trimmed, 'data', 'mesh');
expdisp(path_save_A);
% now is a good choice to go back up and adjust the w and val_c parameters.

%% coarsen the outline
% we now coarsen the outline again to get rid of redundant nodes. We are
% choosing the resolution w*150 for the new outline resolution because one
% pixel corresponds to approximately 150m and we blurred w pixels together.

path_save_final = compose('outlines/step4/%s_w%i_c%f.exp', name_outline, w, val_c);
path_save_final = path_save_final{1};

tic;
expcoarsen(path_save_final, path_save_A, w*150);
toc;

%% take another look for your final judgement

plotmodel(md_trimmed, 'data', 'mesh');
expdisp(path_save_final);

