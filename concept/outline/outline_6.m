% Step 6: manually trim outline

% Analoguously to step 3, we now go over the outline again and manually
% refine / coarsen it. 

%% Step 1: create a folder, move data, etc.

filename = 'outlines/step5/domain_150m_trimmed2_w5_c0.66.exp';

%% Step 3: create model for background plot

% For trimming the outline, I find it easiest to have a mesh in the
% background to help me judge how the outline geometry is affecting the
% mesh. 

tic;
md = triangle(model, 'outlines/step1/domain_step1.exp', 20000);
toc;
% Here I'm using the original outline domain_step1.exp from step 1 instead of
% domain_step1_modified.exp so that if I make several adjustments I can
% still judge how much I've taken out compared to the original.

%% Step 4: backround plot

% I decided here to use the trimmed outline as background plot to not get
% distracted by details I've already trimmed off in script 3.

%plotmodel(md, 'data', 'mesh');
plotmodel(md_trimmed, 'data', 'mesh')

% Alternatively, it might make sense to trim the outline based on ice
% thickness information. I'm not trying that out right now to not get
% sidetracked, but it might be an idea for the future.

%% Step 5: Use exptool

% We now use exptool to manually cut out larger areas. 
exptool(filename);
% You'll need to confirm in the command window again that you want to
% modify the file.

% The tools I found most useful here are "remove points" and "add points in
% existing profile" (or something like that, I don't have exptools open
% right now). There shouldn't be any more big islands to cut off. The
% things I decided to trim were very sharp edges. Also, sometimes it
% happened that the new outline included larger areas that were excluded in
% the original outline - in those cases I moved the new outline to exclude
% those areas again.

% When done, click "quit".

%% Further refinement:

% If you want to refine the outline further, make a copy of the current
% state first and name it "domain_step1_modified<j+1>". Remember to adjust
% the index j in the code above.

%% Happy?

% When you are happy with your adjustments, copy the latest version of your
% outline, and rename the copy to "domain_<resolution>m_trimmed.exe". 
% If you are confident in it, you can delete the remaining .exp files in 
% this folder, but I suggest to keep them for now as backup.

