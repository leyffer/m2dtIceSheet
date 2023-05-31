% Step 4: manually trim outline

% In this script we open one of the outlines from script 3 and manually
% remove artefacts and stronger bottlenecks. The reason for doing this is 
% that if you take a closer look at
% the outline, you'll find a lot of areas on the coast that fall more into
% the island category (bigger land masses) but have been connected to our
% main Greenland ice by a couple of nodes. I expect that having these
% bottleneck areas is going to be very hard on the conditioning of the
% forward computations. For this reason I've decided to take them out.

%% Step 1: create a folder, move data

resolution = 150;
% we make a separate folder for each resolution. Make a folder "outlines/step3-<resolution>m"

% Copy the file outlines/step2/domain_coarsened<resolution>.exp from the previous script
% into the folder "outlines/step3-<resolution>m" and rename it to 
% domain_<resolution>m_trimmed1.exe
% We keep a copy of it in the outlines/step2 folder for savekeeping.

%% Step 2: adjust iterator

% adjust iterator to adjustment version
j=2;
% When making adjustments in several steps, we make a copy for save
% keeping after each step. In your first run of this script, keep it at
% j=1;

filename = compose('outlines/step3-%im/domain_%im_trimmed%i.exp', resolution, resolution, j);
filename = filename{1};

%% Step 3: create model for background plot

% This can take a while. If you've created the model already, skip this
% part.

% For trimming the outline, I find it easiest to have a mesh in the
% background to help me judge how the outline geometry is affecting the
% mesh. 

tic;
md = triangle(model, 'outlines/step1/domain_step1.exp', 20000);
toc;

% I'm using the original outline domain_step1.exp from step 1 instead of
% domain_step1_modified.exp so that if I make several adjustments I can
% still judge how much I've taken out compared to the original.

% If you want to iterate over your modifications, comment in
%md = triangle(model, filename, 20000);

%% Step 4: backround plot

plotmodel(md, 'data', 'mesh');

% Alternatively, it might make sense to trim the outline based on ice
% thickness information. I'm not trying that out right now to not get
% sidetracked, but it might be an idea for the future.

%% Step 5: Use exptool

% We now use exptool to manually cut out larger areas. 
exptool(filename);
% You'll need to confirm in the command window again that you want to
% modify the file.

% First things first:
% the created outline might not be closed. To not run into issues, the
% first thing you should always do it close it:
% 1. click "close profile"
% 2. select the outline
% 3. press enter

% Next, use the hand in the matlab window to move your vision along the
% coastline. If you've found an area you want to cut off:
% 1. zoom in such that you have a good vision of where you want to make the
% cut
% 2. select "out larger area" in the exptool window
% 3. click on the two nodes that limit your cut
% 4. click on the area to be cut out (I always forget this...)
% 5. press enter
% 6. close the profile as described above

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

