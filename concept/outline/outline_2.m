% Step 2: identify Greenland

% in step 1 we have identified several outlines of which one will be the
% main Greenland ice sheet, and the others will be islands. In this script
% we open the files for the individual outlines manually to find out which
% one we are actually interested in.

%% look at individual outlines

% First, look into the folder outlines/step1 and find out the maximum
% number of outlines that have been chosen. For me right now, this is 9.
% Choose i as an index between 1 and this number.
i = 1;
% if you take a look at the size of the files, you can also make a good
% initial guess which one is going to be Greenland

% generate file name
filename = compose('outlines/step1/domain_%i.exp', i);
filename = filename{1};

% look at the outline
% we could generate a file for the outline. However, when doing this
% repeatedly, Matlab crashes for me. Presumably because the main Greenland
% outline is so large.

% md_coarse=triangle(model, filename, 5000);
% plotmodel(md_coarse, 'data', 'mesh')

% Important note:
% if the code lines 26-27 crash, it does not mean the outline file is
% corrupted. I assumed this incorrectly at some point and wasted a lot of
% time. If you open matlab again and run the same code, it will likely
% work. If you run it a couple times more with the same index, it will
% likely crash again.

% To avoid creating the mesh, we use exptools to take a look at the outline
expdisp(filename)

% 1. Press y. 
% The exptool window will appear. For me it appears on

% 2. Take a look at the outline. 

% 3. Quit in the exptool window without any changes.
% For me, the exptool window appears on a different monitor than Matlab 
% (to the far left), so it's a bit annoying to notice that it even opened.

% 4. Decide if you want to keep the file
% if you've found the main Greenland ice sheet, rename it to domain_step1
% you can delete all other .exp files in the folder.
