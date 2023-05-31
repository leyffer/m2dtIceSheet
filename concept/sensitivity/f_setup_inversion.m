function [md] = f_setup_inversion(md)

    %Control general
    md.inversion.iscontrol=1;
    md.inversion.nsteps=30;
    md.inversion.step_threshold=0.99*ones(md.inversion.nsteps,1);
    md.inversion.maxiter_per_step=5*ones(md.inversion.nsteps,1);
    
    %Cost functions
    md.inversion.cost_functions=[101 103 501];
    md.inversion.cost_functions_coefficients=ones(md.mesh.numberofvertices,3);
    md.inversion.cost_functions_coefficients(:,1)= 300; %350;
    md.inversion.cost_functions_coefficients(:,2)= 5e-3; %0.6;
    md.inversion.cost_functions_coefficients(:,3)= 1e-4; %2e-6;
    % regularization values suggested ...
    % ... on the ISSM website: 350, 0.6, 2e-6
    % ... in [Seroussi et al, 2013]: 300, 1.5, 1e-7;


    %Controls
    md.inversion.control_parameters={'FrictionCoefficient'};
    md.inversion.gradient_scaling(1:md.inversion.nsteps)=50;
    md.inversion.min_parameters=1*ones(md.mesh.numberofvertices,1);
    md.inversion.max_parameters=200*ones(md.mesh.numberofvertices,1);
    
    %Additional parameters
    md.stressbalance.restol=0.01; md.stressbalance.reltol=0.1;
    md.stressbalance.abstol=NaN;
    md.toolkits=toolkits;

end