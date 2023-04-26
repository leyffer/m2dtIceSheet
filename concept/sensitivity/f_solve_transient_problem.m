function [vel, height] = f_solve_transient_prblem(friction)

    % load parameterized model
    md = loadmodel('./Models/Greenland.Parameterization');

    %Update model friction fields accordingly
    md.friction.coefficient=friction;

    % time stepping
    md.timestepping.time_step=0.1;
    md.timestepping.final_time=10;
    
    md.inversion.iscontrol=0;
    md.transient.isthermal=1;
    md.transient.isstressbalance=1;
    md.transient.ismasstransport=1;
    md.transient.isgroundingline=0;
    md.transient.ismovingfront=0;
    md.transient.issmb=1;

    md.cluster=generic('name',oshostname,'np',8);

    md = setflowequation(md,'SSA','all');
    md = solve(md, 'Transient');

    vel = md.results.TransientSolution(end).Vel;
    height = md.results.TransientSolution(end).Thickness;

end