function [friction_field] = f_solve_inverse_problem(para, mode)

    % load parameterized model
    md = loadmodel('./Models/Greenland.Parameterization');

    % decide on ice thickness adjustment: 
    thickness = md.geometry.thickness;
    switch mode
        case 1
            % random variations everywhere
            mult = 0.01 * randn(size(md.geometry.thickness, 1), 1);
            thickness_adjustment = thickness.*mult;
        case 2
            % one uniform parameter
            mult = para;
            thickness_adjustment = thickness.*mult;
        case 3
            % adjust ice thickness: add one value everywhere
            thickness_adjustment = para * ones(md.geometry.thickness, 1);
        case 4
            % adjust ice thickness: with given field
            thickness_adjustment = para;
        otherwise
            fprintf("invalid mode")
    end

    md.geometry.thickness = max(md.geometry.thickness + thickness_adjustment, 0);
    md.geometry.base = md.geometry.surface-md.geometry.thickness;

    % setup parameters for inversion
    md = setup_inversion(md);

    %Go solve
    md.cluster=generic('name',oshostname,'np',8);
    md.verbose=verbose('solution',false,'control',true);
    md=solve(md,'Stressbalance');

    friction_field = md.results.StressbalanceSolution.FrictionCoefficient;
end