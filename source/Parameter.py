try:
    import fenics as dl
    # for finite element setup
    class Parameter(dl.UserExpression):
        """
        This is a helper class for dealing with subdomains in FEniCS
        """

        def __init__(self, materials, k_i, **kwargs):
            super(Parameter, self).__init__(**kwargs)

            self.materials = materials
            self.k_i = k_i

        def eval_cell(self, values, x, cell):
            values[0] = self.k_i[self.materials[cell.index]]

        def set_k(self, para):
            self.k_i = para

        def value_shape(self):
            return ()
except ImportError:
    print("Parameter class not available")
    class Parameter():
        def __init__(self):
            return
