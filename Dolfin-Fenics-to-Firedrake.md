# Converting Dolfin-Fenics code to Firedrake

- No string expressions allowed, replace with Firedrake code (means some functions, e.g., `max`, are not available and have to be replaced, e.g., `firedrake.gt` greater than conditional)
- Firedrake is sensitive to using the same mesh object (idenitcal mesh in a different MeshGeometry object will not work), function object, function space object, etc.
- Firedrake has different plotting commands. These have been moved into the `FOM` class, so replace `dl.plot` with `fom.plot` wherever
