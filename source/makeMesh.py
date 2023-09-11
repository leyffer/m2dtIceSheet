"""! Mesh generator using Gmsh"""

import gmsh

def makeMesh(meshDim:int = 50,
              meshShape:str = "houses",
              show:bool = False,
              filename:str = None) -> str:
    """! Create the mesh for the FOM
        @param meshDim  Number of mesh nodes on each boundary
        @param meshShape  Either "square" or "houses" specifying the desired mesh
        @param show  Bring up the gmsh gui if True
        @param filename  Filename to write the mesh to
        @return  A string indicating the filename of the generated mesh"""
    # Initialize Gmsh
    gmsh.initialize()

    if filename is None:
        filename = meshShape + ".msh"

    # Create points for domain boundary
    point1 = gmsh.model.geo.add_point(x = 0, y = 0, z = 0, meshSize = 1/meshDim)
    point2 = gmsh.model.geo.add_point(1, 0, 0, 1/meshDim)
    point3 = gmsh.model.geo.add_point(1, 1, 0, 1/meshDim)
    point4 = gmsh.model.geo.add_point(0, 1, 0, 1/meshDim)
    # Connect points for the domain boundary
    line1 = gmsh.model.geo.add_line(startTag = point1, endTag = point2)
    line2 = gmsh.model.geo.add_line(point2, point3)
    line3 = gmsh.model.geo.add_line(point3, point4)
    line4 = gmsh.model.geo.add_line(point4, point1)
    # Connect lines to curve for domain boundary
    curveLoop1 = gmsh.model.geo.add_curve_loop(curveTags = [line1, line2, line3, line4])

    if meshShape == "square":
        # Create Plane Surface from Curve Loop
        planeSurface1 = gmsh.model.geo.add_plane_surface(wireTags = [curveLoop1])

        # Create Physical Groups from Lines and Surfaces
        bottom_wall_1 = gmsh.model.geo.add_physical_group(1, [line1], name = "bottom_wall")
        right_wall_2  = gmsh.model.geo.add_physical_group(1, [line2], name = "right_wall")
        top_wall_3    = gmsh.model.geo.add_physical_group(1, [line3], name = "top_wall")
        left_wall_4   = gmsh.model.geo.add_physical_group(1, [line4], name = "left_wall")
        domain_5      = gmsh.model.geo.add_physical_group(dim = 2, tags = [planeSurface1], name = "domain")

    if meshShape == "houses":
        # Create points for first house
        point5 = gmsh.model.geo.add_point(0.25, 0.15, 0, 1/meshDim)
        point6 = gmsh.model.geo.add_point(0.5,  0.15, 0, 1/meshDim)
        point7 = gmsh.model.geo.add_point(0.5,  0.4,  0, 1/meshDim)
        point8 = gmsh.model.geo.add_point(0.25, 0.4,  0, 1/meshDim)
        # Connect points for the first house boundary
        line5 = gmsh.model.geo.add_line(point5, point6)
        line6 = gmsh.model.geo.add_line(point6, point7)
        line7 = gmsh.model.geo.add_line(point7, point8)
        line8 = gmsh.model.geo.add_line(point8, point5)
        # Connect lines to curve for the first house boundary
        curveLoop2 = gmsh.model.geo.add_curve_loop([line5, line6, line7, line8])

        # Create points for second house
        point9  = gmsh.model.geo.add_point(0.6,  0.6,  0, 1/meshDim)
        point10 = gmsh.model.geo.add_point(0.75, 0.6,  0, 1/meshDim)
        point11 = gmsh.model.geo.add_point(0.75, 0.85, 0, 1/meshDim)
        point12 = gmsh.model.geo.add_point(0.6,  0.85, 0, 1/meshDim)
        # Connect points for the second house boundary
        line9  = gmsh.model.geo.add_line(point9,  point10)
        line10 = gmsh.model.geo.add_line(point10, point11)
        line11 = gmsh.model.geo.add_line(point11, point12)
        line12 = gmsh.model.geo.add_line(point12, point9)
        # Connect lines to curve for the second house boundary
        curveLoop3 = gmsh.model.geo.add_curve_loop([line9, line10, line11, line12])

        # Create surface of domain with holes where the houses are
        planeSurface1 = gmsh.model.geo.add_plane_surface(wireTags = [curveLoop1, curveLoop2, curveLoop3])

        # Create physical surfaces and curves
        bottom_wall_1 = gmsh.model.geo.add_physical_group(dim = 1, tags = [line1], name = "bottom_wall")
        right_wall_2  = gmsh.model.geo.add_physical_group(1, [line2], name = "right_wall")
        top_wall_3    = gmsh.model.geo.add_physical_group(1, [line3], name = "top_wall")
        left_wall_4   = gmsh.model.geo.add_physical_group(1, [line4], name = "left_wall")
        houses_5      = gmsh.model.geo.add_physical_group(1, [line5, line6, line7, line8, line9, line10, line11, line12], name = "left_wall")
        domain_6      = gmsh.model.geo.add_physical_group(2, [planeSurface1], name = "domain")

    # Create the relevant Gmsh data structures from the Gmsh model
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate()

    # Write the mesh data
    gmsh.write(filename)

    if show:
        # Create graphical user interface
        gmsh.fltk.run()

    # Finalize the Gmsh API
    gmsh.finalize()
    return filename
