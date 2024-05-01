from netgen.geom2d import SplineGeometry
from ngsolve import Mesh

def GenerateMesh(maxh=0.07,curve_order=3):
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inflow"))
    geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
    mesh = Mesh( geo.GenerateMesh(maxh=maxh))
    mesh.Curve(curve_order)
    return mesh

if __name__ == "__main__":
    mesh = GenerateMesh()
    Draw(mesh)
