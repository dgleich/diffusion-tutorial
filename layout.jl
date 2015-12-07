module DiffusionTutorialLayout

using PyCall
@pyimport igraph

""" 
This function computes a graph layout of a sparse matrix
by calling igraph via the python interface, so you must 
have PyCall installed in Julia and 
python-igraph installed as well. There are a bunch
of pip modules called igraph, only one of which
is the right one, so use
    pip install python-igraph
if you aren't sure!

@param A - the input sparse matrix, the values are ignored, it is treated as a directed graph
@param layoutname - an igraph layout, e.g.

    bipartite: bipartite layout 
    circle, circular: circular layout 
    drl: DrL layout for large graphs 
    drl_3d: 3D DrL layout for large graphs
    fr, fruchterman_reingold: Fruchterman-Reingold layout
    fr_3d, fr3d, fruchterman_reingold_3d: 3D Fruchterman-Reingold layout 
    grid: regular grid layout in 2D 
    grid_3d: regular grid layout in 3D 
    graphopt: the graphopt algorithm 
    gfr, grid_fr, grid_fruchterman_reingold: grid-based Fruchterman-Reingold layout 
    kk, kamada_kawai: Kamada-Kawai layout 
    kk_3d, kk3d, kamada_kawai_3d: 3D Kamada-Kawai layout 
    lgl, large, large_graph: Large Graph Layout 
    mds: multidimensional scaling layout 
    random: random layout 
    random_3d: random 3D layout 
    rt, tree, reingold_tilford: 
    rt_circular, reingold_tilford_circular: circular Reingold-Tilford tree layout 
    sphere, spherical, circle_3d, circular_3d: spherical layout 
    star: star layout
    sugiyama: Sugiyama layout 

(copied from igraph documentation)


"""
function igraph_layout{T}(A::SparseMatrixCSC{T}, layoutname::AbstractString="lgl")
    (ei,ej) = findnz(A)
    edgelist = [(ei[i]-1,ej[i]-1) for i = 1:length(ei)]
    nverts = size(A)
    G = igraph.Graph(nverts, edges=edgelist, directed=true)
    layoutname = "fr"
    xy = G[:layout](layoutname)
    xy = [ Float64(xy[i][j]) for i in 1:length(xy),  j in 1:length(xy[1])]
end

end