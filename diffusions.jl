module DiffusionAlgorithms



function _noiterfunc{T}(iter::Int, x::Vector{T})
end

"""
The fully generic function
x <- alpha*x + gamma*v
"""
function _applyv{T}(x::Vector{T}, v, alpha::T, gamma::T)
    x *= alpha
    x += gamma*v
end

function _applyv{T}(x::Vector{T}, v::T, alpha::T, gamma::T)
    gv = gamma*v
    @simd for i in 1:length(x)
        @inbounds x[i] = alpha*x[i] + gv
    end
end

function _applyv{T}(x::Vector{T}, v::Vector{T}, alpha::T, gamma::T)
    @simd for i in 1:length(x)
        @inbounds x[i] = alpha*x[i] + gamma*v[i]
    end
end

function _applyv{T}(x::Vector{T}, v::SparseMatrixCSC{T,Int}, 
                    alpha::T, gamma::T)
    @simd for i in 1:length(x)
        @inbounds x[i] *= alpha
    end
    vvals = nonzeros(v)
    vrows = rowvals(v)
    @simd for j in nzrange(v,1) 
        @inbounds x[vrows[j]] += gamma*vvals[j]
    end
end


"""
This function computes the strongly personalized PageRank 
vector of a column sub-stochastic matrix P. 
This call allocates no extra memory.
pagerank_power 
    x - the solution vector 
    y - an extra vector of memory
    P - a duck typed matrix to apply the stochastic operator
        the type P must support P*x 
    alpha - the value of alpha in PageRank
    v - a duck typed vector to apply the personalization
        the type v must support x += v where x is a Vector{T}
        examples of v include a scalar, a sparsevec, or a Vector{T}
    tol - the solution tolerance in the error norm. 
"""
function pagerank_power!{T}(x::Vector{T}, y::Vector{T}, 
    P, alpha::T, v, tol::T, 
    maxiter::Int, iterfunc::Function)
    ialpha = 1./(1.-alpha)
    xinit = x
    _applyv(x,v,0.,1.) # iteration number 0
    iterfunc(0,x)
    for iter=1:maxiter
        # y = P*x
        A_mul_B!(y,P,x)
        gamma = 1.-alpha*sum_kbn(y)
        delta = 0.
        _applyv(y,v,alpha,gamma)
        @simd for i=1:length(x)
            @inbounds delta += abs(y[i] - x[i]) # TODO implement Kahan summation
        end
        x,y = y,x # swap
        iterfunc(iter,x)
        if delta*ialpha < tol
            break
        end
    end
    if !(x === xinit) # x is not xinit, so we need to copy it over
        xinit[:] = x
    end
    xinit # always return xinit 
end

function single_seed_pagerank_power(P,alpha::Float64,seed::Int64)
    n = size(P,1)
    v = sparsevec([seed],[1.],n)
    x = zeros(n)
    y = zeros(n)
    return pagerank_power!(x, y, P, alpha, v, 1.e-8, 10000, _noiterfunc)
end

function katz_power!{T}(x::Vector{T}, y::Vector{T}, A, alpha::T, lam1::T, v, tol::Float64, maxiter::Int)
    ialpha = 1./(1.-alpha)
    xinit = x
    _applyv(x,v,0.,1.) # iteration number 0
    for iter=1:maxiter
        # y = P*x
        A_mul_B!(y,A,x)
        gamma = 1.-alpha
        _applyv(y,v,alpha/lam1,gamma)
        delta = 0.
        @simd for i=1:length(x)
            @inbounds delta += (y[i] - x[i])^2 # TODO implement Kahan summation
        end
        x,y = y,x # swap
        if sqrt(delta)*ialpha < tol
            break
        end
    end
    if !(x === xinit) # x is not xinit, so we need to copy it over
        xinit[:] = x
    end
    xinit # always return xinit 
end

function single_seed_katz_power(A,alpha::Float64,lam1::Float64,seed::Int64)
    n = size(A,1)
    v = sparsevec([seed],[1.],n)
    x = zeros(n)
    y = zeros(n)
    return katz_power!(x, y, A, alpha, lam1, v, 1.e-8, 10000)
end


function single_seed_katz_power(A,alpha::Float64,seed::Int64)
    lam1 = real(eigs(A)[1][1])
    return katz_power!(A, alpha, lam1, seed)
end



    
"""
This computes a vector 
    exp(-(I-P)) v
where P is a column stochastic matrix 
"""
function stochastic_heat_kernel_series!{T}(
    x::Vector{T}, y::Vector{T}, z::Vector{T},
    P, t::T, v, eps::T, 
    maxiter::Int)
    
    iexpt = exp(-t)
    _applyv(y,v,0.,1.) # iteration number 0
    # scale by iexpt
    @simd for i=1:length(x)
        @inbounds x[i] = iexpt*y[i]
    end
    
    eps_exp_t = eps*exp(t)
    err = exp(t)-1.
    coeff = 1.
    
    for k=1:maxiter
        A_mul_B!(z,P,y)       # compute z = P*y
        coeff = coeff*t/k  
        @simd for i=1:length(x)
            @inbounds x[i] = x[i] + (iexpt*coeff)*z[i]
        end
        y,z = z,y # swap
        
        err = err - coeff
        if err < eps_exp_t
            break
        end
    end
    x
end



function single_seed_stochastic_heat_kernel_series(P,t::Float64,seed::Int64)
    n = size(P,1)
    v = sparsevec([seed],[1.],n)
    x = zeros(n)
    y = zeros(n)
    z = zeros(n)
    return stochastic_heat_kernel_series!(x, y, z, P, t, v, 1.e-8, 10000)
end

function _hk_taylor_degree(t::Float64, eps::Float64, maxdeg::Int) 
    eps_exp_t = eps*exp(t)
    err = exp(t)-1.
    coeff = 1.
    k::Int = 0
    while err > eps_exp_t && k < maxdeg
        k += 1
        coeff = coeff*t/k
        err = err - coeff
    end
    return max(k,1)
end

function _hk_psis(N::Int,t::Float64,eps::Float64)
    psis = zeros(Float64,N)
    psis[N] = 1.;
    for k=N-1:-1:1
        psis[k] = psis[k+1]*t/(k+1.) + 1.
    end
    
    pushcoeffs = zeros(Float64,N+1)
    
    pushcoeffs[1] = (exp(t)*eps/N)/psis[1]
    for k=2:N
        pushcoeffs[k] = pushcoeffs[k-1]*psis[k-1]/psis[k]
    end
        
    return psis, pushcoeffs
end


function hk_relax{T}(A::SparseMatrixCSC{T,Int}, seed::Int, t::Float64, eps::Float64, maxdeg::Int, maxpush::Int)
    colptr = A.colptr
    rowval = A.rowval
    n = size(A,1)
    
    N = _hk_taylor_degree(t,eps,maxdeg)
    psis, pushcoeffs = _hk_psis(N,t,eps)
    
    exp_eps_t = exp(t)*eps
    
    x = Dict{Int,Float64}()     # Store x, r as dictionaries
    r = Dict{Tuple{Int,Int},Float64}()     # initialize residual
    Q = Tuple{Int,Int}[]        # initialize queue
    npush = 0.
    
    # TODO handle a generic seed
    r[(seed,0)] = 1.
    push!(Q,(seed,0))
    
    npush = 1
    
    while length(Q) > 0 && npush <= maxpush
        v,j = shift!(Q)
        rvj = r[(v,j)]

        r[(v,j)] = 0. 
        x[v] = get(x,v,0.) + rvj

        dv = Float64(colptr[v+1]-colptr[v]) # get the degree
        update = t*rvj/(j+1.)
        mass = update/dv
        
        for nzi in colptr[v]:(colptr[v+1] - 1)
            
            u = rowval[nzi]
            next = (u,j+1)
            if j+1 == N
                x[u] += mass
                continue
            end
            rnext = get(r,next,0.)
            thresh = dv*pushcoeffs[j+1]
            if rnext < thresh && (rnext + mass) >= thresh
                push!(Q, (u,j+1))
            end
            r[next] = rnext + mass
        end
        npush += colptr[v+1]-colptr[v]
    end
    return x
end

function hk_relax_solution{T}(A::SparseMatrixCSC{T,Int}, t::Float64, seed::Int, eps::Float64)
    maxdeg = 100000
    maxpush = 10^9
    
    return hk_relax(A,seed,t,eps,maxdeg,maxpush)
end

function ppr_push{T}(A::SparseMatrixCSC{T,Int}, seed::Int, 
    alpha::Float64, eps::Float64, maxpush::Int)
    colptr = A.colptr
    rowval = A.rowval
    n = size(A,1)
        
    x = Dict{Int,Float64}()     # Store x, r as dictionaries
    r = Dict{Int,Float64}()     # initialize residual
    Q = Int[]        # initialize queue
    npush = 0.
    
    # TODO handle a generic seed
    r[seed] = 1.
    push!(Q,seed)
    
    pushcount = 0
    pushvol = 0
    
    @inbounds while length(Q) > 0 && pushcount <= maxpush
        pushcount += 1
        u = shift!(Q)
        du = Float64(colptr[u+1]-colptr[u]) # get the degree
        pushval = r[u] - 0.5*eps*du
        x[u] = get(x,u,0.0) + (1-alpha)*pushval
        r[u] = 0.5*eps*du
        
        pushval = pushval*alpha/du
        
        for nzi in colptr[u]:(colptr[u+1] - 1)
            pushvol += 1
            v = rowval[nzi]
            dv = Float64(colptr[v+1]-colptr[v]) # degree of v
            rvold = get(r,v,0.)
            rvnew = rvold + pushval
            r[v] = rvnew
            if rvnew > eps*dv && rvold <= eps*dv
                push!(Q,v)
            end
        end
    end

    return x
end

function ppr_push_solution{T}(A::SparseMatrixCSC{T,Int}, alpha::Float64, seed::Int, eps::Float64)
    maxpush = round(Int,max(1./(eps*(1.-alpha)), 2.*10^9))
    return ppr_push(A,seed,alpha,eps,maxpush)
end

function local_sweep_cut{T,V}(A::SparseMatrixCSC{T,Int}, x::Dict{Int,V})
    colptr = A.colptr
    rowval = A.rowval
    n = size(A,1)
    Gvol = A.colptr[n+1]
    
    sx = sort(collect(x), by=x->x[2], rev=true)
    S = Set{Int64}()
    volS = 0.
    cutS = 0.
    bestcond = 1.
    beststats = (1,1)
    bestset = Set{Int64}()
    for p in sx
        if length(S) == n-1
            break
        end
        u = p[1] # get the vertex
        volS += colptr[u+1] - colptr[u]
        for nzi in colptr[u]:(colptr[u+1] - 1)
            v = rowval[nzi]
            if v in S
                cutS -= 1.
            else
                cutS += 1.
            end
        end
        push!(S,u)
        if cutS/min(volS,Gvol-volS) < bestcond
            bestcond = cutS/min(volS,Gvol-volS)
            bestset = Set(S)
            beststats = (cutS,min(volS,Gvol-volS))
        end
    end
    return bestset, bestcond, beststats
end
 
function degree_normalized_sweep_cut!{T,V}(A::SparseMatrixCSC{T,Int}, x::Dict{Int,V})
    colptr = A.colptr
    rowval = A.rowval
    
    for u in keys(x)
        x[u] = x[u]/(colptr[u+1] - colptr[u])
    end
    
    return local_sweep_cut(A,x)
end

"""
Grow a cluster from a seed using a sequence of personalized PageRank vectors
"""
function ppr_grow{T}(A::SparseMatrixCSC{T,Int}, seed::Int)
    epsvals = logspace(-2,-6.5,32)
    
    alpha = 0.99
    ntrials = length(epsvals)
    
    ppr = ppr_push_solution(A,alpha,seed,epsvals[1])
    bestset,bestcond,beststats = degree_normalized_sweep_cut!(A,ppr)

    for i=2:ntrials
        ppr = ppr_push_solution(A,alpha,seed,epsvals[i])
        set,cond,stats = degree_normalized_sweep_cut!(A,ppr)
        if cond < bestcond
            bestset = set
            bestcond = cond
            beststats = stats
        end
    end
    
    return bestset, bestcond, beststats
    
end

function hk_grow{T}(A::SparseMatrixCSC{T,Int}, seed::Int)
    epsvals = [1.e-4,1.e-3,5.e-3,1.e-2]
    tvals = [10. 20. 40. 80.]
    
    @assert length(epsvals) == length(tvals)
    
    ntrials = length(tvals)
    
    hkvec = hk_relax_solution(A,tvals[1],seed,epsvals[1])
    bestset,bestcond,beststats = degree_normalized_sweep_cut!(A,hkvec)

    for i=2:ntrials
        hkvec = hk_relax_solution(A,tvals[i],seed,epsvals[i])
        set,cond,stats = degree_normalized_sweep_cut!(A,hkvec)
        if cond < bestcond
            bestset = set
            bestcond = cond
            beststats = stats
        end
    end
    
    return bestset, bestcond, beststats
end

function ppr_grow_one{T}(A::SparseMatrixCSC{T,Int}, seed::Int, alpha::Float64, eps::Float64)
    ppr = ppr_push_solution(A,alpha,seed,eps)
    return degree_normalized_sweep_cut!(A,ppr)
end

function hk_grow_one{T}(A::SparseMatrixCSC{T,Int}, seed::Int, t::Float64, eps::Float64)
    hkvec = hk_relax_solution(A,t,seed,eps)
    return degree_normalized_sweep_cut!(A,hkvec)
end


end # end module