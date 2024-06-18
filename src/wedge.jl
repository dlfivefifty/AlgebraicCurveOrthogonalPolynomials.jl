function gausswedge(n, a=0, b=0, c=0, σ=1)
    x,w_x = gaussradau(n,a,c)
    y,w_y = gaussradau(n,b,c)
    reverse!(x); reverse!(w_x)
    reverse!(y); reverse!(w_y)
    x .= (1 .- x)./2; ldiv!(jacobimoment(a,c), w_x);
    y .= (1 .- y)./2; ldiv!(jacobimoment(b,c)/σ, w_y);
    [x; ones(length(x)-1)], [ones(length(x)); y[1:end-1]], [w_x[1:end-1]; w_x[end]+w_y[end]; w_y[1:end-1]]
end

_binomial(n, k) = exp(loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1))
logpochhammer(a, n) = loggamma(a+n) - loggamma(a)

function wedgep(n, a, b, c, x, y)
    Pᶜᵃ = jacobi(c,a,0..1)
    Pᶜᵇ = jacobi(c,b,0..1)
    Pᶜᵃ[x,n+1] + Pᶜᵇ[y,n+1] - _binomial(n+c,n)
end

function wedgeq(n, a, b, c, x, y, σ=1)
    Pᶜᵃ = jacobi(c+2,a,0..1)
    Pᶜᵇ = jacobi(c+2,b,0..1)
    exp(logpochhammer(c+a+2,n)-logpochhammer(a+1,n-1))*(1-x) * Pᶜᵃ[x,n] - 
        exp(logpochhammer(c+b+2,n)-logpochhammer(b+1,n-1))/σ * (1-y)*Pᶜᵇ[y,n]
end

function wedger(n, a, b, c, x, y, σ=1)
    a == b && σ == 1 && return wedgeq(n,a,b,c,x,y)
    x̃,ỹ,w = gausswedge(n+2,a,b,c,σ)
    p̃ = wedgep.(n,a,b,c, x̃,ỹ)
    μ = dot(p̃,Diagonal(w),wedgeq.(n,a,b,c,x̃,ỹ,σ))
    wedgeq(n,a,b,c,x,y,σ) - μ/dot(p̃,Diagonal(w),p̃) * wedgep(n,a,b,c,x,y)
end

wedgep(n, x, y) = wedgep(n, 0,0,0, x,y)
wedgeq(n, x, y) = wedgeq(n, 0,0,0, x,y)
wedger(n, x, y) = wedgeq(n, x, y)

function wedgemassmatrix(n)
    x,y,w = gausswedge(n)
    N = 2n-1
    ret = Vector{Float64}(undef, N)
    ret[1] = dot(p0, Diagonal(w), p0)
    for m = 1:n-1
        p0 = wedgep.(m, x, y)
        r0 = wedger.(m, x, y)
        ret[2m] = dot(p0, Diagonal(w), p0)
        ret[2m+1] = dot(r0, Diagonal(w), r0)
    end
    BlockedArray(Diagonal(ret), [1; fill(2,n-1)], [1; fill(2,n-1)])
end

function plan_wedgetransform(n)
    N = 2n-1
    x,y,w = gausswedge(n)
    ret = Array{Float64}(undef, N, N)
    p0 = wedgep.(0, x, y)
    σ = dot(p0, Diagonal(w), p0)
    ret[1,:] .= wedgep.(0, x, y) .* w ./ σ
    for m = 1:n-1
        p0 = wedgep.(m, x, y)
        r0 = wedger.(m, x, y)
        σ = dot(p0, Diagonal(w), p0)
        μ = dot(r0, Diagonal(w), r0)
        ret[2m,:] .= wedgep.(m, x, y) .* w ./ σ
        ret[2m+1,:] .= wedger.(m, x, y) .* w ./ μ
    end
    BlockedArray(ret, [1; fill(2,n-1)], [N])
end

wedgetransform(v::AbstractVector) = plan_wedgetransform(length(v) ÷ 2 + 1) * v



struct Wedge{T} <: Domain{SVector{2,T}} end
Wedge() = Wedge{Float64}()

function in(p::SVector{2}, d::Wedge)
    x,y = p
    (x == 1 && 0 ≤ y ≤ 1) || (y == 1 && 0 ≤ x ≤ 1)
end

struct JacobiWedge{T} <: AlgebraicOrthogonalPolynomial{2,T}
    a::T
    b::T
    c::T
    σ::T
    JacobiWedge{T}(a, b, c, σ) where T = new{T}(a, b, c, σ)
end

JacobiWedge(a::A, b::B, c::C, σ::Σ=1) where {A,B,C,Σ} = JacobiWedge{float(promote_type(A,B,C,Σ))}(a,b,c,σ)

axes(P::JacobiWedge) = (Inclusion(Wedge()),blockedrange([1; Fill(2,∞)]))

function getindex(P::JacobiWedge, xy::SVector{2}, j::BlockIndex{1})
    K,k = block(j),blockindex(j)
    k == 1 ? wedgep(Int(K)-1, P.a, P.b, P.c, xy...) : wedger(Int(K)-1, P.a, P.b, P.c, xy..., P.σ)
end

function getindex(P::JacobiWedge, xy::SVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end