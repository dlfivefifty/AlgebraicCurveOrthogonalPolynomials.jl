"""
CubicCurve represents y^2 = x * (1-x) * (t-x) for x in 0..1
"""
struct CubicCurve{T} <: Domain{T}
    t::T
end

CubicCurve{T}() where T = CubicCurve{T}(zero(T))
CubicCurve() = CubicCurve{Float64}()
CubicCurve(a::CubicCurve) = a
CubicCurve{T}(a::CubicCurve) where T = CubicCurve{T}(a.h)
cardinality(::CubicCurve) = ℵ₁

show(io::IO, a::CubicCurve) = print(io, "CubicCurve($(a.t))")

# TODO: Make sure in domain
checkpoints(d::CubicCurve{T}) where T = [SVector(0.1, sqrt(0.1*(1-0.1)*(d.t-0.1)))]

function in(xy::StaticVector{2}, d::CubicCurve)
    x,y = xy
    0 ≤ x ≤ 1 && y^2 ≈ x*(1-x)*(d.t-x)
end

const CubicCurveInclusion{T} = Inclusion{SVector{2,T},CubicCurve{T}}
CubicCurveInclusion{T}() where T = Inclusion{SVector{2,T},CubicCurve{T}}(CubicCurve{T}())
CubicCurveInclusion{T}(a::Number) where T = Inclusion{SVector{2,T},CubicCurve{T}}(CubicCurve{T}(a))
CubicCurveInclusion(a) = CubicCurveInclusion{typeof(a)}(a)



struct LegendreCubic{V,PP,QQ} <: AlgebraicOrthogonalPolynomial{2,V}
    t::V
    P::PP
    Q::QQ
end
LegendreCubic{V}(t, P::PP, Q::QQ) where {V,PP,QQ} = LegendreCubic{V,PP,QQ}(t,P,Q)

function LegendreCubic{V}(t) where V
    P = SemiclassicalJacobi(t, 0, 0, 0)
    Q = SemiclassicalJacobi(t,  1, 1, 1, P)
    LegendreCubic{V}(t, Normalized(P), Normalized(Q))
end


LegendreCubic(t) = LegendreCubic{float(typeof(t))}(t)

# block sizes are [1,2,3,3,3,3…] so cum is [1,3,6,9,…]
axes(P::LegendreCubic{T}) where T = (CubicCurveInclusion(P.t), BlockedOneTo(Vcat(1, 3:3:∞)))

==(P::LegendreCubic, Q::LegendreCubic) = P.t == Q.t

function getindex(P::LegendreCubic{T}, xy::StaticVector{2}, J::Block{1}) where T
    x,y = xy
    t = P.t
    J == Block(1) && return [one(T)]
    J == Block(2) && return [P.P[x,2], y*P.Q[x,1]]
    J̃ = Int(J)
    m = 3J̃ ÷ 2
    isodd(J̃) && return [P.P[x,m-1], y*P.Q[x, m-2], P.P[x, m]]
    #iseven(J̃)
    return [y*P.Q[x, m-3], P.P[x,m-1], y*P.Q[x, m-2]]
end

function getindex(P::LegendreCubic{T}, xy::StaticVector{2}, Jj::BlockIndex{1}) where T
    x,y = xy
    J,j = block(Jj),blockindex(Jj)
    P[xy,J][j]
end

getindex(P::LegendreCubic{T}, xy::StaticVector{2}, Jj::BlockOneTo) where T = layout_getindex(P, xy, Jj)

getindex(P::LegendreCubic, xy::StaticVector{2}, j::Int) = P[xy, findblockindex(axes(P,2),j)]
summary(io::IO, P::LegendreCubic{V}) where V = print(io, "LegendreCubic($(P.t))")

abstract type AbstractLegendreCubicJacobi{T} <: AbstractBlockBandedMatrix{T} end
Base.copy(J::AbstractLegendreCubicJacobi) = J # immutable

axes(::AbstractLegendreCubicJacobi) = (BlockedOneTo(Vcat(1, 3:3:∞)), BlockedOneTo(Vcat(1, 3:3:∞)))

function getindex(X::AbstractLegendreCubicJacobi, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end


struct LegendreCubicJacobiX{T} <: AbstractLegendreCubicJacobi{T}
    X_P
    X_Q
end

struct LegendreCubicJacobiY{T} <: AbstractLegendreCubicJacobi{T}
    R
end

LegendreCubicJacobiX(X_P, X_Q) = LegendreCubicJacobiX{promote_type(eltype(X_P),eltype(X_Q))}(X_P,X_Q)
LegendreCubicJacobiY(R) = LegendreCubicJacobiY{eltype(R)}(R)

function BlockArrays.viewblock(X::LegendreCubicJacobiX{T}, kj::Block{2}) where T
    k,j = kj.n
    X_P, X_Q = X.X_P, X.X_Q
    k == j == 1 && return reshape([X_P[1,1]],1,1)
    (k,j) == (2,1) && return reshape([X_P[2,1],0], 2, 1)
    (k,j) == (1,2) && return [X_P[1,2] 0]
    k == 1 && return zeros(T,1,3)
    j == 1 && return zeros(T,3,1)
    k == j == 2 && return [X_P[2,2] 0; 0 X_Q[1,1]]
    (k,j) == (3,2) && return [X_P[3,2] 0; 0 X_Q[2,1]; 0 0]
    (k,j) == (2,3) && return [X_P[2,3] 0 0; 0 X_Q[2,1] 0]
    k == 2 && return zeros(T,2,3)
    j == 2 && return zeros(T,3,2)
    if k == j
        m = 3k ÷ 2
        if isodd(k) 
            return [X_P[m-1,m-1] 0              X_P[m-1,m]; 
                    0            X_Q[m-2,m-2]   0       ;
                    X_P[m,m-1] 0              X_P[m,m]]
        else
            return [X_Q[m-3,m-3] 0              X_Q[m-3,m-2]; 
                    0            X_P[m-1,m-1]   0       ;
                    X_Q[m-2,m-3] 0              X_Q[m-2,m-2]]
        end
    elseif k == j+1
        m = 3j ÷ 2
        if iseven(k) 
            return [0        X_Q[m-1,m-2]             0; 
                    0           0                   X_P[m+1,m];
                    0           0                     0]
        else
            return [0        X_P[m,m-1]             0; 
                    0           0                   X_Q[m-1,m-2];
                    0           0                     0]
        end
    elseif k+1 == j
        m = 3k ÷ 2
        if iseven(j) 
            return [0           0             0; 
                    X_Q[m-2,m-1]           0                   0;
                    0           X_P[m,m+1]                     0]
        else
            return [0        0             0; 
                    X_P[m,m-1]           0                   0;
                    0           X_Q[m-1,m-2]                     0]
        end
    else
        zeros(T, 3, 3)
    end
end

function BlockArrays.viewblock(Y::LegendreCubicJacobiY{T}, kj::Block{2}) where T
    k,j = kj.n
    R = Y.R
    k == j == 1 && return zeros(T,1,1)
    (k,j) == (2,1) && return reshape([0,R[1,1]], 2, 1)
    (k,j) == (1,2) && return [0 R[1,1]]
    k == 1 && return zeros(T,1,3)
    j == 1 && return zeros(T,3,1)
    k == j == 2 && return [ 0       R[1,2]; 
                            R[1,2]  0]
    (k,j) == (3,2) && return [  0           R[1,3]; 
                                R[2,2]      0; 
                                0            R[1,4]]
    (k,j) == (2,3) && return [0 R[2,2] 0; 
                                R[1,3] 0 R[1,4]]
    k == 2 && return zeros(T,2,3)
    j == 2 && return zeros(T,3,2)
    if k == j
        m = 3k ÷ 2
        if isodd(k) 
            return [0           R[m-2,m-1]          0;
                    R[m-2,m-1]      0               R[m-2,m];
                    0           R[m-2,m]            0]
        else
            return [0           R[m-3,m-1]          0;
                    R[m-3,m-1]      0               R[m-2,m-1];
                    0           R[m-2,m-1]          0]
        end
    elseif k == j+1
        m = 3j ÷ 2
        if iseven(k)
            return [R[m-1,m-1]      0               R[m-1,m];
                    0               R[m-2,m+1]      0;
                    0               0               R[m,m]]
        else
            return [R[m-3,m]      0               R[m-2,m];
                    0               R[m-1,m-1]      0;
                    0               0               R[m-2,m+1]]
        end
    elseif k+1 == j
        m = 3k ÷ 2
        if iseven(j) 
            return [R[m-1,m-1]      0               0;
                    0                R[m-2,m+1]     0;
                    R[m-1,m]        0               R[m,m]]
        else
            return [R[m-3,m]      0                 0;
                    0               R[m-1,m-1]      0;
                    R[m-2,m]       0               R[m-2,m+1]]
        end
    else
        zeros(T, 3, 3)
    end
end



jacobimatrix(::Val{1}, P::LegendreCubic) = LegendreCubicJacobiX(jacobimatrix(P.P), jacobimatrix(P.Q))
jacobimatrix(::Val{2}, P::LegendreCubic) = LegendreCubicJacobiY(P.Q \ P.P)