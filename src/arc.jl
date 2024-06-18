"""
The arc y ≥ h
"""
struct Arc{T} <: Domain{SVector{2,T}} 
    h::T
end
Arc{T}() where T = Arc{T}(zero(T))
Arc() = Arc{Float64}()
Arc(a::Arc) = a
Arc{T}(a::Arc) where T = Arc{T}(a.h)
cardinality(::Arc) = ℵ₁

show(io::IO, a::Arc) = print(io, "Arc($(a.h))")

# TODO: Make sure in domain
checkpoints(::Arc{T}) where T = [CircleCoordinate(0.1),CircleCoordinate(1.32),CircleCoordinate(2.9)]

function in(xy::StaticVector{2}, d::Arc)
    x,y = xy
    norm(xy) == 1 && y ≥ d.h
end

==(a::Arc, b::Arc) = a.h == b.h

const ArcInclusion{T} = Inclusion{CircleCoordinate{T},Arc{T}}
ArcInclusion{T}() where T = Inclusion{CircleCoordinate{T},Arc{T}}(Arc{T}())
ArcInclusion{T}(a::Number) where T = Inclusion{CircleCoordinate{T},Arc{T}}(Arc{T}(a))
ArcInclusion(a) = ArcInclusion{typeof(a)}(a)


"""
y^a on arc
"""
struct UltrasphericalArcWeight{T} <: Weight{T}
    h::T
    a::T
end

axes(P::UltrasphericalArcWeight{T}) where T = (ArcInclusion(P.h),)
getindex(P::UltrasphericalArcWeight, xy::StaticVector{2}) = xy[2]^P.a

"""
Ortogonal polynomials w.r.t. y^a for y^2 + x^2 = 1, y ≥ h
"""
struct UltrasphericalArc{V,TT,UU} <: AlgebraicOrthogonalPolynomial{2,V}
    h::V
    a::V
    T::TT
    U::UU
end
UltrasphericalArc{V}(h, a, T::TT, U::UU) where {V,TT,UU} = UltrasphericalArc{V,TT,UU}(h,a,T,U)

show(io::IO, P::UltrasphericalArc) = print(io, "UltrasphericalArc($(P.h), $(P.a))")

function UltrasphericalArc{V}(h,a=zero(h)) where V
    T = SemiclassicalJacobi(2/(1-h), -one(a)/2+a/2, zero(a), -one(a)/2+a/2)
    U = SemiclassicalJacobi(2/(1-h),  one(a)/2+a/2, zero(a), one(a)/2+a/2, T)
    UltrasphericalArc{V}(h, a, Normalized(T), Normalized(U))
end

function UltrasphericalArc{V}(h, a, P) where V
    T = SemiclassicalJacobi(2/(1-h), -one(a)/2+a/2, 0, -one(a)/2+a/2, P.T.P)
    U = SemiclassicalJacobi(2/(1-h),  one(a)/2+a/2, 0, one(a)/2+a/2, T)
    UltrasphericalArc{V}(h, a, Normalized(T), Normalized(U))
end


UltrasphericalArc(h, a) = UltrasphericalArc{float(promote_type(typeof(h),typeof(a)))}(h, a)
UltrasphericalArc(h, a, P) = UltrasphericalArc{float(promote_type(typeof(h),typeof(a)))}(h, a, P)
UltrasphericalArc(a) = UltrasphericalArc(zero(a), a)
UltrasphericalArc() = UltrasphericalArc(0)

axes(P::UltrasphericalArc{T}) where T = (ArcInclusion(P.h), BlockedOneTo(1:2:∞))

==(P::UltrasphericalArc, Q::UltrasphericalArc) = P.a == Q.a


function getindex(P::UltrasphericalArc{T}, xy::StaticVector{2}, JR::BlockOneTo) where T
    ret = BlockedVector{T}(undef, (axes(P,2)[JR],))
    isempty(ret) && return ret


    x,y = xy
    h = P.h
    ỹ = (1-y)/(1-h)
    ret[1] = sqrt(1-h) * P.T[ỹ,1]
    for J in Block(2):JR[end]
        ret[J] = [x*P.U[ỹ,Int(J)-1]/sqrt(1-h) ,  sqrt(1-h)*P.T[ỹ,Int(J)]]
    end
    ret    
end

summary(io::IO, P::UltrasphericalArc{V}) where V = print(io, "UltrasphericalArc($(P.h), $(P.a))")

function transform_ldiv(Pn::SubQuasiArray{T,2,<:UltrasphericalArc,<:Tuple{Inclusion,OneTo}}, f::AbstractQuasiVector{V}) where {T,V}
    _,jr = parentindices(Pn)
    P = parent(Pn)
    @assert P.h == 0
    N = maximum(jr)
    ret = Array{T}(undef, N)
    ret[1:2:end] = P.T[:,Base.OneTo((N+1) ÷ 2)] \ BroadcastQuasiVector{T}(y -> f[CircleCoordinate(asin(1-y))] + f[CircleCoordinate(π-asin(1-y))], axes(P.T,1))
    if N > 1
        ret[2:2:end] = P.U[:,Base.OneTo(N÷2)] \ BroadcastQuasiVector{T}(function(y)
            xy = CircleCoordinate(asin(1-y))
            xỹ = CircleCoordinate(π-asin(1-y))
            (f[xy] - f[xỹ])/xy[1]
        end, axes(P.T,1))
    end
    ldiv!(2,ret)
end

function transform_ldiv(Pn::SubQuasiArray{T,2,<:UltrasphericalArc,<:Tuple{Inclusion,<:BlockSlice}}, f::AbstractQuasiVector{V}) where {T,V}
    parent(Pn)[:,OneTo(size(Pn,2))] \ f
end

abstract type AbstractUltrasphericalArcJacobi{T} <: AbstractBlockBandedMatrix{T} end
Base.copy(J::AbstractUltrasphericalArcJacobi) = J # immutable

struct UltrasphericalArcJacobiX{T} <: AbstractUltrasphericalArcJacobi{T}
    R
end

UltrasphericalArcJacobiX(R) = UltrasphericalArcJacobiX{eltype(R)}(R)

struct UltrasphericalArcJacobiY{T} <: AbstractUltrasphericalArcJacobi{T}
    X_T
    X_U
end

UltrasphericalArcJacobiY(X_T, X_U) = UltrasphericalArcJacobiY{promote_type(eltype(X_T),eltype(X_U))}(X_T,X_U)



function BlockArrays.viewblock(X::UltrasphericalArcJacobiX{T}, kj::Block{2}) where T
    k,j = kj.n
    R = X.R
    k == j == 1 && return zeros(T,1,1)
    (k,j) == (2,1) && return reshape([R[1,1]; zero(T)], 2, 1)
    (k,j) == (1,2) && return [R[1,1] zero(T)]
    k == 1 && return zeros(T,1,2)
    j == 1 && return zeros(T,2,1)
    k == j && return [zero(T) R[j-1,j]; R[j-1,j] zero(T)]
    k == j+1 && return [zero(T) R[j,j]; R[j-1,j+1] zero(T)]
    k+1 == j && return [zero(T) R[k-1,k+1]; R[k,k] zero(T)]
    zeros(T,2,2)
end

function BlockArrays.viewblock(Y::UltrasphericalArcJacobiY{T}, kj::Block{2}) where T
    k,j = kj.n
    X_T, X_U = Y.X_T, Y.X_U
    k == j == 1 && return reshape([X_T[1,1]],1,1)
    (k,j) == (2,1) && return reshape([zero(T),X_T[2,1]], 2, 1)
    (k,j) == (1,2) && return [zero(T) X_T[1,2]]
    k == 1 && return zeros(T,1,2)
    j == 1 && return zeros(T,2,1)
    k == j && return [X_U[k-1,j-1] zero(T); zero(T) X_T[k,j]]
    return [X_U[k-1,j-1] zero(T); zero(T) X_T[k,j]]
end

function getindex(X::AbstractUltrasphericalArcJacobi, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end

axes(::AbstractUltrasphericalArcJacobi) = (BlockedOneTo(1:2:∞),BlockedOneTo(1:2:∞))

blockbandwidths(::AbstractUltrasphericalArcJacobi) = (1,1)
subblockbandwidths(::AbstractUltrasphericalArcJacobi) = (1,1)

function jacobimatrix(::Val{1}, P::UltrasphericalArc)
    R = P.U \ P.T
    h = P.h
    UltrasphericalArcJacobiX((1-h)*R)
end

function jacobimatrix(::Val{2}, P::UltrasphericalArc{T}) where T
    X_T = jacobimatrix(P.T)
    X_U = jacobimatrix(P.U)
    h = P.h
    UltrasphericalArcJacobiY(I + (h-1)*X_T, I + (h-1)*X_U)
end

struct UltrasphericalArcConversion{T} <: AbstractBlockBandedMatrix{T}
    R_T
    R_U
end
UltrasphericalArcConversion(R_T::AbstractMatrix{T}, R_U::AbstractMatrix{U}) where {T,U} = UltrasphericalArcConversion{promote_type(T,U)}(R_T, R_U)

axes(::UltrasphericalArcConversion) = (BlockedOneTo(1:2:∞),BlockedOneTo(1:2:∞))

blockbandwidths(::UltrasphericalArcConversion) = (0,2)

function UltrasphericalArcConversion(P::AbstractQuasiMatrix, Q::AbstractQuasiMatrix)
    @assert Q.a == P.a + 2
    R_T = Q.T \ P.T
    R_U = Q.U \ P.U
    UltrasphericalArcConversion(R_T, R_U)
end

function BlockArrays.viewblock(R::UltrasphericalArcConversion{T}, kj::Block{2}) where T
    k,j = kj.n
    R_T, R_U = R.R_T, R.R_U
    k == j == 1 && return reshape([R_T[1,1]],1,1)
    j == 1 && return reshape([zero(T),R_T[k,1]], 2, 1)
    k == 1 && return [zero(T) R_T[1,j]]
    k == j && return [R_U[k-1,j-1] zero(T); zero(T) R_T[k,j]]
    return [R_U[k-1,j-1] zero(T); zero(T) R_T[k,j]]
end

function getindex(X::UltrasphericalArcConversion, k::Int, j::Int)
    ki,ji = findblockindex.(axes(X), (k,j))
    X[block(ki),block(ji)][blockindex(ki),blockindex(ji)]
end


ArrayLayouts.MemoryLayout(::Type{<:UltrasphericalArcConversion}) = LazyBlockBandedLayout()
ArrayLayouts.MemoryLayout(::Type{<:AbstractUltrasphericalArcJacobi}) = LazyBlockBandedLayout()
Base.BroadcastStyle(::Type{<:AbstractUltrasphericalArcJacobi}) = LazyBandedMatrices.LazyArrayStyle{2}()


function \(Q::UltrasphericalArc{T}, P::UltrasphericalArc{V}) where {T,V}
    P == Q && return FillArrays.SquareEye{promote_type(T,V)}((axes(P,2),))
    UltrasphericalArcConversion(P, Q)
end
    
