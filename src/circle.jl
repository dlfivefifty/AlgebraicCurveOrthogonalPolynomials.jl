struct CircleCoordinate{T} <: StaticVector{2,T}
    θ::T
end

getindex(S::CircleCoordinate, k::Int) = k == 1 ? cos(S.θ) : sin(S.θ)
norm(S::CircleCoordinate{T}) where T = one(T)

in(::CircleCoordinate, ::UnitCircle) = true

function convert(::Type{CircleCoordinate{T}}, xy::SVector{2}) where T
    x,y = xy
    CircleCoordinate{T}(atan(y,x))
end

const CircleInclusion{T} = Inclusion{CircleCoordinate{T},UnitCircle{T}}
CircleInclusion{T}() where T = Inclusion{CircleCoordinate{T},UnitCircle{T}}(UnitCircle{T}())


"""
y^a on unit circle
"""
struct UltrasphericalCircleWeight{T} <: Weight{T}
    a::T
end

axes(P::UltrasphericalCircleWeight{T}) where T = (CircleInclusion{T}(),)
getindex(P::UltrasphericalCircleWeight, xy::StaticVector{2}) = xy[2]^P.a


"""
Ortogonal polynomials w.r.t. uniform weight on circle. Equivalent to
Fourier.
"""
struct LegendreCircle{T} <: AlgebraicOrthogonalPolynomial{2,T} end
LegendreCircle() = LegendreCircle{Float64}()


==(::LegendreCircle, ::LegendreCircle) = true
axes(P::LegendreCircle{T}) where T = (CircleInclusion{T}(), BlockedOneTo(1:2:∞))

function getindex(P::LegendreCircle{T}, xy::StaticVector{2}, j::BlockIndex{1}) where T
    x,y = xy
    K,k = block(j),blockindex(j)
    K == Block(1) && return one(T)
    k == 1 ? y*ChebyshevU{T}()[x,Int(K)-1] : ChebyshevT{T}()[x,Int(K)]
end


function getindex(P::LegendreCircle, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

function getindex(P::LegendreCircle{T}, xy::StaticVector{2}, JR::BlockOneTo) where T
    N = Int(maximum(JR))
    ret = BlockedVector{T}(undef, Vcat(1, Fill(2,N-1)))
    isempty(JR) && return ret
    x,y = xy
    copyto!(view(ret.blocks,1:2:2N-1), view(ChebyshevT{T}(),x,OneTo(N)))
    copyto!(view(ret.blocks,2:2:2N-1), view(ChebyshevU{T}(),x,OneTo(N-1)))
    lmul!(y, view(ret.blocks,2:2:2N-1))
    ret
end

checkpoints(::LegendreCircle{T}) where T = CircleCoordinate.(checkpoints(Fourier{T}()))
grid(Pn::SubQuasiArray{T,2,<:LegendreCircle,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T = 
    CircleCoordinate.(grid(Fourier{T}()[:,parentindices(Pn)[2]]))
factorize(Pn::SubQuasiArray{T,2,<:LegendreCircle,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(Pn), ShuffledR2HC{T}(size(Pn,2)))

factorize(L::SubQuasiArray{T,2,<:LegendreCircle,<:Tuple{<:Inclusion,<:BlockSlice{BlockRange1{OneTo{Int}}}}},d...) where T =
    ProjectionFactorization(factorize(parent(L)[:,OneTo(size(L,2))],d...),parentindices(L)[2])

    

function jacobimatrix(::Val{1}, ::LegendreCircle{T}) where T
    F = Fourier{T}()
    θ = axes(F,1)
    F \ (cos.(θ) .* F)
end

function jacobimatrix(::Val{2}, ::LegendreCircle{T}) where T
    F = Fourier{T}()
    θ = axes(F,1)
    F \ (sin.(θ) .* F)
end

"""
Ortogonal polynomials w.r.t. y^a
"""
struct UltrasphericalCircle{T} <: AlgebraicOrthogonalPolynomial{2,T}
    a::T
    UltrasphericalCircle{T}(a) where T = new{T}(a)
end

UltrasphericalCircle(a::T) where T = UltrasphericalCircle{float(T)}(a)

axes(P::UltrasphericalCircle{T}) where T = axes(LegendreCircle{T}())

function getindex(P::UltrasphericalCircle{T}, xy::StaticVector{2}, j::BlockIndex{1}) where T
    x,y = xy
    K,k = block(j),blockindex(j)
    K == Block(1) && return one(T)
    k == 1 ? y*Jacobi((P.a + 1)/2, (P.a + 1)/2)[x,Int(K)-1] :  Jacobi((P.a - 1)/2, (P.a - 1)/2)[x,Int(K)] 
end

function getindex(P::UltrasphericalCircle, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

function getindex(P::UltrasphericalCircle{T}, xy::StaticVector{2}, JR::BlockOneTo) where T
    N = Int(maximum(JR))
    ret = BlockedVector{T}(undef, Vcat(1, Fill(2,N-1)))
    isempty(JR) && return ret
    x,y = xy
    copyto!(view(ret.blocks,1:2:2N-1), view(Jacobi((P.a - 1)/2, (P.a - 1)/2),x,OneTo(N)))
    copyto!(view(ret.blocks,2:2:2N-1), view(Jacobi((P.a + 1)/2, (P.a + 1)/2),x,OneTo(N-1)))
    lmul!(y, view(ret.blocks,2:2:2N-1))
    ret
end


function ldiv(C::UltrasphericalCircle{V}, f) where V
    P = LegendreCircle{V}()
    cfs = P \ f;
    c = paddeddata(cfs);
    c[1:2:end] = cheb2jac(c[1:2:end], (C.a-1)/2, (C.a-1)/2);
    c[2:2:end] = cheb2jac(ultra2cheb(c[2:2:end], 1), (C.a+1)/2, (C.a+1)/2);
    cfs
end
