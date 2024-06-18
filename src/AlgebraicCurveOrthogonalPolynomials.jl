module AlgebraicCurveOrthogonalPolynomials
using FastGaussQuadrature, FastTransforms, SpecialFunctions, LinearAlgebra, BlockBandedMatrices, BlockArrays,
    ForwardDiff, ClassicalOrthogonalPolynomials, DomainSets, StaticArrays, ContinuumArrays, QuasiArrays, SemiclassicalOrthogonalPolynomials,
    MultivariateOrthogonalPolynomials, FillArrays, ArrayLayouts, LazyBandedMatrices, LazyArrays

import ForwardDiff: jacobian
import ForwardDiff: jacobian, Dual, gradient, value, partials
import FillArrays: SquareEye
import LinearAlgebra: eigvals, eigen, isapprox, norm, factorize
import FastGaussQuadrature: jacobimoment
import QuasiArrays: DefaultQuasiArrayStyle, cardinality, LazyQuasiArrayStyle
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show, summary, ==, oneto, diff, copy, sum, size, axes
import ContinuumArrays: Weight, grid, ℵ₁, ℵ₀, @simplify, ProjectionFactorization, plan_grid_transform, unweighted, weight, transform_ldiv
import ClassicalOrthogonalPolynomials: checkpoints, ShuffledR2HC, TransformFactorization, ldiv, paddeddata, jacobimatrix, orthogonalityweight, SetindexInterlace, WeightedBasis, HalfWeighted, plotgrid, golubwelsch, recurrencecoefficients
import MultivariateOrthogonalPolynomials: BlockOneTo, ModalInterlace, BlockRange1, Plan, ModalTrav
import BlockArrays: block, blockindex, BlockSlice, blockcolsupport
import BlockBandedMatrices: BlockTridiagonal, AbstractBlockBandedMatrix, blockbandwidths, subblockbandwidths, _BandedBlockBandedMatrix, band
import SemiclassicalOrthogonalPolynomials: HalfWeighted
import LazyArrays: LazyVector
import SingularIntegrals: Associated, associated, stieltjes
import LazyBandedMatrices: Tridiagonal, SymTridiagonal, BandedMatrix

import LazyBandedMatrices.LazyArraysBlockBandedMatricesExt: LazyBlockBandedLayout

export quarticjacobi, blocksymtricirculant, unroll, randspeccurve, speccurve, specgrid, speccurvemat, symroll, symunroll, spec2alg,
        wedgep, wedgeq, wedger, wedgetransform, plan_wedgetransform, plan_squaretransform, gausswedge, JacobiWedge, LegendreSquare, gausssquare,
        HermLaurent, ImHermLaurent, jointeigen, jointeigvals, BlockTridiagonal, LegendreCircle, UltrasphericalCircle, Block, SVector, CircleCoordinate,
        UltrasphericalArc, LegendreCubic, hermlaurent, TwoBandWeight, TwoBandJacobi




"""
    blocksymtricirculant(A,B, N)

Creates a Block N x N  Symmetric-Tridiagonal-Circulant matrix with diagonal
blocks A and off-diagonal block B.
"""
function blocksymtricirculant(A, B, N)
    M = size(A,1)
    ret = BlockMatrix(zeros(eltype(A),M*N,M*N), Fill(M,N), Fill(M,N))
    for K = 1:N
        ret[Block(K,K)] = A
    end
    for K = 1:N-1
        ret[Block(K,K+1)] = B
        ret[Block(K+1,K)] = B'
    end
    ret[Block(1,N)] = B'
    ret[Block(N,1)] = B
    ret
end

function symunroll(a)
    m = length(a)
    n = (isqrt(8m+1)-1) ÷ 2
    @assert sum(1:n) == m # double check formula...
    A = similar(a, n, n)
    k̃ = 1
    for j = 1:n, k = 1:j
        A[k,j] = A[j,k] = a[k̃]
        k̃ += 1
    end
    A
end

symroll(A) = mortar([A[1:k,k] for k=1:size(A,2)])

# a holds the non-symmetric entries of A
function _unroll(a, b)
    n = isqrt(length(b))
    B = reshape(b,n,n)
    @assert length(a) == sum(1:n)
    A = similar(a, n, n)
    k̃ = 1
    for j = 1:n, k = 1:j
        A[k,j] = A[j,k] = a[k̃]
        k̃ += 1
    end
    A,B
end

# split into a and b terms
function _unroll(x)
    N = length(x)
    n = (isqrt(1 + 24N)-1) ÷ 6
    m = sum(1:n)
    _unroll(x[1:m], x[m+1:end])
end

"""
unroll variables into Symmetric Ax and non-symmetric Bx
"""
function unroll(x)
    @assert iseven(length(x))
    N = length(x) ÷ 2
    Ax,Bx = _unroll(x[1:N])
    Ay,By = _unroll(x[N+1:end])
    Ax,Bx,Ay,By
end

abstract type AlgebraicOrthogonalPolynomial{d,T} <: MultivariateOrthogonalPolynomial{d,T} end


include("circle.jl")
include("arc.jl")
include("wedge.jl")
include("square.jl")
include("cubic.jl")
include("quartic.jl")

include("algcurvapprox.jl")

include("hermlaurent.jl")
include("twoband.jl")

end # module
