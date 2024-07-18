struct ComplexUnitCircle <: Domain{ComplexF64} end
in(x::Number, ::ComplexUnitCircle) = abs(x) ≈ 1
cardinality(::ComplexUnitCircle) = ℵ₁
checkpoints(::ComplexUnitCircle) = [exp(0.1im),exp(-0.7im)]


abstract type AbstractHermLaurent{N,T,N₂} <: Basis{SHermitianCompact{N,T,N₂}} end

"""
Represents Hermitian-valued Laurent series of the form

    A[1] + A[2]/z + A[2]'z + A[3]/z^2 + A[3]'z^2 + …

where A[k] is real. Reduces to cosine series when A[k] is 1x1.
"""
struct HermLaurent{N,T,N₂} <: AbstractHermLaurent{N,T,N₂} end

"""
Represents Hermitian-valued Laurent series of the form

    im*(A[1]/z - A[1]'z) + im*(A[2]/z^2 - A[2]'z^2) + …

where A[k] is real. Reduces to sine series when A[k] is 1x1.
"""
struct ImHermLaurent{N,T,N₂} <: AbstractHermLaurent{N,T,N₂} end

HermLaurent{N}() where N = HermLaurent{N,ComplexF64,sum(1:N)}()
ImHermLaurent{N}() where N = ImHermLaurent{N,ComplexF64,sum(1:N)}()

HermLaurent(::ImHermLaurent{N,T,N₂}) where {N,T,N₂} = HermLaurent{N,T,N₂}()
ImHermLaurent(::HermLaurent{N,T,N₂}) where {N,T,N₂} = ImHermLaurent{N,T,N₂}()


function hermlaurent(A, B)
    N = size(A,1)
    H = HermLaurent{N}()
    H * BlockedVector([SHermitianCompact{N}(A).lowertriangle; vec(B); Zeros(∞)],(axes(H,2),))
end

axes(F::HermLaurent{N,T,N₂}) where {N,T,N₂} = (Inclusion(ComplexUnitCircle()),blockedrange([N₂; Fill(N^2,∞)]))
axes(F::ImHermLaurent{N,T,N₂}) where {N,T,N₂} = (Inclusion(ComplexUnitCircle()),blockedrange(Fill(N^2,∞)))


==(::HermLaurent{N}, ::HermLaurent{N}) where N = true
==(::ImHermLaurent{N}, ::ImHermLaurent{N}) where N = true

"""
    returns a tuple which tells if an entry lies on the diag
"""
@generated function _hermitian_isdiag(::Val{N}) where N
    indexmat = Vector{Bool}()
    i = 0
    for col = 1 : N, row = col : N
        push!(indexmat, row == col)
    end
    quote
        Base.@_inline_meta
        return $(tuple(indexmat...))
    end
end

function getindex(F::HermLaurent{N,T,N₂}, z::Number, Jj::BlockIndex{1}) where {N,T,N₂}
    z in axes(F,1) || throw(BoundsError())
    J,j = block(Jj),blockindex(Jj)
    isd = _hermitian_isdiag(Val(N))
    
    ret = zero(SVector{N₂,T})
    if J == Block(1)
        ret = setindex(ret,one(T),j)
    else
        k = Int(J)
        ind = StaticArrays._hermitian_compact_indices(Val(N))
        j̃,lower = ind[j]
        ret = if isd[j̃]
            setindex(ret, z^(k-1) + z^(1-k), j̃)
        elseif lower
            setindex(ret, z^(1-k), j̃)
        else
            setindex(ret, z^(k-1), j̃)
        end
    end
    SHermitianCompact{N,T,N₂}(ret)
end

function getindex(F::ImHermLaurent{N,T,N₂}, z::Number, Jj::BlockIndex{1}) where {N,T,N₂}
    z in axes(F,1) || throw(BoundsError())
    J,j = block(Jj),blockindex(Jj)
    isd = _hermitian_isdiag(Val(N))
    
    ret = zero(SVector{N₂,T})
    k = Int(J)
    ind = StaticArrays._hermitian_compact_indices(Val(N))
    j̃,lower = ind[j]
    ret = if isd[j̃]
        setindex(ret, im*(z^(-k) - z^k), j̃)
    elseif lower
        setindex(ret, im*z^(-k), j̃)
    else
        setindex(ret, -im*z^k, j̃)
    end
    SHermitianCompact{N,T,N₂}(ret)
end

getindex(F::AbstractHermLaurent, z::Number, j::Integer) = F[z, findblockindex(axes(F,2),j)]

summary(io::IO, H::HermLaurent{N}) where N = print(io, "$(N)-dimensional HermLaurent")
summary(io::IO, H::ImHermLaurent{N}) where N = print(io, "$(N)-dimensional ImHermLaurent")

# preserve real coefficients
ldiv(H::HermLaurent{N,T}, ::HermLaurent{N,V}) where {T,V,N} = SquareEye{real(promote_type(T,V))}((axes(H,2),))

function ldiv(H::HermLaurent{N}, f::AbstractQuasiVector) where N
    F = Fourier{ComplexF64}()
    M = SetindexInterlace(SMatrix{N,N,ComplexF64},fill(F,N^2)...)
    θ = axes(M,1)
    c = M \ BroadcastQuasiVector{eltype(M)}(θ -> f[exp(im*θ)], θ)
    BS = BlockArrays.blockcolsupport(c,Block(1))
    if isempty(BS)
        ret = BlockedVector([Vector{Float64}(); Zeros(∞)], (axes(H,2),))        
    else
        M = BS[end]
        ret = BlockedVector([Vector{Float64}(undef,last(axes(H,2)[M])); Zeros(∞)], (axes(H,2),))
        
        ret[Block(1)] = real(SHermitianCompact{N}(reshape(c[Block(1)],N,N)).lowertriangle)
        for K = Block(2):M
            ret[K] = real(im*c[2K-2] + c[2K-1])/2
        end
    end
    ret
end

const HermLaurentExpansion{N} = ApplyQuasiVector{<:Any,typeof(*),<:Tuple{HermLaurent{N},Any}}
const ImHermLaurentExpansion{N} = ApplyQuasiVector{<:Any,typeof(*),<:Tuple{ImHermLaurent{N},Any}}

for op in (:+, :-)
    @eval begin
        function broadcasted(::typeof($op), A::UniformScaling, B::HermLaurentExpansion{N}) where N
            H,_ = B.args
            $op(H * BlockedArray([SHermitianCompact{N}(A(N)).lowertriangle; Zeros(∞)], (axes(H,2),)), B)
        end
        function broadcasted(::typeof($op), A::HermLaurentExpansion{N}, B::UniformScaling) where N
            H,_ = A.args
            $op(A, H * BlockedArray([SHermitianCompact{N}(B(N)).lowertriangle; Zeros(∞)], (axes(H,2),)))
        end
    end
end




# checkcommutes(A::AbstractMatrix, B::AbstractMatrix) = A*B ≈ B*A || throw(ArgumentError("Do not commute"))
# checkcommutes(X::HermLaurent{<:Any,<:SVector{1}}, Y::HermLaurent{<:Any,<:SVector{1}}) = checkcommutes(X.A[1], Y.A[1])

# """
#    _mul_hermlaurent_terms(X, Y)

# returns the non-positive terms of Laurent series X*Y.
# This suffices to check commutation, and under the conditions X and Y commute
# we can then construct the HermLaurent.

# We do it on SVector just because I haven't worked out the general pattern.
# """

# # (A + B/z + B'z)*(C + D/z + D'z) == A*C + B*D' + B'D + 1/z * (A*D + B*C) + 1/z^2 * B*D
# function _mul_hermlaurent_terms(X::SVector{2}, Y::SVector{2})
#     A,B = X
#     C,D = Y
#     SVector(A*C + B*D' + B'D, A*D + B*C, B*D)
# end
# # (A + B/z + B'z + C/z^2 + C'*z^2)*(E + F/z + F'z) == A*E + B*F' + B'F + 1/z * (A*F + B*E + C*F') + 1/z^2 * (B*F + C*E) + 1/z^3 * (C*F) + …
# function _mul_hermlaurent_terms(X::SVector{3}, Y::SVector{2})
#     A,B,C = X
#     E,F = Y
#     SVector(A*E + B*F' + B'F, A*F + B*E + C*F', B*F + C*E, C*F)
# end

# # (A + B/z + B'z)*(E + F/z + F'z  + G/z^2 + G'*z^2) == A*E + B*F' + B'F + 1/z * (A*F + B*E + B'G) + 1/z^2 * (A*G + B*F) + 1/z^3 * (B*G) + …
# function _mul_hermlaurent_terms(X::SVector{2}, Y::SVector{3})
#     A,B = X
#     E,F,G = Y
#     SVector(A*E + B*F' + B'F, A*F + B*E + B'G, A*G + B*F, B*G)
# end

# function _mul_hermlaurent_terms(X::SVector{3}, Y::SVector{3})
#     A,B,C = X
#     E,F,G = Y
#     SVector(A*E + B*F' + B'F + C*G' + C'G, A*F + B*E  + B'G + C*F', A*G + B*F  + C*E, B*G + C*F, C*G)
# end

# # (A + B/z + B'z + C/z^2 + C'*z^2)*(E + F/z + F'z + G/z^2 + G*z^2) ==
# #          A*E + B*F' + B'F + C*G' + C'G     +
# #  1/z  * (A*F + B*E  + B'G + C*F'       )    +
# # 1/z^2 * (A*G + B*F  + C*E)                            +
# # 1/z^3 * (B*G + C*F) + …
# # 1/z^4 * (C*G)

norm(A::HermLaurentExpansion) = norm(A.args[2])

# function padisapprox(X::AbstractVector, Y::AbstractVector)
#     tol = 10*(norm(map(norm,X)) + norm(map(norm,Y)))*eps()
#     m = min(length(X), length(Y))
#     for k = 1:m
#         isapprox(X[k], Y[k]; atol=tol)
#     end
#     for k = m+1:length(X)
#         norm(X[k]) ≤ tol || return false
#     end
#     for k = m+1:length(Y)
#         norm(Y[k]) ≤ tol || return false
#     end
#     return true
# end


# checkcommutes(X::HermLaurent, Y::HermLaurent) =
#     padisapprox(_mul_hermlaurent_terms(X.A,Y.A), _mul_hermlaurent_terms(Y.A,X.A)) || throw(ArgumentError("Do not commute"))

function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(*), X::HermLaurentExpansion{N},  Y::HermLaurentExpansion{N}) where N
    # checkcommutes(X, Y)
    H = HermLaurent{N}()
    H * (H \ BroadcastQuasiVector{promote_type(eltype(X),eltype(Y))}(z -> X[z]*Y[z], axes(H,1)))
end


# # Special cases for powers. We have commutation for free. (I didn't just use A .* B because some more cases were known)
# # (A + B/z + B'z)^2 == A^2 + B*B' + B'B + 1/z * (A*B + B*A) + 1/z^2 * B^2 + z * (B'*A + A*B') + z^2 * (B')^2
# function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{2}})
#     A,B = F.A
#     HermLaurent(B*B'+A^2 + B'B, B*A+A*B, B^2)
# end

# function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{3}}, ::Base.RefValue{Val{2}})
#     A,B,C = F.A
#     HermLaurent(
#     C*C' + B*B' + A^2 + B'B + C'C,
#     C*B' + B*A + A*B + B'*C,
#     C*A + B^2 + A*C,
#     C*B + B*C,
#     C^2)
# end

# function broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{3}})
#     A,B = F.A
#     HermLaurent(B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A,
#                     B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2,
#                     B*A*B + B^2*A + A*B^2, B^3)
# end

# broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, F::HermLaurent{<:Any,<:SVector{2}}, ::Base.RefValue{Val{4}}) = (F.^2).^2


broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, f::HermLaurentExpansion, ::Base.RefValue{Val{N}}) where N = 
    f.args[1] / f.args[1] \ (f.^N)

function broadcasted(::LazyQuasiArrayStyle{1}, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, f::ImHermLaurentExpansion, ::Base.RefValue{Val{N}}) where N
    H = HermLaurent(f.args[1])
    H / H \ (f.^N)
end




# padbroadcast(F, A::SVector{N}, B::SVector{N}) where N = broadcast(F, A, B)

# @generated function padbroadcast(F, A::SVector{N}, B::SVector{M}) where {N,M}
#     if N < M
#         :(vcat(broadcast(F, A, B[SOneTo($N)]), broadcast.(F, zero(eltype(eltype(A))), reverse(reverse(B)[SOneTo($(M-N))]))))
#     else # M < N
#         :(vcat(broadcast(F, A[SOneTo($M)], B), broadcast.(F, reverse(reverse(A)[SOneTo($(N-M))]), zero(eltype(eltype(B))))))
#     end
# end


# for op in (:*, :/)
#     @eval broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof($op), X::HermLaurent, c::Number) = HermLaurent(broadcast($op, X.A, c))
# end
# for op in (:*, :\)
#     @eval broadcasted(::DefaultQuasiArrayStyle{1}, ::typeof($op), c::Number, X::HermLaurent) = HermLaurent(broadcast($op, c, X.A))
# end

# isapprox(F::HermLaurent, B::UniformScaling) = F ≈ HermLaurent(B(size(F[1],1)))
# isapprox(B::UniformScaling, F::HermLaurent) = HermLaurent(I(size(F[1],1))) ≈ F

# isapprox(X::HermLaurent, Y::HermLaurent) = padisapprox(X.A, Y.A)



# blocksymtricirculant(X::HermLaurent, N) = blocksymtricirculant(X.A..., N)
# function BlockTridiagonal(X::HermLaurent)
#     A,B = X.A
#     mortar(Tridiagonal(Fill(Matrix(B'),∞), Fill(A,∞), Fill(B,∞)))
# end

# ###
# # Operators
# ###
# # note it's difff w.r.t. θ.
# diff(X::HermLaurent) = HermLaurent(-im .* (0:length(X.A)-1) .* X.A)

_hermlaurent_size(::AbstractHermLaurent{N}) where N = N
_hermlaurent_size2(::AbstractHermLaurent{N,T,N₂}) where {N,T,N₂} = N₂

function diff(H::HermLaurent; dims=1)
    N = _hermlaurent_size(H)
    N₂ = _hermlaurent_size2(H)
    iH = ImHermLaurent(H)
    T = real(eltype(eltype(H)))
    iH * _BandedBlockBandedMatrix(BlockVcat(Zeros{T}(N₂), mortar(Fill.((-one(T))*oneto(∞), N^2)))', (axes(iH,2),axes(H,2)), (-1,1), (0,0))
end

function diff(iH::ImHermLaurent; dims=1)
    N = _hermlaurent_size(iH)
    N₂ = _hermlaurent_size2(iH)
    H = HermLaurent(iH)
    T = real(eltype(eltype(H)))
    H * _BandedBlockBandedMatrix(mortar(Fill.((one(T))*oneto(∞), N^2))', (axes(H,2),axes(iH,2)), (1,-1), (0,0))
end