using BlockBandedMatrices, BlockArrays, InfiniteLinearAlgebra, Plots

e = 0.001
B = [e 0; 2 e]
A = [1 2.; 2 1]
C = Matrix(B')
J = mortar(Tridiagonal(Fill(C,∞), Fill(A,∞), Fill(B,∞)))
J = mortar(convert(AbstractMatrix{Matrix{ComplexF64}}, J.blocks))

G = function(z)
    U,L = ul(J - z*I)
    inv(L[Block(1,1)])
end

xx = range(-10,10;length=1000)
W = x -> real((G(x+1E-8im) - G(x-1E-8im))/(2π*im))

M = W.(xx)
p = plot(; title="e = $e")
for j = 1:4
    plot!(xx, getindex.(M,j))
end; p

##
# Old
##


B = randn(2,2)
A = randn(2,2) - 10I #; A = A + A'
C = Matrix(B')

N = 501; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))

function ul(A::Matrix)
    L,U = lu(A[end:-1:1,end:-1:1], Val(false))
    L[end:-1:1,end:-1:1] , U[end:-1:1,end:-1:1]
end

N = 100; U,L = ul(Matrix(J[Block.(1:N),Block.(1:N)])-(3+eps()im)I)

X = U[1:2,1:2]*L[1:2,1:2]

@test A - B*inv(X)*C ≈ X
@test inv(J)[Block(1,1)] ≈ inv(X)
R = inv(X)*C
@test A - B*R ≈ C*inv(R)
z,V = eigen(R)
@test norm((A-z[1]*B - C/z[1])V[:,1]) ≤ 1E-10
@test det(A-z[1]*B - C/z[1]) ≤ 1E-10
@test det(z[1]*(A)-z[1]^2*B - C) ≤ 1E-10
@test det(z[1]*(A)/B-z[1]^2 *I - C/B) ≤ 1E-10
@test det(z[1]^2 *I - z[1]*(A)/B + C/B) ≤ 1E-10

λ = filter!(λ -> abs(λ) ≤ 1, eigvals([zeros(2,2) Matrix(I,2,2); -C/B (A)/B]))




# inv(X) == R*inv(B)
R = -inv(X)*B
B'*R + A - x*I + B*inv(R)

B*R + A - x*I + B'*R

function op(J, x)
    N = blocksize(J,1)
    BlockedArray([Matrix(I,2,2) zeros(2,2N-2); (J-x*I)[Block.(1:N-1),:]] \ [Matrix(I,2,2); zeros(2N-2,2)], fill(2,N), [2])
end
N = 110; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))
P = op(BigFloat.(J), 10)
R = P[Block(N-1,1)]/(P[Block(N,1)])

B'R + A - x*I + B*inv(R)

X̃ = -inv(R*inv(B))
@test A-x*I - B*inv(X̃)*C ≈ X̃
inv(X)

R = -inv(X)*B

## triangular

B = tril(randn(2,2))
x = 10
A = randn(2,2); A = A + A' - x*I
C = Matrix(B')

N = 501; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))

function ul(A)
    L,U = lu(A[end:-1:1,end:-1:1], Val(false))
    L[end:-1:1,end:-1:1] , U[end:-1:1,end:-1:1]
end



U,L = ul(Matrix(J - x*I))

