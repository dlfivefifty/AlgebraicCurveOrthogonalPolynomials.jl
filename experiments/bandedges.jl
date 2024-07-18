###
# Try to recast bandends problem as linear algebra
###

using Plots, BlockBandedMatrices, BlockArrays, ApproxFun

X = z -> A + B/z + B'z
Ẋ = z -> -im*B/z + im*B'z

λ = z -> eigvals(Hermitian(X(z)))
λ̇ = function(z)
    _,Q = eigen(Hermitian(X(z)))
    real(diag(Q'Ẋ(z)*Q))
end

function eigplot!(λ)
    for j = 1:length(λ(1))
        plot!(θ -> λ(exp(im*θ))[j], 0, π)
    end
    current()
end

function eigplot(λ)
    p = plot()
    eigplot!(λ)
    p
end


##
# periodic symbol
##
A = [1 1; 1 -1]; B = [0 0; 1 0]






##
# random triangular
##

d = 3
A = zeros(d,d); A = A + A'
B = tril(randn(d,d))

eigplot(λ)
eigplot!(λ̇)

r1 = roots(Fun(θ -> λ̇(exp(im*θ))[1], 0..π))
r2 = roots(Fun(θ -> λ̇(exp(im*θ))[2], 0..π))


chop(Fun(z -> prod(λ(z)), Circle()).coefficients,1E-5)


plot!(θ -> real(det(X(exp(im*θ)))), 0, π)





##
# random symbol
##
d = 2
A = randn(d,d); A = A + A'
B = randn(d,d)


r1 = roots(Fun(θ -> λ̇(exp(im*θ))[1], 0..π))
r2 = roots(Fun(θ -> λ̇(exp(im*θ))[2], 0..π))
r = exp.(im.*sort([0,r1[2],r2[2],π]))

Ẋ.(r[2])  
λ̇(r[2])
λ̇(r[3])


## Monodromy matrix

II = Matrix(I,2,2)
ZZ = 0II

N = 100; J = mortar(Tridiagonal(fill(Matrix(B'),N-1), fill(A,N),fill(B,N-1)))

scatter(eigvals(Matrix(J)), zeros(size(J,2)))

M = p -> inv([II ZZ ZZ;
            ZZ II ZZ;
            B' (A-p*I) B])[end-3:end,1:4]

# powers of M tell us inv(J-z*I)
JzIinv = BlockedArray(inv([II zeros(2,2N-2); ZZ II zeros(2,2N-4); J[Block.(2:N-1),Block.(1:N)]]), fill(2,N), fill(2,N))

@test JzIinv[Block(5,1)] ≈ (M(0)^3)[3:end,1:2]

@test det(M(0)) ≈ 1 



λ(r[2])
p = λ(r[3])[1]

eigvals(M(p-0.1))
eigvals(M(p+0.1))

M(p)

(tr(M(p))/2)^2

eigvals(X(r[2]))


tr(M(0)^2)


###
#
###

d = 3
A = randn(d,d); A = A + A'
B = randn(d,d)

p = plot()
for j = 1:d
    plot!(θ -> λ(exp(im*θ))[j], 0, π)
end; p