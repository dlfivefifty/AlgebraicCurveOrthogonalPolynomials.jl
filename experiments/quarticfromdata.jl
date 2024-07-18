using DelimitedFiles, BlockArrays, Test
import ForwardDiff: value, Dual

Aˣ = BlockedArray(readdlm("experiments/Ax.csv"), fill(4,8), fill(4,8))
Bˣ = BlockedArray(readdlm("experiments/Bx.csv"), fill(4,8), fill(4,8))
Aʸ = BlockedArray(readdlm("experiments/Ay.csv"), fill(4,8), fill(4,8))
Bʸ = BlockedArray(readdlm("experiments/By.csv"), fill(4,8), fill(4,8))

function qr_nullspace(N, A)
    Q, R = qr(A',Val(true))
    # @assert norm(R[end-N+1:end,:]) ≤ 1E-12
    Matrix(Q)[:,end-N+1:end]
end

LinearAlgebra.nullspace(A::Matrix{<:Dual}) = nullspace(value.(A))

function qr_newunroll(Aˣ, Bˣ, c)
    N = size(Aˣ,1)

    S = Matrix(symjac(N))
    Tr = Matrix(trjac(N))
    CmBx = Matrix(cmjac(Bˣ))
    CmAx = Matrix(cmjac(Aˣ))
    J = [0S             CmBx;
        CmBx*S    cmjac(Aˣ);
        CmAx*S   (CmBx*Tr + Matrix(cmjac(Bˣ')))]

    K = qr_nullspace(size(nullspace(value.(J)),2),J)
    K = K*(K') # normalise to enforce continuity

    (Aʸ, Bʸ) = comunroll(K*c)
    (Aˣ, Bˣ),(Aʸ, Bʸ)
end

function nl(p)
    _,(Aʸ, Bʸ) = qr_newunroll(Aˣ, Bˣ,p)
    X² = lrntsquare(Bˣ, Aˣ);  X⁴ = lrntsquare(X²...)
    # X² = tuple(zeros(4,4),zeros(4,4),X²...)
    Y² = lrntsquare(Bʸ, Aʸ);  Y⁴ = lrntsquare(Y²...)
    # Y² = tuple(zeros(4,4),zeros(4,4),Y²...)
    sum(norm.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I)) .^2)
end; nl(p)
p = randn(sum(1:N)+N^2)
result = optimize(nl, p, Newton(); autodiff=:forward)
p = Optim.minimizer(result)
(Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(Aˣ, Bˣ, p)
scatter(vec(specgrid(X,Y)))




Aˣ = BlockedArray(readdlm("experiments/Ax.csv"), fill(4,8), fill(4,8)) |> Matrix
Bˣ = BlockedArray(readdlm("experiments/Bx.csv"), fill(4,8), fill(4,8)) |> Matrix
Aʸ = BlockedArray(readdlm("experiments/Ay.csv"), fill(4,8), fill(4,8)) |> Matrix
Bʸ = BlockedArray(readdlm("experiments/By.csv"), fill(4,8), fill(4,8)) |> Matrix


X = z -> Aˣ + Bˣ/z + z*Bˣ'
Y = z -> Aʸ + Bʸ/z + z*Bʸ'

specgrid(X,Y)

# evolution of points

tt = range(0,2π; length=50)
Z = acos.(real(hcat(eigvals.(X.(exp.(im.*tt)))...))')
scatter([tt; 2π .+ tt[2:end]], [Z; Z[2:end,:]]; legend=false)

scatter(real(hcat(eigvals.(X.(exp.(im.*tt)))...)))

scatter(tt,acos.(real(hcat(eigvals.(X.(exp.(im.*tt)))...))'))



###
# UNITARY


B = qr(randn(4,4)).Q/2
T = z -> B/z + B'z
tt = range(0,2π; length=50)
Z = real(hcat(eigvals.(Hermitian.(T.(exp.(im.*tt))))...))'
scatter([tt; 2π .+ tt[2:end]], [Z; Z[2:end,:]]; legend=false)

using RandomMatrices
N = 4
Aˣ = zeros(N,N)
Bˣ = rand(Haar(1),4)
newunroll(Aˣ, Bˣ,p)

function nl(p)
    _,(Aʸ, Bʸ) = qr_newunroll(Aˣ, Bˣ,p)
    X² = lrntsquare(Bˣ, Aˣ);  X⁴ = lrntsquare(X²...)
    # X² = tuple(zeros(4,4),zeros(4,4),X²...)
    Y² = lrntsquare(Bʸ, Aʸ);  Y⁴ = lrntsquare(Y²...)
    # Y² = tuple(zeros(4,4),zeros(4,4),Y²...)
    sum(norm.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I)) .^2)
end; nl(p)


function nl(p)
    Bˣ,c = qr(reshape(p[1:N^2],N,N)).Q/2,p[N^2+1:end]
    _,(Aʸ, Bʸ) = qr_newunroll(zeros(N,N), Bˣ,c)
    X² = lrntsquare(Bˣ, Aˣ);  X⁴ = lrntsquare(X²...)
    # X² = tuple(zeros(4,4),zeros(4,4),X²...)
    Y² = lrntsquare(Bʸ, Aʸ);  Y⁴ = lrntsquare(Y²...)
    # Y² = tuple(zeros(4,4),zeros(4,4),Y²...)
    # vcat(vec.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I))...)
    sum(norm.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I)) .^2)
end; nl(p)

# Bˣ = rand(Haar(1),4)/2
N = 4

p = randn(N^2 + sum(1:N)+N^2)
p = p - jacobian(nl,p) \ nl(p); norm(nl(p))
p = randn(N^2 + 7)

p = randn(N^2 + sum(1:N)+N^2)
result = optimize(nl, p; autodiff=:forward, iterations=10_000)

p = Optim.minimizer(result)
Bˣ,c = qr(reshape(p[1:N^2],N,N)).Q/2,p[N^2+1:end]
(Aˣ, Bˣ),(Aʸ, Bʸ) = newunroll(Aˣ, Bˣ, c)
scatter(vec(specgrid(X,Y)))