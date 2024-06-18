using DelimitedFiles, BlockArrays, Test

Ax = BlockedArray(readdlm("experiments/Ax.csv"), fill(4,8), fill(4,8))
Bx = BlockedArray(readdlm("experiments/Bx.csv"), fill(4,8), fill(4,8))
Ay = BlockedArray(readdlm("experiments/Ay.csv"), fill(4,8), fill(4,8))
By = BlockedArray(readdlm("experiments/By.csv"), fill(4,8), fill(4,8))


X = z -> Ax + Bx/z + z*Bx'
Y = z -> Ay + By/z + z*By'

 z = exp(0.1im); norm(X(z)Y(z) - Y(z)X(z))
 z = exp(0.1im); norm(X(z)^4 + Y(z)^4 - I)

Ax[Block(1,2)] |> svdvals
Ax[Block(2,3)] |> svdvals
Ax[Block(3,4)] |> svdvals
Ax[Block(4,5)] |> svdvals
Ax[Block(5,6)] |> svdvals
Ax[Block(6,7)] |> svdvals
Ax[Block(7,8)] |> svdvals
Bx[Block(8,1)] |> svdvals

norm(Ax*Ay-Ay*Ax + Bx*By' + Bx'*By - By*Bx' - By'*Bx)
norm(Bx*By-By*Bx)

z = exp(0.1im)
@test Bx^4 ≈ -By^4
@test X(z)^2 ≈ Bx^2/z^2 + (Ax*Bx + Bx*Ax)/z + Ax^2 + Bx*Bx' + Bx'*Bx + z*(Bx'*Ax + Ax*Bx') + z^2*(Bx^2)'
@test X(z)^4 ≈ Bx^4/z^4 + 
            (Bx^2*Ax*Bx + Bx^3*Ax + Ax*Bx^3 + Bx*Ax*Bx^2)/z^3 + 
            (Bx^2 * (Ax^2 + Bx*Bx' + Bx'*Bx) + (Ax^2 + Bx*Bx' + Bx'*Bx)*Bx^2 + (Ax*Bx + Bx*Ax)^2)/z^2 + 
            (Bx^2*(Bx'*Ax + Ax*Bx') + (Bx'*Ax + Ax*Bx')*Bx^2 + (Ax*Bx + Bx*Ax) * (Ax^2 + Bx*Bx' + Bx'*Bx) + (Ax^2 + Bx*Bx' + Bx'*Bx) * (Ax*Bx + Bx*Ax))/z +
            Bx^2 * (Bx^2)' + (Bx^2)' * Bx^2 + (Ax*Bx + Bx*Ax) * (Bx'*Ax + Ax*Bx') + (Bx'*Ax + Ax*Bx') * (Ax*Bx + Bx*Ax) + (Ax^2 + Bx*Bx' + Bx'*Bx)^2 +
            (Bx^4/z^4 + 
            (Bx^2*Ax*Bx + Bx^3*Ax + Ax*Bx^3 + Bx*Ax*Bx^2)/z^3 + 
            (Bx^2 * (Ax^2 + Bx*Bx' + Bx'*Bx) + (Ax^2 + Bx*Bx' + Bx'*Bx)*Bx^2 + (Ax*Bx + Bx*Ax)^2)/z^2 + 
            (Bx^2*(Bx'*Ax + Ax*Bx') + (Bx'*Ax + Ax*Bx')*Bx^2 + (Ax*Bx + Bx*Ax) * (Ax^2 + Bx*Bx' + Bx'*Bx) + (Ax^2 + Bx*Bx' + Bx'*Bx) * (Ax*Bx + Bx*Ax))/z)'

B^3*B' + B^2 * B'*B + B*B'*B^2 + B'*B^3 = D^3*D' + D^2 * D'*D + D*D'*D^2 + D'*D^3

B^2 * (B^2)' + (B^2)' * B^2  + B*B'*B*B' + B*(B')^2*B + B'*B^2*B' + B'*B*B'*B + B^2 * (B^2)' + 
    (D^2)' * D^2  + D*D'*D*D' + D*(D')^2*D + D'*D^2*D' + D'*D*D'*D = I

C = Matrix(Ax[Block(1,2)])
using Plots
scatter(vcat([real.(eigvals(C/z + z*C')) for z in exp.(range(0,2π;length=100)*im)]...), zeros(400))

# Q*C*V'/z + z*V*C'*Q'
