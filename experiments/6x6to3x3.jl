using AlgebraicCurveOrthogonalPolynomials, Plots, ForwardDiff, BlockArrays
import ForwardDiff: jacobian, gradient
import AlgebraicCurveOrthogonalPolynomials: cm

# Optimise


Ax = BlockedArray(A66x, fill(3,2), fill(3,2))
Bx = BlockedArray(B66x, fill(3,2), fill(3,2))
Ay = BlockedArray(A66y, fill(3,2), fill(3,2))
By = BlockedArray(B66y, fill(3,2), fill(3,2))


Ax1 = Ax[Block(1,1)]
Ax2 = Ax[Block(2,2)]
Bx1 = Ax[Block(1,2)]
Bx2 = Bx[Block(2,1)]

Ay1 = Ay[Block(1,1)]
Ay2 = Ay[Block(2,2)]
By1 = Ay[Block(1,2)]
By2 = By[Block(2,1)]

# We want to conjugate by Q = Diagonal([…,I,Q,I,Q,…])
# so that B1*Q' == Q*B2
#
svdvals(By1)
svdvals(By2)

polar(By1)[2]
polar(By2')[2]
By1
By2
Q = [0 -1 0; 0 0 -1; -1 0 0]
# Q = Q * Diagonal([-1,1,-1])
@test Bx1*Q' ≈ Q*Bx2
@test By1*Q' ≈ Q*By2
@test Q*Ax2*Q' ≈ Ax1
@test Q*Ay2*Q' ≈ Ay1


Bx = Q*Bx2
By = Q*By2
Ax = Ax1
Ay = Ay1

X = z -> Bx/z + Ax + Bx'*z
Y = z -> By/z + Ay + By'*z

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)


function qopt(p)
    Q = qr(reshape(p,3,3)).Q
    vec(By1*Q' - Q*By2)
end
p = randn(9)
p = p - svd(jacobian(qopt,p)) \ qopt(p); norm(qopt(p))


Ỹ = BlockedArray([A66y B66y; B66y' A66y], fill(3,4), fill(3,4))


A66x





Z = zeros(6,6)
Q1 = Matrix(I,6,6)[[1,2,3,6,5,4],:]
Q = [Q1 Z; Z Q1]
X̃ = BlockedArray(Q*[A66x B66x; B66x' A66x]*Q', fill(3,4), fill(3,4))
Ỹ = BlockedArray(Q*[A66y B66y; B66y' A66y]*Q', fill(3,4), fill(3,4))

Ax = X̃[Block(1,1)]
Bx = X̃[Block(1,2)]

Ay = Ỹ[Block(1,1)]
By = Ỹ[Block(1,2)]


X = z -> B66x/z + A66x + B66x'*z
Y = z -> B66y/z + A66y + B66y'*z

scatter(vec(specgrid(X,Y)))


comroll(A, B) = [symroll(A); vec(B)]

function comunroll(p) 
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 24m))/6)
    symunroll(p[1:sum(1:N)]),reshape(p[sum(1:N)+1:end],N,N)
end
Aˣ = A66x; Bˣ = B66x;

Aˣ = Ay; Bˣ = By;
conds = function(p)
    Aʸ, Bʸ = comunroll(p)
    [vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ))]
end

X = z -> Bˣ/z + Aˣ + Bˣ'*z
Y = z -> Bʸ/z + Aʸ + Bʸ'*z


p = comroll(Aˣ, Bˣ)
K = nullspace(jacobian(conds,p))
Aʸ, Bʸ = comunroll(K*randn(size(K,2))); scatter(vec(specgrid(X,Y)))

###
# X = B/z + A + B'*z
# X^2 = B^2/z^2 + 
#    (A*B + B*A)/z +
#    A^2 + B*B' + B'*B + …
# X^3 = B^3/z^3 +
#           (B*A*B + B^2*A + A*B^2)/z^2 +
#           (B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2)/z +
#           B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A +
#           …
# 
###

"""
 Squares B/z + A + B'*z 
"""
lrntsquare(A,B) = (B*B'+A^2 + B'B, B*A+A*B, B^2)

lrntcube(A, B) = (B*B'*A + B*A*B' + A^3 + A*B*B' + A*B'*B + B'*A*B + B'*B*A, 
                    B*A^2 + B^2*B' + B*B'*B + A^2*B + A*B*A + B'*B^2,
                    B*A*B + B^2*A + A*B^2, B^3)


function conds2(c)
    Aʸ, Bʸ = comunroll(K*c)
    Y²₀,Y²₁,Y²₂ = lrntsquare(Aʸ, Bʸ)
    X²₀,X²₁,X²₂ = lrntsquare(Aˣ, Bˣ)
    X³₀,X³₁,X³₂,X³₃ = lrntcube(Aˣ, Bˣ)
    vcat(vec(4Y²₀ - (I - Aˣ - X²₀ + X³₀)),
         vec(4Y²₁ - (Bˣ - X²₁ + X³₁)),
         vec(4Y²₂ - (- X²₂ + X³₂)),
         vec(X³₃))
end

c = randn(3)
c = c - jacobian(conds2,c) \ conds2(c); norm(conds2(c)) # does not converge




# Y(z)^2 ≈ 1/4*(I - X(z))^2*(I + X(z))
# Y(z)^2 ≈ 1/4*(I - 2X(z) + X(z)^2)*(I + X(z))
# 4Y(z)^2 ≈ (I - X(z) - X(z)^2 + X(z)^3)
X⁴
\^4 

eigvals(X(1))

speccurve
Ax
Bx

Ay
By



###
# cubic perturbed circle
###

Ax = BlockedArray(Symmetric(diagm(2 => 0.5*[1,1,1,1])), fill(3,2), fill(3,2))
Bx = BlockedArray(diagm(-4 => 0.5*[1,1]), fill(3,2), fill(3,2))
ε = 0.1
ϕ = asin(ε)/2
c0 = cos(ϕ)/2
c1 = sin(ϕ)/2
c2 = -c0
c3 = -c1
Ay = BlockedArray(Symmetric(diagm(1 => [c1,c2,c1,c2,c1],3 => [c0,c3,c0])), fill(3,2), fill(3,2))
By = BlockedArray(diagm(-3 => [c3,c0,c3]), fill(3,2), fill(3,2))
By[6,1] = c2



Ax1 = Ax[Block(1,1)]
Ax2 = Ax[Block(2,2)]
Bx1 = Ax[Block(1,2)]
Bx2 = Bx[Block(2,1)]

Ay1 = Ay[Block(1,1)]
Ay2 = Ay[Block(2,2)]
By1 = Ay[Block(1,2)]
By2 = By[Block(2,1)]


# We want to conjugate by Q = Diagonal([…,I,Q,I,Q,…])
# so that B1*Q' == Q*B2
#
function qopt(p)
    Q = qr(reshape(p,3,3)).Q
    [vec(By1*Q' - Q*By2); vec(Bx1*Q' - Q*Bx2)]
end
p = randn(9)
p = p - 0.1*gradient(qopt,p); p /= norm(p); qopt(p)




p = p - svd(jacobian(qopt,p)) \ qopt(p); p /= norm(p); norm(qopt(p))