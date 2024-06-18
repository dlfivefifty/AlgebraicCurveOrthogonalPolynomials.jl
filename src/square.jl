function gausssquare(n)
    x,w = gaussradau(n)
    r = -x
    ([x; fill(-1,n-1); r; fill(1,n-1)], 
    [fill(-1, n); x[2:end]; fill(1, n); r[2:end]], 
    [2w[1]; w[2:end]; w[2:end]; 2w[1]; w[2:end]; w[2:end]])
end

struct Square{T} <: Domain{SVector{2,T}} end
Square() = Square{Float64}()

function in(p::SVector{2}, d::Square)
    x,y = p
    (abs(x) == 1 && -1 ≤ y ≤ 1) || (abs(y) == 1 && -1 ≤ x ≤ 1)
end

struct LegendreSquare{T} <: Basis{T} end
LegendreSquare() = LegendreSquare{Float64}()

axes(P::LegendreSquare) = (Inclusion(Square()),blockedrange([1:3; Fill(4,∞)]))


function getindex(P::LegendreSquare{T}, xy::SVector{2}, K::Block) where T
    x,y = xy
    n = Int(K)-1
    m = n÷2
    a,b,c = -1/2,-1/2,0
    K == Block(1) && return T[1]
    K == Block(2) && return T[x, y]
    K == Block(3) && return T[wedgep(1,a,b,c,x^2,y^2),x*y,wedger(1,a,b,c,x^2,y^2)]
    iseven(n) && return T[wedgep(m,a,b,c,x^2,y^2),wedger(m,a,b,c,x^2,y^2),x*y*wedgep(m-1,a+1,b+1,c,x^2,y^2),x*y*wedger(m-1,a+1,b+1,c,x^2,y^2)]
    T[x*wedgep(m,a+1,b,c,x^2,y^2),x*wedger(m,a+1,b,c,x^2,y^2,3),y*wedgep(m,a,b+1,c,x^2,y^2),y*wedger(m,a,b+1,c,x^2,y^2,1/3)]
end

function getindex(P::LegendreSquare{T}, xy::SVector{2}, k::Int) where T
    k == 1 && return one(T)
    k ≤ 3 && return P[xy,Block(2)][k-1]
    k ≤ 6 && return P[xy,Block(3)][k-3]
    P[xy,Block((k+1) ÷ 4 + 2)][mod1(k+2,4)]
end


function plan_squaretransform(n)
    N = 6 + 4*(n-3)
    x,y,w = gausssquare(n+1)
    ret = Array{Float64}(undef, N, N)
    P = LegendreSquare()[SVector.(x,y),Block.(1:n)]
    M = P' * Diagonal(w) * P
    BlockedArray(M \ (P' * Diagonal(w)), [1:3; fill(4,n-3)], [length(x)])
end
