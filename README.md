## AlgebraicCurveOrthogonalPolynomials.jl
A Julia package for orthogonal polynomials on algebraic curves


This package contains ongoing research on orthogonal polynomials on
algebraic curves. That is, given an algebraic curve in 2D defined by
`S = {(x,y) : p(x,y) = 0}` for some bivariate polynomial `p` we wish
to construct polynomials orthogonal with respect to an inner product
supported on a  subset `Ω ⊆ S`. Note to avoid degenerecies  these polynomials need to
 be thought of as polynomials modulo the vanishing ideal `I(S)` associated to `S`.
The general theory and construction is not at yet possible, so we
outline some specific cases that we have implemented below. 

This is funded by a Leverhulme Trust Research Project Grant on
"Constructive approximation on algebraic curves and surfaces".

# Arc

We can construct orthogonal polynomials on an arc, that is, 
`Ω = {(cos(θ), sin(θ)) : a ≤ θ ≤ b}`, which is a subset of 
the circle `{(x,y) : x^2 + y^2 = 1}`. 
We parameterise points on the circle by angle, using a special
type `CircleCoordinate(θ)`:
```julia
julia> CircleCoordinate(0.1)
2-element CircleCoordinate{Float64} with indices SOneTo(2):
 0.9950041652780258
 0.09983341664682815
```
For now we only support the half circle `y ≥ 0` with the 
weight `y^a`, which we construct via `UltrasphericalArc(a)`,
which is implemented in the framework of ContinuumArrays.jl:
```julia
julia> P = UltrasphericalArc() # uniform weight on the arc
UltrasphericalArc(0.0)

julia> P[CircleCoordinate(0.1),1:5] # first 5 polynomials
5-element Array{Float64,1}:
 1.0
 0.9950041652780258
 0.7024490016371341
 2.030105652576658
 0.06160390817639964
```
Note there are two (and only two) degree-`d` polynomials
apart from `d = 1`. This is accessible as the columns of `P`
are blocked a la BlockArrays.jl:
```julia
julia> P[CircleCoordinate(0.1), Block.(1:3)]
3-blocked 5-element BlockedArray{Float64,1,Array{Float64,1},Tuple{BlockedUnitRange{StepRange{Int64,Int64}}}}:
 1.0                
 ───────────────────
 0.9950041652780258 
 0.7024490016371341 
 ───────────────────
 2.030105652576658  
 0.06160390817639964
```