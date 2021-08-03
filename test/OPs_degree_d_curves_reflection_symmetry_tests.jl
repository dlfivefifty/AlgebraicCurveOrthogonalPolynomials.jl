using QuasiArrays, ContinuumArrays, OrthogonalPolynomialsQuasi, LinearAlgebra, BlockBandedMatrices, Test

include("OPs_degree_d_curves_reflection_symmetry.jl")

OrthogonalPolynomialsQuasi.singularitiesbroadcast(::typeof(*), ::OrthogonalPolynomialsQuasi.NoSingularities, L::LegendreWeight) = L
OrthogonalPolynomialsQuasi.singularitiesbroadcast(::typeof(*), L::LegendreWeight, ::OrthogonalPolynomialsQuasi.NoSingularities) = L

P = Normalized(OrthogonalPolynomialsQuasi.Jacobi(0,0))
x = axes(P,1)
a = 1.5
ϕ = x->  x^4 - (a^2 + 1)*x^2 + a^2
d = 4
ϕm = x.^4 - (a^2 + 1)*x.^2 .+ a^2
Pϕ = LanczosPolynomial(ϕm)
Jp = P\(x.*P)
Jpϕ = Pϕ\(x.*Pϕ)
Cϕ = Pϕ\P

@testset "even degree 4 curve" begin
    nmax = 20
    C, Jx, Jy, Cinds = LanczosSymCurve(Jp,Jpϕ,Cϕ,nmax,d)
    Cmat = C[1:qnk2ind(nmax,d-1,d),1:Cinds[qnk2ind(nmax,d-1,d)]]
    # Connection matrix
    @test norm(Cmat*Cmat'-I) < 1E-15
    # spectrum
    λ = eigvals(Matrix(Jx + im*Jy))
    λv = λ[end-30]
    xv = real(λv)
    @test sqrt(ϕ(xv))-(-imag(λv)) < 0.1 || sqrt(ϕ(xv))- imag(λv) < 0.1
end;

@testset "multiplication by x" begin
    d = 4
    nmax = 10
    X = XP(Jp,Jpϕ,nmax,d)
    xv = -0.1
    lhs = xv*P[xv,1]
    rhs = X[1,1]*P[xv,1] + X[1,2]*P[xv,2]
    @test lhs ≈ rhs
    n = nmax
    r = pnk2ind(n,0)
    cr = pnk2ind(n+1,0)
    cl = pnk2ind(n-1,0)
    lhs = xv*P[xv,n+1]
    rhs = X[r,cl]*P[xv,n] + X[r,r]*P[xv,n+1] + X[r,cr]*P[xv,n+2]
    @test lhs ≈ rhs;
    lhs = xv*Pϕ[xv,1]
    rhs = X[3,3]*Pϕ[xv,1] + X[3,5]*Pϕ[xv,2]
    @test lhs ≈ rhs;
    n = nmax
    r = pnk2ind(n,1)
    cr = pnk2ind(n+1,1)
    cl = pnk2ind(n-1,1)
    lhs = xv*Pϕ[xv,n]
    rhs = X[r,cl]*Pϕ[xv,n-1] + X[r,r]*Pϕ[xv,n] + X[r,cr]*Pϕ[xv,n+1]
    @test lhs ≈ rhs
end;

@testset "multiplication by y" begin
    d = 4
    nmax = 10
    Y = YP(Cϕ,nmax,d)
    xv = 0.1
    n = nmax
    lhs = ϕ(xv)*Pϕ[xv,n]
    rhs = 0
    #terms = minimum([n+1,d+1])
    r = pnk2ind(n,1)
    for k = 0:d
        c = pnk2ind(n-1+k,0)
        rhs += Y[r,c]*P[xv,n+k]
    end
    @test lhs ≈ rhs

    d = 4
    nmax = 10
    Y = YP(Cϕ,nmax,d)
    xv = 0.1
    n = nmax
    lhs = P[xv,n+1]
    rhs = 0
    terms = minimum([n+1,d+1])
    r = pnk2ind(n,0)
    for k = 1:terms
        c = pnk2ind(n+1-terms+k,1)
        rhs += Y[r,c]*Pϕ[xv,n+1-terms+k]
    end
    @test lhs ≈ rhs;
end;

@testset "degree 4 curve" begin

    a = 1.25
    b = 1.75
    ϕ = x -> x^4 + (b-a)*x^3 - (a*b + 1)*x^2 + (a-b)*x + a*b
    ϕm = x.^4 + (b-a)*x.^3 - (a*b + 1)*x.^2 + (a-b)*x .+ a*b
    Pϕ = LanczosPolynomial(ϕm)
    Jp = P\(x.*P)
    Jpϕ = Pϕ\(x.*Pϕ)
    Cϕ = Pϕ\P

    nmax = 31
    C, Jx, Jy, Cinds = LanczosSymCurve(Jp,Jpϕ,Cϕ,nmax,d)
    Cmat = C[1:qnk2ind(nmax-3,d-1,d),1:Cinds[qnk2ind(nmax,d-1,d)]]
    # Connection matrix
    @test norm(Cmat*Cmat'-I) < 2E-14
    # spectrum
    λ = eigvals(Matrix(Jx + im*Jy))
    λv = λ[end-30]
    xv = real(λv)
    @test sqrt(ϕ(xv))-(-imag(λv)) < 0.1 ||  sqrt(ϕ(xv))-imag(λv) < 0.1
end;

@testset "degree 5 curve" begin
    a = 1.25
    b = 2
    c = 1.75
    ϕ = x -> -x^5 + (a-b+c)*x^4 + (a*b-a*c+b*c+1)*x^3 + (-a*b*c-a+b-c)*x^2 + (-a*b+a*c-b*c)*x + a*b*c
    d = 5
    ϕm =-x.^5 + (a-b+c)*x.^4 + (a*b-a*c+b*c+1)*x.^3 + (-a*b*c-a+b-c)*x.^2 + (-a*b+a*c-b*c)*x .+ a*b*c
    Pϕ = LanczosPolynomial(ϕm)
    Jp = P\(x.*P)
    Jpϕ = Pϕ\(x.*Pϕ)
    Cϕ = Pϕ\P

    nmax = 31
    C, Jx, Jy, Cinds = LanczosSymCurve(Jp,Jpϕ,Cϕ,nmax,d)
    Cmat = C[1:qnk2ind(nmax-3,d-1,d),1:Cinds[qnk2ind(nmax,d-1,d)]]
    # Connection matrix
    @test norm(Cmat*Cmat'-I) < 1E-11
    # spectrum
    λ = eigvals(Matrix(Jx + im*Jy))
    λv = λ[end-30]
    xv = real(λv)
    @test sqrt(ϕ(xv))-(-imag(λv)) < 0.1 ||  sqrt(ϕ(xv))-imag(λv) < 0.1
end;

@testset "even degree 6 curve" begin
    ϕ = x -> -x^6 + 3.515625 + 4.8125*x^4 - 7.328125*x^2
    d = 6
    ϕm = -x.^6 + 4.8125*x.^4 - 7.328125*x.^2 .+ 3.515625
    Pϕ = LanczosPolynomial(ϕm)
    Jp = P\(x.*P)
    Jpϕ = Pϕ\(x.*Pϕ)
    Cϕ = Pϕ\P

    nmax = 31
    C, Jx, Jy, Cinds = LanczosSymCurve(Jp,Jpϕ,Cϕ,nmax,d)
    Cmat = C[1:qnk2ind(nmax-3,d-1,d),1:Cinds[qnk2ind(nmax,d-1,d)]]
    # Connection matrix
    @test norm(Cmat*Cmat'-I) < 2E-12
    # spectrum
    λ = eigvals(Matrix(Jx + im*Jy))
    λv = λ[end-30]
    xv = real(λv)
    @test sqrt(ϕ(xv))-(-imag(λv)) < 0.1 ||  sqrt(ϕ(xv))-imag(λv) < 0.1
end;
