using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test, QuasiArrays

@testset "Circle" begin
    @testset "LegendreCircle" begin
        P = LegendreCircle()
        T,U = ChebyshevT(),ChebyshevU()
        xy = axes(P,1)

        @testset "eval" begin
            x = 0.1; y = sqrt(1-x^2)
            @test P[SVector(x,y),Block(1)] == [1.0]
            @test P[SVector(x,y),Block(2)] == [y*U[0.1,1], T[0.1,2]]
            @test P[SVector(x,y),Block(3)] == [y*U[0.1,2], T[0.1,3]]
        end

        @testset "axes" begin
            x = 0.1; y = sqrt(1-x^2)
            @test xy[SVector(x,y)] == SVector(x,y)
            @test xy[SVector(x,y)] isa CircleCoordinate
            @test xy[CircleCoordinate(0.1)] ≡ CircleCoordinate(0.1)
            @test CircleCoordinate(0.1) in xy
            @test first.(xy)[SVector(x,y)] == 0.1
        end

        @testset "Clenshaw" begin
            @test (P * [1; zeros(∞)])[CircleCoordinate(0.1)] ≈ 1
            @test (P * [0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ sin(0.1)
            @test (P * [0; 0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ cos(0.1)
        end

        @testset "transform" begin
            x,y = first.(xy),last.(xy)
            @test x[CircleCoordinate(0.1)] == cos(0.1) 
            @test P[:,Base.OneTo(5)] \ x ≈ [0,0,1,0,0]
            @test (P \ y)[Block(2)] ≈ [1,0]
            @test (P \ x)[Block(2)] ≈ [0,1]

            u = P * (P \ cos.(x .+ exp.(y)))
            @test u[CircleCoordinate(0.1)] ≈ cos(cos(0.1) + exp(sin(0.1)))
        end

        @testset "Jacobi operator" begin
            x,y = first.(xy),last.(xy)
            X = P \ (x .* P)
            Y = P \ (y .* P)
            s = CircleCoordinate(0.1)
            x̃,ỹ = s
            @test x̃*P[s,Block.(1:10)]' ≈ P[s,Block.(1:11)]' * X[Block.(1:11),Block.(1:10)]
            @test ỹ*P[s,Block.(1:10)]' ≈ P[s,Block.(1:11)]' * Y[Block.(1:11),Block.(1:10)]
        end
    end

    @testset "UltrasphericalCircle" begin
        P = UltrasphericalCircle(0)
        
        @testset "Legendre as UltrasphericalCircle" begin
            x = 0.1; y = sqrt(1-x^2)
            
            T,U = Jacobi(-1/2,-1/2),Jacobi(1/2,1/2)
            @test P[SVector(x,y),Block(1)] == [1.0]
            @test P[SVector(x,y),Block(2)] == [y*U[0.1,1], T[0.1,2]]
            @test P[SVector(x,y),Block(3)] == [y*U[0.1,2], T[0.1,3]]

            @test (P * [1; zeros(∞)])[CircleCoordinate(0.1)] ≈ 1
            @test (P * [0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ sin(0.1)
            @test (P * [0; 0; 1; zeros(∞)])[CircleCoordinate(0.1)] ≈ cos(0.1)/2

            xy = axes(P,1)
            x,y = first.(xy),last.(xy)
            @test (P \ y)[Block(2)] ≈ [1,0]
            @test (P \ x)[Block(2)] ≈ [0,2]
            u = P * (P \ cos.(x .+ exp.(y)))
            @test u[CircleCoordinate(0.1)] ≈ cos(cos(0.1) + exp(sin(0.1)))
        end

        @testset "1" begin
            C = UltrasphericalCircle(1)
            xy = axes(C,1)
            x,y = first.(xy),last.(xy)
            u = C * (C \ cos.(x .+ exp.(y)))
            @test u[CircleCoordinate(0.1)] ≈ cos(cos(0.1) + exp(sin(0.1)))

            v = C \ P[:,10]
            @test (C*v)[CircleCoordinate(0.1)] ≈ P[CircleCoordinate(0.1),10]
        end

        @testset "Sparse relationships" begin
            T,C = LegendreCircle(),UltrasphericalCircle(2)
            xy = axes(C,1)
            x,y = first.(xy),last.(xy)
            @test norm((C \ T[:,20])[Block.(1:8)]) ≤ 100eps()
        end
    end
end