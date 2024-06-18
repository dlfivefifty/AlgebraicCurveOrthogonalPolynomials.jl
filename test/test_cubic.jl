using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, BlockArrays, LazyBandedMatrices, LazyArrays, Test
import AlgebraicCurveOrthogonalPolynomials: LegendreCubicJacobiX

@testset "cubic" begin
    P = LegendreCubic(2)
    xy = axes(P,1)
    x = 0.1
    y = sqrt(x*(1-x)*(P.t-x))
    xy = SVector(x, y)
    
    @testset "evaluation" begin
        @test P[xy,1] == 1
        @test P[xy,2] == P.P[x,2]
        @test P[xy,3] == y*P.Q[x,1]
        @test P[xy,6] == P.P[x,4]
    end

    @testset "jacobi" begin
        X_P = jacobimatrix(P.P);
        X_Q = jacobimatrix(P.Q);
        R = P.Q \ P.P;
        R[1:10,1:10]

        X = jacobimatrix(Val(1), P)
        Y = jacobimatrix(Val(2), P)

        N = 4
        @test x*P[xy,Block.(1:N)]' ≈ P[xy,Block.(1:N+1)]' * X[Block.(1:N+1), Block.(1:N)]
        @test y*P[xy,Block.(1:N)]' ≈ P[xy,Block.(1:N+1)]' * Y[Block.(1:N+1), Block.(1:N)]

        # @testset "blocked" begin 
        #     # make 1-blocks
        #     X_P_bl = BlockedArray(view(X_P,2:∞,2:∞), Ones{Int}(∞), Ones{Int}(∞))
        #     X_Q_bl = BlockedArray(X_Q, Ones{Int}(∞), Ones{Int}(∞))
        #     Z = Zeros(axes(X_P_bl))
        #     X_tail = BlockedArray(BlockBroadcastArray{Float64}(hvcat, 2, X_P_bl, Z, Z, X_Q_bl),
        #                     Vcat(2,Fill(3,∞)), Vcat(2,Fill(3,∞)));

        #     r = BlockedArray([X_P[2,1]; Zeros(∞)], (axes(X_tail,1),))

        #     X = BlockHvcat(2, X_P[1,1], r', r, X_tail);
        # end

        @testset "symbol" begin
            t = P.t
            φ = z -> (z + sqrt(z-1)sqrt(z+1))/2
            c = -1/(2*φ(2t-1))

            R = Y.R
            @test R[200,201]/R[200,200] ≈ c atol=1e-2
            @test R[200,202]/R[200,200] ≈ -1 atol=1e-2
            @test R[200,203]/R[200,200] ≈ -c atol=1e-2


            α = sqrt(-1/(64c)) # found by solving Y^2 = X*(I-X)*(t*I-X) for α
            @test R[300,300] ≈ α atol=1e-2

            @test X[Block(21,22)]≈ [0    0 0;
                                    1/4 0 0;
                                    0   1/4 0] atol=1e-2
            @test X[Block(22,22)] ≈ [1/2    0       1/4; 
                                     0      1/2     0;
                                     1/4     0      1/2] atol=1e-2

            @test Y[Block(21,22)] ≈ α*[ 1    0    0;
                                        0   -c    0;
                                        c    0    1] atol =1e-2
            @test Y[Block(22,22)] ≈ α*[ 0   -1    0;
                                       -1    0    c;
                                        0    c    0] atol =1e-2
            @test Y[Block(23,22)] ≈ α*[-c    0   -1;
                                        0    1    0;
                                        0    0   -c] atol =1e-2

            @test Y[Block(22,23)] ≈ α*[-c    0    0;
                                        0    1    0;
                                       -1    0   -c] atol =1e-2
            @test Y[Block(23,23)] ≈ α*[ 0    c    0;
                                        c    0   -1;
                                        0   -1    0] atol =1e-2
            @test Y[Block(24,23)] ≈ α*[ 1    0    c;
                                        0   -c    0;
                                        0    0    1] atol =1e-2     
                                        
            @testset "6x6 symbols" begin
                t = 2
                c = -1/(2*φ(2t-1)) # -1/(3+2sqrt(2))
                α = sqrt(-1/(64c))
                X̄_0 = [1/2    0       1/4; 
                        0      1/2     0;
                        1/4     0      1/2]
                X̄_1 = [0        0   0;
                        1/4     0   0;
                        0       1/4 0]

                Ȳ_1 = α*[ 1    0    0;
                          0   -c    0;
                          c    0    1]
                Ỹ_1 = α*[ -c    0    0;
                          0     1    0;
                          -1    0   -c]

                Ȳ_0 = α*[ 0    c    0;
                          c    0   -1;
                          0   -1    0]
                Ỹ_0 = α*[ 0   -1    0;
                         -1    0    c;
                          0    c    0]

                Z = zeros(3,3)
                X_0 = [X̄_0 X̄_1; X̄_1' X̄_0]
                X_1 = [Z Z; X̄_1 Z]
                Y_0 = [Ȳ_0 Ȳ_1; Ȳ_1' Ỹ_0]
                Y_1 = [Z Z; Ỹ_1 Z]

                X = z -> X_0 + X_1/z + X_1'*z
                Y = z -> Y_0 + Y_1/z + Y_1'*z

                z = exp(0.1im)
                @test X(z)Y(z) ≈ Y(z)X(z)
                @test Y(z)^2 ≈ X(z)*(I-X(z))*(t*I-X(z))
            end
        end
    end
end

