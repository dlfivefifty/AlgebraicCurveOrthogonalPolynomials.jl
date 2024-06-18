using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials,
    BandedMatrices, BlockBandedMatrices, BlockArrays, QuasiArrays, ArrayLayouts,
    SemiclassicalOrthogonalPolynomials, Test, Random, LazyArrays, LazyBandedMatrices
using ForwardDiff, StaticArrays
import ClassicalOrthogonalPolynomials: jacobimatrix


@testset "arc" begin
    @testset "semicircle" begin
        @testset "OPs" begin
            @testset "Legendre arc" begin
                P = UltrasphericalArc()

                x,y = CircleCoordinate(0.1)
                @test P[CircleCoordinate(0.1),1] == P.T[1-y,1]
                @test P[CircleCoordinate(0.1),2] ≈ x*P.U[1-y,1]
                @test P[CircleCoordinate(0.1),3] ≈ P.T[1-y,2]
                @test P[CircleCoordinate(0.1),4] ≈ x*P.U[1-y,2]
                @test P[CircleCoordinate(0.1),5] ≈ P.T[1-y,3]

                @testset "expansion" begin
                    𝐱 = axes(P,1)
                    x,y = first.(𝐱),last.(𝐱)
                    f = exp.(cos.(y) .+ exp.(x))
                    u = P * [P[:,Base.OneTo(40)] \ f; Zeros(∞)];
                    @test u[CircleCoordinate(0.1)] ≈ f[CircleCoordinate(0.1)]
                    u = P * (P \ f)
                    @test u[CircleCoordinate(0.1)] ≈ f[CircleCoordinate(0.1)]

                    @test P == P
                    @test P \ P ≡ Eye((axes(P,2),))

                    u = P / P \ @.(sqrt((x+2)^2-1))
                    @test u[CircleCoordinate(acos(0.1))] ≈ sqrt(2.1^2 - 1)
                end

                @testset "Jacobi" begin
                    T,U = P.T,P.U;
                    X_T = jacobimatrix(T)
                    X_U = jacobimatrix(U)
                    R = U \ T;
                    L = T \ (SemiclassicalJacobiWeight(2,1,0,1) .* U);
                    @test L[1:10,1:10]' ≈ R[1:10,1:10]
                    𝐱 = CircleCoordinate(0.1)
                    x,y = 𝐱
                    @test y*T[y,1:3]' ≈ T[y,1:4]'*X_T[1:4,1:3]
                    @test (1-y) * P[𝐱, 1] ≈ P[𝐱,[1,3]]'* X_T[1:2, 1]
                    X̃_T = I-X_T;
                    X̃_U = I-X_U;
                    @test x * P[𝐱, 1] ≈ P[𝐱,2] * R[1,1]
                    @test x * P[𝐱, 2] ≈ P[𝐱,1:2:5]' * R[1,1:3]
                    @test x * P[𝐱, 3] ≈ P[𝐱,2:2:4]' * R[1:2,2]
                    @test x * P[𝐱, 4] ≈ P[𝐱,3:2:7]' * R[2,2:4]
                    @test x * P[𝐱, 5] ≈ P[𝐱,2:2:6]' * R[1:3,3]
                    @test y * P[𝐱, 1] ≈ P[𝐱,1:2:3]' * X̃_T[1:2, 1]
                    @test y * P[𝐱, 2] ≈ P[𝐱,2:2:4]' * X̃_U[1:2, 1]
                    @test y * P[𝐱, 3] ≈ P[𝐱,1:2:5]' * X̃_T[1:3, 2]
                    @test y * P[𝐱, 4] ≈ P[𝐱,2:2:6]' * X̃_U[1:3, 2]


                    X = jacobimatrix(Val(1), P)
                    Y = jacobimatrix(Val(2), P)
                    @test X[Block(1,1)] == zeros(1,1)
                    @test X[Block(2,1)] isa Matrix
                    @test issymmetric(X[1:10,1:10])
                    @test P[𝐱,Block.(1:6)]' * X[Block.(1:6),Block.(1:5)] ≈ x * P[𝐱,Block.(1:5)]'
                    @test P[𝐱,Block.(1:6)]' * Y[Block.(1:6),Block.(1:5)] ≈ y * P[𝐱,Block.(1:5)]'

                    @testset "nice syntax" begin
                        P = UltrasphericalArc()
                        𝐱 = axes(P,1)
                        x,y = first.(𝐱),last.(𝐱)
                        X = (x .* P).args[2]
                        @test MemoryLayout(X) isa AlgebraicCurveOrthogonalPolynomials.LazyBlockBandedLayout
                        
                        X = P \ (x .* P)
                        Y = P \ (y .* P)
                        # TODO: overload broadcasted(*, ::Ones, ...) for this case

                        @test X[Block.(1:5),Block.(1:5)] == jacobimatrix(Val(1), P)[Block.(1:5),Block.(1:5)]
                    end
                end
            end
            @testset "a = 1" begin
                P = UltrasphericalArc()
                U = UltrasphericalArc(2)
                @test norm((U \ P[:,10])[Block.(1:3)]) ≤ 100eps()
            end
        end

        @testset "Circulant" begin
            Ax = Matrix(0.5I,2,2)
            Bx = Matrix(0.25I,2,2)
            a₁₂ = (1 + sqrt(2))/4
            a₂₁ = (1 - sqrt(2))/4
            Ay = [0 -0.5; -0.5 0]
            By = [0 a₁₂; a₂₁ 0]

            N = 5
            X = blocksymtricirculant(Ax, Bx, N)
            Y = blocksymtricirculant(Ay, By, N)
            @test X == X'
            @test Y == Y'
            @test X*Y ≈ Y*X
            @test X^2 + Y^2 ≈ I(2N)

            x = z -> Ax + (Bx/z + Bx'*z)
            y = z -> Ay + (By/z + By'*z)
            @test x(1) ≈ I(2)
            @test norm(y(1)) ≤ 10eps()
            @test norm(x(-1)) ≤ 10eps()
            @test y(-1) ≈ [0 -1; -1 0]
            @test eigvals(y(-1)) ≈ [-1,1]

            @testset "Nonlinear Function" begin
                E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
                Σ = mortar(Fill([1 0; 0 1], N)')
                Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')
                @test Σ*X*E ≈ Ax + Bx + Bx'
                @test Ω*X*E ≈ Ax - Bx - Bx'

                F = (X,Y) -> [Σ*X*E - I, Σ*Y*E, Ω*X*E, Ω*Y*E - [0 -1; -1 0],  X*Y - Y*X, X^2 + Y^2 - I]
                @test norm(norm.(F(X,Y))) ≤ 10eps()
            end


            # @testset "Nonlinear solve" begin
            #     xd = randn(7)
            #     yd = randn(7)
            #     Ax,Bx = unroll(xd...)
            #     Ay,By = unroll(yd...)
            #     N = 7
            #     X = blocksymtricirculant(Ax, Bx, N)
            #     Y = blocksymtricirculant(Ay, By, N)

            #     [(X*Y - Y*X)[2:5,1]; (X*Y - Y*X)[3,2]; (X*Y - Y*X)[5,2]; (X^2 + Y^2 - I)[1:6,1]; (X^2 + Y^2 - I)[2:6,2]]

            #     Ax*Ay-Ay*Ax + Bx*By'-By'*Bx + Bx'*By - By*Bx'
            #     Ax*By-Ay*Bx + Bx*Ay-By*Ax
            #     Bx*By - By*Bx

            #     Ax^2 + Bx*Bx' + Bx'*Bx + Ay^2 + By*By' + By'*By - I
            #     Bx*Ax + Ax*Bx + By*Ay + Ay*By # Bx'*Ax + Ax*Bx' + By'*Ay + Ay*By'
            #     Bx^2 + By^2
            # end
        end

        # Ax = [0 0.0; 0.0 0]; Bx = [0.5 0; 0 0.5]
        # Ay = [0 0.0; 0.0 0]; By = [0 0.5; -0.5 0]
        function F_circle(x)
            Ax,Bx,Ay,By = unroll(x)
            N = 5
            X = blocksymtricirculant(Ax, Bx, N)
            Y = blocksymtricirculant(Ay, By, N)
            E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
            Σ = mortar(Fill([1 0; 0 1], N)')
            Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')

            vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E + I, Ω*Y*E,  X*Y - Y*X, X^2 + Y^2 - I])...)
            # vcat(map(vec,[Σ*X*E - I, Ω*X*E + I,  X*Y - Y*X, X^2 + Y^2 - I])...)
        end

        function F_semicircle(x)
            Ax,Bx,Ay,By = unroll(x)
            N = 5
            X = blocksymtricirculant(Ax, Bx, N)
            Y = blocksymtricirculant(Ay, By, N)
            E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
            Σ = mortar(Fill([1 0; 0 1], N)')
            Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')

            vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E, Ω*Y*E - [0 -1; -1 0],  X*Y - Y*X, X^2 + Y^2 - I])...)
        end

        Random.seed!(0)

        @testset "Newton" begin
            @testset "circle" begin
                p = randn(14)
                for _ = 1:15
                    J = ForwardDiff.jacobian(F_circle,p); p = p - (J \ F_circle(p))
                end
                Ax,Bx,Ay,By = unroll(p)
                @test norm(Ax) ≤ 10eps()
                @test Bx ≈ [0.5 0; 0 0.5]
                @test norm(Ay) ≤ 10eps()
                @test By ≈ [0 0.5; -0.5 0]
            end
            @testset "semicircle" begin
                x0 = randn(14)
                for _ = 1:10
                    J = ForwardDiff.jacobian(F_semicircle,x0); x0 = x0 - (J \ F_semicircle(x0))
                end
                Ax,Bx,Ay,By = unroll(x0)
                @test Ax ≈ Matrix(0.5I,2,2)
                @test Bx ≈ Matrix(0.25I,2,2)
                a₁₂ = (1 + sqrt(2))/4
                a₂₁ = (1 - sqrt(2))/4
                @test Ay ≈ [0 -0.5; -0.5 0]
                @test By ≈ [0 a₁₂; a₂₁ 0]
            end
        end

        @testset "Jacobi" begin
            x₂ = -(1/4)*sqrt(4+2*sqrt(2))
            x₀ = -1/sqrt(2)
            N = 100;
            X = BandedMatrix{Float64}(undef, (N,N), (2,2))
            X[band(0)] = [x₀ 1/2*ones(1,N-2) x₀]
            X[band(1)] .= X[band(-1)] .= 0;
            X[band(2)] = X[band(-2)] = [x₂ 1/4*ones(1,N-4) x₂]

            a₁₂ = (1 + sqrt(2))/4
            a₂₁ = (1 - sqrt(2))/4
            y₁=(1/4)*sqrt(4-2*sqrt(2))
            Y = BandedMatrix{Float64}(undef, (N,N), (3,3))
            Y[band(0)].=0
            Y[band(2)].=Y[band(-2)].=0
            d1 = [y₁ repeat([-1/2 a₂₁],1,Int64(round(N/2))-1)]
            d1[N-1] = y₁
            d3 = repeat([0 a₁₂],1,Int64(round(N/2)))
            Y[band(1)] = Y[band(-1)] = d1[1:N-1]
            Y[band(3)] = Y[band(-3)] = d3[1:N-3]

            @test X*Y ≈ Y*X
            @test X^2 + Y^2 ≈ Eye(N)
        end
    end

    @testset "other arc" begin
        h = 0.1
        P = UltrasphericalArc(h, 0)
        @test_throws BoundsError P[CircleCoordinate(0.0),1]
        @test P[CircleCoordinate(0.4),1] ≈ sqrt(1-h)*P.T[0.1,1] # 0.1 is arbitray

        X = jacobimatrix(Val(1), P);
        Y = jacobimatrix(Val(2), P);

        𝐱 = CircleCoordinate(0.4)
        x,y = 𝐱
        @test P[𝐱,Block.(1:6)]' * X[Block.(1:6),Block.(1:5)] ≈ x * P[𝐱,Block.(1:5)]'
        @test P[𝐱,Block.(1:6)]' * Y[Block.(1:6),Block.(1:5)] ≈ y * P[𝐱,Block.(1:5)]'

        @test (X^2 + Y^2)[Block.(1:5), Block.(1:5)] ≈ I
    end

    @testset "lowering/raising asymptotics -> symbol" begin
        P = UltrasphericalArc()
        X,Y = jacobimatrix(Val(1),P),jacobimatrix(Val(2),P)

        φ = z -> (z + sqrt(z-1)sqrt(z+1))/2

        @testset "Semicircle" begin
            c = -1/(2φ(2*2-1))
            N = 100
            B_x,A_x = X[Block(N-1),Block(N)],X[Block(N),Block(N)]
            B_y,A_y = Y[Block(N-1),Block(N)],Y[Block(N),Block(N)]

            α = 1/(2*(1+c))
            @test B_x ≈ [0 α*c; α 0] atol=1E-3
            @test α^2*c ≈ -1/16 # this ensures X^2 + Y^2 cancel
        end

        @testset "Other arc" begin
            h = 0.1
            # map via (1-x)/(1-h) to 0..1 to determine Jacobi operators
            A_y,B_y = (1+h)/2 * Eye(2), (h-1)/4 * Eye(2)
            t₀ = 2/(1-h)
            c = -1/(2φ(2*t₀-1))
            α = sqrt(-((h-1)/4)^2/c)

            @test α^2 * c ≈ -((h-1)/4)^2
            
            A_x,B_x = α*(1+c)*[0 1; 1 0], α*[0 c; 1 0]
            # check X^2 + Y^2 = I
            X = z -> A_x + (B_x'/z + B_x*z)
            Y = z -> A_y + (B_y'/z + B_y*z)

            @test X(exp(0.2im))^2 + Y(exp(0.2im))^2 ≈ I

            @test B_x^2 ≈ -B_y^2
            @test A_x*B_x + B_x*A_x ≈ -(A_y*B_y + B_y*A_y)
        end
    end
end