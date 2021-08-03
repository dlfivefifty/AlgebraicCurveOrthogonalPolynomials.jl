function LanczosCubic3DPolys(nmax,α₁,β₁,α₂,β₂,b)

    # Generate the OPs inside the cubic y² = ϕ(x) = (1 - x²)(b - x) via Lanczos on the 3-variable/trivariate OPs Pₙ,ₖ,ᵢ

    Y = YP(nmax,α₁,β₁,α₂,β₂,b)
    X = XP(nmax,α₁,β₁,α₂,β₂,b)
    (r,c) = size(X)
    C = zeros(r,2r)

    # store the leftmost and rightmost nonzero columns of each row of C in Cinds

    Cinds = zeros(Int, (nmax+1)*(nmax+2) ÷ 2, 2)
    # Jacobi operators of the OPs
    l,u = 1,1
    rows = 1:nmax+1
    cols = 1:nmax+1
    Jx = BlockBandedMatrix(Zeros{typeof(b)}(sum(rows),sum(cols)), rows,cols, (l,u))
    Jy = BlockBandedMatrix(Zeros{typeof(b)}(sum(rows),sum(cols)), rows,cols, (l,u))

    tol = 1E-9
    f = x -> abs(x) > tol

    # n = 0

    C[1,1] = 1 # Q00 = P001
    Cinds[1,:] = [1 1]

    r = nki2ind(0,0,1)
    inds =  [nki2ind(0,0,1), nki2ind(1,0,1), nki2ind(1,1,1)]
    xQ00 = X[r,inds]
    A0x = [C[1,1]*X[r,inds[1]]*C[1,1]]
    xQ00[1] -= A0x[1,1]*C[1,1]
    B0x = zeros(1,2)
    B0x[1,1] =  norm(xQ00)
    xQ00 = xQ00/B0x[1,1]
    C[2,inds] = xQ00
    lind = findfirst(f,xQ00)
    rind = findlast(f,xQ00)
    Cinds[2,:] = [inds[lind] inds[rind]]
    Jx[Block(1,1)][1,1] = A0x[1,1]
    Jx[Block(1,2)]   = B0x
    Jx[Block(2,1)] = B0x';

    inds = [nki2ind(1,1,2)]
    yQ00 = Y[r,inds]
    B0y = zeros(1,2)
    B0y[1,2] =  norm(yQ00)
    yQ00 = yQ00/B0y[1,2]
    C[3,inds] =  yQ00
    lind = findfirst(f,yQ00)
    rind = findlast(f,yQ00)
    Cinds[3,:] = [inds[lind] inds[rind]]
    Jy[Block(1,2)]  = B0y
    Jy[Block(2,1)] = B0y'

    # n = 1

    l, r = Xminds(Cinds[2,1]:Cinds[2,2])
    xQ10 = C[2,Cinds[2,1]:Cinds[2,2]]*X[Cinds[2,1]:Cinds[2,2],1:r]
    A1x = zeros(2,2)
    A1x[1,1] = dot(xQ10[Cinds[2,1]:Cinds[2,2]],C[2,Cinds[2,1]:Cinds[2,2]])
    xQ10[1] -= Jx[Block(2,1)][1,1]
    B1x = zeros(2,3)
    B1x[1,1] = norm(xQ10)
    C[4,1:r] = xQ10/B1x[1,1]
    lind = findfirst(f,xQ10)
    rind = findlast(f,xQ10)
    Cinds[4,:] = [lind[2] rind[2]]

    l, r = Xminds(Cinds[3,1]:Cinds[3,2])
    xQ11 = C[3,Cinds[3,1]:Cinds[3,2]]*X[Cinds[3,1]:Cinds[3,2],1:r]
    A1x[2,2] = dot(xQ11[1,Cinds[3,1]:Cinds[3,2]],C[3,Cinds[3,1]:Cinds[3,2]])
    xQ11[1,Cinds[3,1]:Cinds[3,2]] -= A1x[2,2]*C[3,Cinds[3,1]:Cinds[3,2]]
    B1x[2,2] = norm(xQ11)
    xQ11 = xQ11/B1x[2,2]
    C[5,1:r] = xQ11
    Jx[Block(2,2)] = A1x
    lind = findfirst(f,xQ11)
    rind = findlast(f,xQ11)
    Cinds[5,:] = [lind[2] rind[2]];
    Jx[Block(2,3)]   = B1x
    Jx[Block(3,2)] = B1x'

    l, r = Yminds(Cinds[2,1]:Cinds[2,2])
    yQ10 = C[2,Cinds[2,1]:Cinds[2,2]]*Y[Cinds[2,1]:Cinds[2,2],1:r]
    A1y = zeros(2,2)
    A1y[1,2] = A1y[2,1] = dot(yQ10[Cinds[3,1]:Cinds[3,2]],C[3,Cinds[3,1]:Cinds[3,2]])
    yQ10[Cinds[3,1]:Cinds[3,2]] -= A1y[1,2]*C[3,Cinds[3,1]:Cinds[3,2]]
    B1y = zeros(2,3)
    B1y[1,2] = dot(yQ10[1,Cinds[5,1]:Cinds[5,2]],C[5,Cinds[5,1]:Cinds[5,2]])

    l, r = Yminds(Cinds[3,1]:Cinds[3,2])
    yQ11 = C[3,Cinds[3,1]:Cinds[3,2]]*Y[Cinds[3,1]:Cinds[3,2],1:r]
    yQ11[1] -= B0y[1,2]
    yQ11[Cinds[2,1]:Cinds[2,2]] -= A1y[2,1]*C[2,Cinds[2,1]:Cinds[2,2]]
    B1y[2,1] = dot(yQ11[Cinds[4,1]:Cinds[4,2]],C[4,Cinds[4,1]:Cinds[4,2]])
    yQ11[Cinds[4,1]:Cinds[4,2]] -= B1y[2,1]*C[4,Cinds[4,1]:Cinds[4,2]]
    B1y[2,3] = norm(yQ11)
    Jy[Block(2,2)] = A1y
    Jy[Block(2,3)]   = B1y
    Jy[Block(3,2)] = B1y';
    C[nk2ind(2,2),1:r] = yQ11/B1y[2,3]
    lind = findfirst(f,yQ11)
    rind = findlast(f,yQ11)
    Cinds[nk2ind(2,2),:] = [lind[2] rind[2]]

    #nmax = 10
    for n = 2:nmax-1
        # Compute Ax
        Ax = zeros(n+1,n+1)
        # diagonal entries
        for k = 0:n
            r = nk2ind(n,k)
            Ax[k+1,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],Cinds[r,1]:Cinds[r,2]]*C[r,Cinds[r,1]:Cinds[r,2]]
        end
        # off-diagonal entries
        for k = 0:n-2
            r = nk2ind(n,k)
            for c = k+3:2:n+1
                Ax[k+1,c] = Ax[c,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],Cinds[r+c-k-1,1]:Cinds[r+c-k-1,2]]*C[r+c-k-1,Cinds[r+c-k-1,1]:Cinds[r+c-k-1,2]]
            end
        end
        Jx[Block(n+1,n+1)] = Ax

        # Orthogonalize xQn to form Qₙ₊₁,₀, …, Qₙ₊₁,ₙ
        Bx = zeros(n+1,n+2)
        for k = 0:n
            r = nk2ind(n,k)
            # Multiply  by x
            l, right = Xminds(Cinds[r,1]:Cinds[r,2])
            xQn = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],1:right]
            # Subtract off components along lower degree OPs
            # first for OPs of degree n-1
            if k < n
                for c = k+1:2:n
                    l = nk2ind(n-1,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n+1,n)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            end
            # next for OPs of degree n
            if mod(k,2)==0
                for c = 1:2:n+1
                    l = nk2ind(n,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n+1,n+1)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            else
                for c = 2:2:n+1
                    l = nk2ind(n,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n+1,n+1)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            end
            # Finally OPs of degree n+1
            if k > 1
                if mod(k,2) == 0
                    for c = 1:2:k-1
                        l = nk2ind(n+1,c-1)
                        xQn[Cinds[l,1]:Cinds[l,2]] -= Bx[k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                    end
                else
                    for c = 2:2:k-1
                        l = nk2ind(n+1,c-1)
                        xQn[Cinds[l,1]:Cinds[l,2]] -= Bx[k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                    end
                end
            end
            # Normalise
            Bx[k+1,k+1] = norm(xQn)
            C[nk2ind(n+1,k),1:right] = xQn/Bx[k+1,k+1]
            lind = findfirst(f,xQn)
            rind = findlast(f,xQn)
            Cinds[nk2ind(n+1,k),:] = [lind[2] rind[2]]

            # Compute the entries in column k+1 of Bx:

            if k < n-1
                for row = k+3:2:n+1
                    ind = nk2ind(n,row-1)
                    Bx[row,k+1] = C[ind,Cinds[ind,1]:Cinds[ind,2]]'*X[Cinds[ind,1]:Cinds[ind,2],lind[2]:rind[2]]*C[nk2ind(n+1,k),lind[2]:rind[2]]
                end
            end
        end
        Jx[Block(n+1,n+2)] = Bx
        Jx[Block(n+2,n+1)] = Bx'

        # Compute Ay
        Ay = zeros(n+1,n+1)
        for k = 0:n-1
            r = nk2ind(n,k)
            for c = k+2:2:n+1
                Ay[k+1,c] = Ay[c,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*Y[Cinds[r,1]:Cinds[r,2],Cinds[r+c-k-1,1]:Cinds[r+c-k-1,2]]*C[r+c-k-1,Cinds[r+c-k-1,1]:Cinds[r+c-k-1,2]]
            end
        end
        Jy[Block(n+1,n+1)] = Ay

        # Compute By column by column
        By = zeros(n+1,n+2)
        ind = nk2ind(n+1,0)
        # first column
        for r = 2:2:n+1
            l = nk2ind(n,r-1)
            By[r,1] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
        end
        # rest of the columns
        for c = 2:n+1
            ind = nk2ind(n+1,c-1)
            for r = c-1:2:n+1
                l = nk2ind(n,r-1)
                By[r,c] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
            end
        end
        # orthogonalize yQ_{n,n} to find Q_{n+1,n+1}
        # Multiply Q_{n,n} by y
        r = nk2ind(n,n)
        l, right = Yminds(Cinds[r,1]:Cinds[r,2])
        yQn = C[r,Cinds[r,1]:Cinds[r,2]]'*Y[Cinds[r,1]:Cinds[r,2],1:right]
        # subtract off OPs of degree n-1
        l = nk2ind(n-1,n-1)
        yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n+1,n)][n+1,n]*C[l,Cinds[l,1]:Cinds[l,2]]
        # subtract off OPs of degree n
        if mod(n+1,2)==0
            for c = 1:2:n+1
                l = nk2ind(n,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n+1,n+1)][n+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        else
            for c = 2:2:n+1
                l = nk2ind(n,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n+1,n+1)][n+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        end
        # subtract off OPs of degree n+1
        if mod(n+1,2)==0
            for c = 1:2:n
                l = nk2ind(n+1,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= By[n+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        else
            for c = 2:2:n
                l = nk2ind(n+1,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= By[n+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        end

        # normalise
        By[n+1,n+2] = norm(yQn)
        C[nk2ind(n+1,n+1),1:right] = yQn/By[n+1,n+2]
        lind = findfirst(f,yQn)
        rind = findlast(f,yQn)
        Cinds[nk2ind(n+1,n+1),:] = [lind[2] rind[2]]
        Jy[Block(n+1,n+2)] = By
        Jy[Block(n+2,n+1)] = By'
    end

    Ax = zeros(nmax+1,nmax+1)
    # compute the diagonal elements
    for k = 0:nmax
        r = nk2ind(nmax,k)
        Ax[k+1,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],Cinds[r,1]:Cinds[r,2]]*C[r,Cinds[r,1]:Cinds[r,2]]
    end
    # off-diagonal elements
    for r = 1:nmax-1
        row = nk2ind(nmax,r-1)
        for c = r+2:2:nmax+1
            Ax[r,c] = Ax[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*X[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
        end
    end

    Jx[Block(nmax+1,nmax+1)] = Ax

    Ay = zeros(nmax+1,nmax+1)
    for r = 1:nmax
        row = nk2ind(nmax,r-1)
        for c = r+1:2:nmax+1
            Ay[r,c] = Ay[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*Y[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
        end
    end
    Jy[Block(nmax+1,nmax+1)] = Ay;

    C, Jx, Jy, Cinds

end


function XP(nmax,α₁,β₁,α₂,β₂,b)

    # Construct the operator representing the multiplication of the 3-variable polynomials by x for all polynomials of degree <= nmax
    # α₁ and β₁ are the parameters of the Jacobi weight of the boundary OPs
    # α₂ and β₂ are the parameters of the Jacobi weight of the OPs in z
    # b > 1 is the third root of the cubic ϕ = (1 - x²)(b - x)

    l,u = 1,1          # block bandwidths
    rows = vcat(1,3*(1:nmax+1))
    cols = vcat(1,3*(1:nmax+1))
    X = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))

    pw = Normalized(SemiclassicalJacobi((b+1)/2, β₁, α₁, 0.0))
    Jw = jacobimatrix(pw)
    pϕw = Normalized(SemiclassicalJacobi((b+1)/2, β₁+1, α₁+1, 1))
    Jϕw = jacobimatrix(pϕw)

    begin
        k = 0
        pzk = Normalized(SemiclassicalJacobi((b+1)/2, 2k+1+β₂, α₂, 0.0))
        Jzk = jacobimatrix(pzk)
        pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
        Jzk1 = jacobimatrix(pzk1)
        Ck1 = pzk1\pzk

        for n = k:nmax
            row = Block(n+1)[1]
            kient = 2*Jw[k+1,k+1] - 1
            X[row,row] = Jzk[n+1,n+1]*kient
            col = Block(n+2)[1]
            X[row,col] = X[col,row] = Jzk[n+1,n+2]*kient

            kient = 2*Jw[k+1,k+2]
            col = Block(n+2)[2]
            X[row,col] = X[col,row] = Ck1[n+1,n+1]*kient
            if n > 0
                col = Block(n+1)[2]
                X[row,col] = X[col,row] = Ck1[n,n+1]*kient
            end
        end

        Jzk = Jzk1
        pzk = pzk1
        Ck = Ck1
    end

    begin
        k = 1
        pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
        Jzk1 = jacobimatrix(pzk1)
        Ck1 = pzk1\pzk

        for n = k:nmax
            # (n,1,1)
            row = nki2ind(n,k,1)

            kient = 2*Jw[k,k+1]
            col = nki2ind(n+1,0,1)
            X[row,col] = X[col,row] = Ck[n,n+2]*kient

            kient = 2*Jw[k+1,k+1] - 1
            X[row,row] = Jzk[n,n]*kient
            col = nki2ind(n+1,k,1)
            X[row,col] = X[col,row] = Jzk[n,n+1]*kient

            kient = 2*Jw[k+1,k+2]
            col = nki2ind(n+1,k+1,2)
            X[row,col] = X[col,row] = Ck1[n,n]*kient
            if n > 1
                col = nki2ind(n,k+1,2)
                X[row,col] = X[col,row] = Ck1[n-1,n]*kient
            end

            # (n,1,2)
            row = nki2ind(n,k,2)

            kient = 2*Jϕw[k,k] - 1
            X[row,row] = Jzk[n,n]*kient
            col = nki2ind(n+1,k,2)
            X[row,col] = X[col,row] = Jzk[n,n+1]*kient

            kient = 2*Jϕw[k,k+1]
            col = nki2ind(n+1,k+1,3)
            X[row,col] = X[col,row] = Ck1[n,n]*kient
            if n > 1
                col = nki2ind(n,k+1,3)
                X[row,col] = X[col,row] = Ck1[n-1,n]*kient
            end

        end
    end

    for k = 2:nmax
        Jzk = Jzk1
        pzk = pzk1
        Ck = Ck1

        pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
        Jzk1 = jacobimatrix(pzk1)
        Ck1 = pzk1\pzk

        for n = k:nmax

            if mod(k,2) == 0

                # i = 1
                d = deg(k,1)
                row = nki2ind(n,k,1)

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d]
                col = nki2ind(n,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k+1,1)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,1)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 2
                d = deg(k,2)
                row = nki2ind(n,k,2)

                kient = 2*Jw[d+1,d]
                col = nki2ind(n+1,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient
                col = nki2ind(n,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                # i = 3
                d = deg(k,3)
                row = nki2ind(n,k,3)

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n+1,k-1,2)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k+1,3)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end


            else
                # i = 1
                d = deg(k,1)
                row = nki2ind(n,k,1)

                kient = 2*Jw[d+1,d]
                col = nki2ind(n+1,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k+1,2)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,2)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 2
                d = deg(k,2)
                row = nki2ind(n,k,2)

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k+1,3)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 3
                d = deg(k,3)
                row = nki2ind(n,k,3)

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n+1,k-1,3)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

            end

        end

    end

    X

end

function YP(nmax,α₁,β₁,α₂,β₂,b)

    # Construct the operator representing the multiplication of the 3-variable polynomials by y for all polynomials of degree <= nmax
    # α₁ and β₁ are the parameters of the Jacobi weight of the boundary OPs
    # α₂ and β₂ are the parameters of the Jacobi weight of the OPs in z
    # b > 1 is the third root of the cubic ϕ = (1 - x²)(b - x)

    l,u = 1,1          # block bandwidths
    rows = vcat(1,3*(1:nmax+1))
    cols = vcat(1,3*(1:nmax+1))
    Y = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))

    pw = Normalized(SemiclassicalJacobi((b+1)/2, β₁, α₁, 0.0))
    pϕw = Normalized(SemiclassicalJacobi((b+1)/2, β₁+1, α₁+1, 1))
    R = pϕw\pw

    k = 0

    pzk = Normalized(SemiclassicalJacobi((b+1)/2, 2k+1+β₂, α₂, 0.0))
    Jzk = jacobimatrix(pzk)
    pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
    Jzk1 = jacobimatrix(pzk1)
    Ck1 = pzk1\pzk

    con = 2^(3/2)

    for n = k:nmax
        row = nki2ind(n,0,1)
        kient = R[1,1]
        col = nki2ind(n+1,1,2)
        Y[row,col] = Y[col,row] = con*Ck1[n+1,n+1]*kient
        if n > 0
            col = nki2ind(n,1,2)
            Y[row,col] = Y[col,row] = con*Ck1[n,n+1]*kient
        end
    end

    Jzk = Jzk1
    pzk = pzk1
    Ck = Ck1

    k = 1

    pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
    Jzk1 = jacobimatrix(pzk1)
    Ck1 = pzk1\pzk

    for n = k:nmax

        row = nki2ind(n,1,1)

        kient = R[1,2]
        col = nki2ind(n,1,2)
        Y[row,col] = Y[col,row] = con*Jzk[n,n]*kient
        col = nki2ind(n+1,1,2)
        Y[row,col] = Y[col,row] = con*Jzk[n,n+1]*kient

        kient = R[2,2]
        col = nki2ind(n+1,2,3)
        Y[row,col] = Y[col,row] = con*Ck1[n,n]*kient
        if n > 1
            col = nki2ind(n,2,3)
            Y[row,col] = Y[col,row] = con*Ck1[n-1,n]*kient
        end

        row = nki2ind(n,1,2)

        kient = R[1,1]
        col = nki2ind(n+1,0,1)
        Y[row,col] = Y[col,row] = con*Ck[n,n+2]*kient

        kient = R[1,2]
        col = nki2ind(n+1,1,1)
        Y[row,col] = Y[col,row] = con*Jzk[n,n+1]*kient

        kient = R[1,4]
        col = nki2ind(n+1,2,1)
        Y[row,col] = Y[col,row] = con*Ck1[n,n]*kient
        if n > 1
            col = nki2ind(n,2,1)
            Y[row,col] = Y[col,row] = con*Ck1[n-1,n]*kient
        end

        kient = R[1,3]
        col = nki2ind(n+1,2,2)
        Y[row,col] = Y[col,row] = con*Ck1[n,n]*kient
        if n > 1
            col = nki2ind(n,2,2)
            Y[row,col] = Y[col,row] = con*Ck1[n-1,n]*kient
        end

    end

    for k = 2:nmax

        Jzk = Jzk1
        pzk = pzk1
        Ck = Ck1

        pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
        Jzk1 = jacobimatrix(pzk1)
        Ck1 = pzk1\pzk

        for n = k:nmax

            if mod(k,2) == 0

                row = nki2ind(n,k,1)
                d = deg(k,1)

                kient = R[d-2,d+1]
                col = nki2ind(n+1,k-1,2)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d-1,d+1]
                col = nki2ind(n,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k+1,2)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,2)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                kient = R[d,d+1]
                col = nki2ind(n+1,k+1,3)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                row = nki2ind(n,k,2)
                d = deg(k,2)

                kient = R[d-1,d+1]
                col = nki2ind(n+1,k-1,2)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                if k >= 4
                    kient = R[d-2,d+1]
                    col = nki2ind(n+1,k-1,3)
                    Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient
                end

                kient = R[d,d+1]
                col = nki2ind(n,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k+1,3)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                row = nki2ind(n,k,3)
                d = deg(k,3)

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k-1,1)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d+1,d+3]
                col = nki2ind(n+1,k,1)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+2]
                col = nki2ind(n+1,k,2)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+4]
                col = nki2ind(n+1,k+1,1)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,1)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

            else

                row = nki2ind(n,k,1)
                d = deg(k,1)

                kient = R[d-2,d+1]
                col = nki2ind(n+1,k-1,3)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d,d+1]
                col = nki2ind(n,k,2)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d-1,d+1]
                col = nki2ind(n,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k+1,3)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                row = nki2ind(n,k,2)
                d = deg(k,2)

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k-1,1)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d+1,d+2]
                col = nki2ind(n+1,k,1)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+4]
                col = nki2ind(n+1,k+1,1)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,1)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                kient = R[d+1,d+3]
                col = nki2ind(n+1,k+1,2)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,2)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

                row = nki2ind(n,k,3)
                d = deg(k,3)

                kient = R[d+1,d+2]
                col = nki2ind(n+1,k-1,1)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d+1,d+1]
                col = nki2ind(n+1,k-1,2)
                Y[row,col] = Y[col,row] = con*Ck[n-k+1,n-k+3]*kient

                kient = R[d+1,d+3]
                col = nki2ind(n+1,k,1)
                Y[row,col] = Y[col,row] = con*Jzk[n-k+1,n-k+2]*kient

                kient = R[d+1,d+4]
                col = nki2ind(n+1,k+1,2)
                Y[row,col] = Y[col,row] = con*Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,2)
                    Y[row,col] = Y[col,row] = con*Ck1[n-k,n-k+1]*kient
                end

            end
        end

    end


    Y

end

function nki2ind(n,k,i)
    # convert the indices (n,k,i) of a polynomial to a linear index
    if n == 0
        ind = 1
    else
        prev = 1 + 3*(n-1)*n ÷ 2 # number of polys of deg <= n-1
        if k == 0
            ind = prev + 1
        elseif k == 1
            ind = prev + 1 + i
        else
            ind = prev + 3*(k-1) + i
        end
    end
    ind
end

deg(k,i) = 3*(k ÷ 2) + mod(k,2) + 1-i

function Xinds(row)
    # For a given row, find the indices of the leftmost and rightmost nonzero entries of X
    n,k,i = ind2nki(row)
    if k == 0 && i == 1
        if n == 0
            cols = [nki2ind(0,0,1) nki2ind(1,1,1)]
        else
            cols = [nki2ind(n-1,0,1) nki2ind(n+1,1,1)]
        end
    elseif k == 1 && i == 1
        cols = [nki2ind(n-1,0,1) nki2ind(n+1,2,2)]
    elseif k == 1 && i == 2
        if n == 1
            cols = [nki2ind(1,1,2) nki2ind(2,2,3)]
        else
            cols = [nki2ind(n-1,1,2) nki2ind(n+1,2,3)]
        end
    elseif mod(k,2) == 0 && i == 1  # k >= 2, n>= 2
        if k == n # k <= n
            cols = [nki2ind(n,k,1) nki2ind(n+1,k+1,1)]
        else
            cols = [nki2ind(n-1,k,1) nki2ind(n+1,k+1,1)]
        end
    elseif mod(k,2) == 0 && i == 2
        cols = [nki2ind(n-1,k-1,1) nki2ind(n+1,k,2)]
    elseif mod(k,2) == 0 && i == 3
        cols = [nki2ind(n-1,k-1,2) nki2ind(n+1,k+1,3)]
    elseif mod(k,2) == 1 && i == 1 # k >= 3, n>= 3
        cols = [nki2ind(n-1,k-1,1) nki2ind(n+1,k+1,2)]
    elseif mod(k,2) == 1 && i == 2
        if k == n
            cols = [nki2ind(n,k,2) nki2ind(n+1,k+1,3)]
        else
            cols = [nki2ind(n-1,k,2) nki2ind(n+1,k+1,3)]
        end
    elseif mod(k,2) == 1 && i == 3
        cols = [nki2ind(n-1,k-1,3) nki2ind(n+1,k,3)]
    end

    cols
end

function Yinds(row)
    # For a given row, find the indices of the leftmost and rightmost nonzero entries of Y
    n,k,i = ind2nki(row)
    if k == 0 && i == 1
        if n == 0
            cols = [nki2ind(1,1,2) nki2ind(1,1,2)]
        elseif n == 1
            cols = [nki2ind(1,1,2) nki2ind(2,1,2)]
        else
            cols = [nki2ind(n-1,1,2) nki2ind(n+1,1,2)]
        end
    elseif k == 1 && i == 1
        if n == 1
            cols = [nki2ind(1,1,2) nki2ind(2,2,3)]
        else
            cols = [nki2ind(n-1,1,2) nki2ind(n+1,2,3)]
        end
    elseif k == 1 && i == 2
        cols = [nki2ind(n-1,0,1) nki2ind(n+1,2,2)]
    elseif mod(k,2) == 0 && i == 1  # k >= 2, n>= 2
        cols = [nki2ind(n-1,k-1,2) nki2ind(n+1,k+1,3)]
    elseif mod(k,2) == 0 && i == 2
        cols = [nki2ind(n-1,k-1,2) nki2ind(n+1,k+1,3)]
    elseif mod(k,2) == 0 && i == 3
        cols = [nki2ind(n-1,k-1,1) nki2ind(n+1,k+1,1)]
    elseif mod(k,2) == 1 && i == 1 # k >= 3, n>= 3
        cols = [nki2ind(n-1,k-1,3) nki2ind(n+1,k+1,3)]
    elseif mod(k,2) == 1 && i == 2
        cols = [nki2ind(n-1,k-1,1) nki2ind(n+1,k+1,2)]
    elseif mod(k,2) == 1 && i == 3
        cols = [nki2ind(n-1,k-1,1) nki2ind(n+1,k+1,2)]
    end

    cols
end

function Xminds(rows)
    cols = Xinds.(rows)
    M = zeros(Int64,length(rows),2)
    for r = 1:length(rows)
       M[r,:] = cols[r]
    end
    mini = minimum(M[:,1])
    maxi = maximum(M[:,2])
    mini, maxi
end

function Yminds(rows)
    cols = Yinds.(rows)
    M = zeros(Int64,length(rows),2)
    for r = 1:length(rows)
       M[r,:] = cols[r]
    end
    mini = minimum(M[:,1])
    maxi = maximum(M[:,2])
    mini, maxi
end

function nk2ind(n,k)
    # Convert the indices (n,k) of a polynomial to a linear index/its index in the vector of polynomials
    if k == n
        out = Int64((n+1)*(n+2)/2)
    else
        out = Int64(n*(n+1)/2)+k+1
    end
    out
end

function ind2nki(ind)
    # convert a linear index to the indices (n,k,i) of the polynomials
    if ind == 1
        out = [0,0,1]
    elseif ind == 2
        out = [1,0,1]
    elseif ind == 3
        out = [1,1,1]
    elseif ind == 4
        out = [1,1,2]
    else
        n = (1 + sqrt(1 + 4/3*(2*ind - 2)))/2
        if mod(n,1) == 0
            out = Int64.([n-1,n-1,3])
        else
            n = Int64(floor(n))
            rem = ind - Int64(3n*(n-1)/2 + 1)
            if rem == 1
                out = [n,0,1]
            elseif rem == 2
                out = [n,1,1]
            elseif rem == 3
                out = [n,1,2]
            else
                k = Int64(floor((rem-1)/3)) +1
                i = Int64(mod(rem-1,3))+1
                out = [n,k,i]
            end
        end
    end
    out
end
