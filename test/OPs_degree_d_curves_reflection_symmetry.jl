function LanczosSymCurve(Jp,Jpϕ,Cϕ,nmax,d)

    # OPs on the curve y² = ϕ(x), where ϕ has degree d ≥ 3

    tol = 1E-8
    f = x -> abs(x)>tol
    l,u = 1,1
    rows = cols = vcat(1:d-1,fill(d,nmax-d+2))
    Jx = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))
    Jy = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))
    r,c = size(Jx)
    X = XP(Jp,Jpϕ,nmax+d-1,d) #?
    Y = YP(Cϕ,nmax+d-1,d)
    C = zeros(r,2r)

    # First construct OPs of degree 1, …, d-1 (first multiply by x and orthogonalize, then multiply by y and orthogonalize)

    C[1:5,1:5] = I(5)
    Cinds = zeros(Int,r,2)
    Cinds[1:5,:] = [collect(1:5) collect(1:5)]
    # orthogonalize degree 0 to obtain degree 1 OPs
    Jx[Block(1,1)] = [X[1,1]]
    Jx[Block(1,2)] = X[Block(1,2)]
    Jx[Block(2,1)] = X[Block(2,1)]
    Jy[Block(1,2)] = Y[Block(1,2)]
    Jy[Block(2,1)] = Y[Block(2,1)]
    Jx[Block(2,2)] = X[Block(2,2)]
    Jy[Block(2,2)] = Y[Block(2,2)]

    # orthogonalize degree 1 OPs to obtain degree 2 OPs
    B1x = [X[Block(2,3)][1,1] 0 0;0 X[Block(2,3)][2,2] 0]
    Jx[Block(2,3)] = B1x
    Jx[Block(3,2)] = B1x'
    yQ11 = Y[qnk2ind(1,1,d),1:pnk2ind(d,0)]'
    B1y = zeros(2,3)
    B1y[1,2] = Y[Block(2,3)][1,2]
    B1y[2,1] = Y[Block(2,3)][2,1]
    yQ11[qnk2ind(0,0,d)] -= Jy[Block(2,1)][2,1]
    yQ11[qnk2ind(1,0,d)] -= Jy[Block(2,2)][2,1]
    yQ11[qnk2ind(2,0,d)] -= B1y[2,1]
    B1y[2,3] = norm(yQ11)
    cr =  yQ11/B1y[2,3]
    C[qnk2ind(2,2,d),1:pnk2ind(d,0)] = cr
    Jy[Block(2,3)] = B1y
    Jy[Block(3,2)] = B1y'
    Cinds[qnk2ind(2,2,d),:] = [findfirst(f,cr)[2],findlast(f,cr)[2]]
    A2x = zeros(3,3)
    A2x[1,1] = X[Block(3,3)][1,1]
    cols = Cinds[qnk2ind(2,2,d),1]:Cinds[qnk2ind(2,2,d),2]
    A2x[1,3] = A2x[3,1] = dot(X[pnk2ind(2,0),cols],C[qnk2ind(2,2,d),cols])
    A2x[2,2] = X[Block(3,3)][2,2]
    A2x[3,3] = C[qnk2ind(2,2,d),cols]'*X[cols,cols]*C[qnk2ind(2,2,d),cols]
    A2y = zeros(3,3)
    A2y[1,2] = A2y[2,1] = Y[Block(3,3)][1,2]
    A2y[2,3] = A2y[3,2] = dot(Y[pnk2ind(2,1),cols],C[qnk2ind(2,2,d),cols])
    Jx[Block(3,3)] = A2x
    Jy[Block(3,3)] = A2y

    # orthogonalize degree n OPs to obtain degree n+1 OPs for n = 2, …, d-2
    for n = 2:d-2
        # Orthogonalize xQn to form Qₙ₊₁,₀, …, Qₙ₊₁,ₙ
        Bx = zeros(n+1,n+2)
        for k = 0:n
            r = qnk2ind(n,k,d)
            # Multiply  by x
            right = Cinds[qnk2ind(n,n,d),2] + 2  # (check!)
            xQn = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],1:right]
            # Subtract off components along lower degree OPs
            # first for OPs of degree n-1
            if k < n
                for c = k+1:2:n
                    l = qnk2ind(n-1,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    xQn[cols] -= Jx[Block(n+1,n)][k+1,c]*C[l,cols]
                end
            end
            # next for OPs of degree n
            if mod(k,2)==0
                for c = 1:2:n+1
                    l = qnk2ind(n,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    xQn[cols] -= Jx[Block(n+1,n+1)][k+1,c]*C[l,cols]
                end
            else
                for c = 2:2:n+1
                    l = qnk2ind(n,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    xQn[cols] -= Jx[Block(n+1,n+1)][k+1,c]*C[l,cols]
                end
            end
            # Finally OPs of degree n+1
            if k > 1
                if mod(k,2) == 0
                    for c = 1:2:k-1
                        l = qnk2ind(n+1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        xQn[cols] -= Bx[k+1,c]*C[l,cols]
                    end
                else
                    for c = 2:2:k-1
                        l = qnk2ind(n+1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        xQn[cols] -= Bx[k+1,c]*C[l,cols]
                    end
                end
            end
            # Normalise
            Bx[k+1,k+1] = norm(xQn)
            cr = xQn/Bx[k+1,k+1]
            C[qnk2ind(n+1,k,d),1:right] = cr
            lind = findfirst(f,cr)
            rind = findlast(f,cr)
            Cinds[qnk2ind(n+1,k,d),:] = [lind[2] rind[2]]

            # Compute the entries in column k+1 of Bx:

            if k < n-1
                for row = k+3:2:n+1
                    ind = qnk2ind(n,row-1,d)
                    Bx[row,k+1] = C[ind,Cinds[ind,1]:Cinds[ind,2]]'*X[Cinds[ind,1]:Cinds[ind,2],lind[2]:rind[2]]*C[qnk2ind(n+1,k,d),lind[2]:rind[2]]
                end
            end
        end
        Jx[Block(n+1,n+2)] = Bx
        Jx[Block(n+2,n+1)] = Bx'

        # Compute By column by column
        By = zeros(n+1,n+2)
        ind = qnk2ind(n+1,0,d)
        # first column
        for r = 2:2:n+1
            l = qnk2ind(n,r-1,d)
            By[r,1] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
        end
        # rest of the columns
        for c = 2:n+1
            ind = qnk2ind(n+1,c-1,d)
            for r = c-1:2:n+1
                l = qnk2ind(n,r-1,d)
                By[r,c] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
            end
        end
        # orthogonalize yQ_{n,n} to find Q_{n+1,n+1}
        # Multiply Q_{n,n} by y
        r = qnk2ind(n,n,d)
        if mod(n,2) == 0
            right = Cinds[r,2] + 3
        else
            right = Cinds[r,2] + 2d-3
        end
        yQn = C[r,Cinds[r,1]:Cinds[r,2]]'*Y[Cinds[r,1]:Cinds[r,2],1:right]
        # subtract off OPs of degree n-1
        l = qnk2ind(n-1,n-1,d)
        cols = Cinds[l,1]:Cinds[l,2]
        yQn[cols] -= Jy[Block(n+1,n)][n+1,n]*C[l,cols]
        # subtract off OPs of degree n
        if mod(n+1,2)==0
            for c = 1:2:n+1
                l = qnk2ind(n,c-1,d)
                cols = Cinds[l,1]:Cinds[l,2]
                yQn[cols] -= Jy[Block(n+1,n+1)][n+1,c]*C[l,cols]
            end
        else
            for c = 2:2:n+1
                l = qnk2ind(n,c-1,d)
                cols = Cinds[l,1]:Cinds[l,2]
                yQn[cols] -= Jy[Block(n+1,n+1)][n+1,c]*C[l,cols]
            end
        end
        # subtract off OPs of degree n+1
        if mod(n+1,2)==0
            for c = 1:2:n
                l = qnk2ind(n+1,c-1,d)
                cols = Cinds[l,1]:Cinds[l,2]
                yQn[cols] -= By[n+1,c]*C[l,cols]
            end
        else
            for c = 2:2:n
                l = qnk2ind(n+1,c-1,d)
                cols = Cinds[l,1]:Cinds[l,2]
                yQn[cols] -= By[n+1,c]*C[l,cols]
            end
        end

        # normalise
        By[n+1,n+2] = norm(yQn)
        cr = yQn/By[n+1,n+2]
        C[qnk2ind(n+1,n+1,d),1:right] = cr
        lind = findfirst(f,cr)
        rind = findlast(f,cr)
        Cinds[qnk2ind(n+1,n+1,d),:] = [lind[2] rind[2]]
        Jy[Block(n+1,n+2)] = By
        Jy[Block(n+2,n+1)] = By'

        Ax = zeros(n+2,n+2)
        # compute the diagonal elements
        for k = 0:n+1
            r = qnk2ind(n+1,k,d)
            cols = Cinds[r,1]:Cinds[r,2]
            Ax[k+1,k+1] = C[r,cols]'*X[cols,cols]*C[r,cols]
        end
        # off-diagonal elements
        for r = 1:n
            row = qnk2ind(n+1,r-1,d)
            for c = r+2:2:n+2
                Ax[r,c] = Ax[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*X[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end

        Jx[Block(n+2,n+2)] = Ax

        Ay = zeros(n+2,n+2)
        for r = 1:n+1
            row = qnk2ind(n+1,r-1,d)
            for c = r+1:2:n+2
                Ay[r,c] = Ay[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*Y[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end
        Jy[Block(n+2,n+2)] = Ay;
    end

    # Now construct OPs of degrees d, d+1, …, nmax by multiplying by y and then orthogonalizing
    for n = d-1:nmax-1
    #for n = d-1:d-1-1
        # Orthogonalize yQn to form Qₙ₊₁,₀, …, Qₙ₊₁,d-1
        By = zeros(d,d)
        for k = 0:d-1
            r = qnk2ind(n,k,d)
            # Multiply  by y
            if mod(n-d+1+k,2) == 0
                right = Cinds[qnk2ind(n,d-1,d),2] + 3
            else
                right = Cinds[qnk2ind(n,d-1,d),2] + 2d-3
            end
            yQn = C[r,Cinds[r,1]:Cinds[r,2]]'*Y[Cinds[r,1]:Cinds[r,2],1:right]
            # Subtract off components along lower degree OPs
            # first for OPs of degree n-1
            if n == d-1
                if k == 0
                    for c = 2:2:d-1
                        l = qnk2ind(n-1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        yQn[cols] -= Jy[Block(n+1,n)][k+1,c]*C[l,cols]
                    end
                else
                    for c = k:2:d-1
                        l = qnk2ind(n-1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        yQn[cols] -= Jy[Block(n+1,n)][k+1,c]*C[l,cols]
                    end
                end
            else
                for c = k+1:2:d
                    l = qnk2ind(n-1,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    yQn[cols] -= Jy[Block(n+1,n)][k+1,c]*C[l,cols]
                end
            end
            # next for OPs of degree n
            if mod(k,2)==0
                for c = 2:2:d
                    l = qnk2ind(n,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    yQn[cols] -= Jy[Block(n+1,n+1)][k+1,c]*C[l,cols]
                end
            else
                for c = 1:2:d
                    l = qnk2ind(n,c-1,d)
                    cols = Cinds[l,1]:Cinds[l,2]
                    yQn[cols] -= Jy[Block(n+1,n+1)][k+1,c]*C[l,cols]
                end
            end
            # Finally OPs of degree n+1
            if k > 1
                if mod(k,2) == 0
                    for c = 1:2:k-1
                        l = qnk2ind(n+1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        yQn[cols] -= By[k+1,c]*C[l,cols]
                    end
                else
                    for c = 2:2:k-1
                        l = qnk2ind(n+1,c-1,d)
                        cols = Cinds[l,1]:Cinds[l,2]
                        yQn[cols] -= By[k+1,c]*C[l,cols]
                    end
                end
            end

            # Normalise
            By[k+1,k+1] = norm(yQn)
            cr = yQn/By[k+1,k+1]
            C[qnk2ind(n+1,k,d),1:right] = cr
            lind = findfirst(f,cr)
            rind = findlast(f,cr)
            Cinds[qnk2ind(n+1,k,d),:] = [lind[2] rind[2]]

            # Compute the entries in column k+1 of By:

            if k < d-2
                for row = k+3:2:d
                    ind = qnk2ind(n,row-1,d)
                    By[row,k+1] = C[ind,Cinds[ind,1]:Cinds[ind,2]]'*Y[Cinds[ind,1]:Cinds[ind,2],lind[2]:rind[2]]*C[qnk2ind(n+1,k,d),lind[2]:rind[2]]
                end
            end
        end

        Jy[Block(n+1,n+2)] = By
        Jy[Block(n+2,n+1)] = By'

        # Compute Bx column by column
        Bx = zeros(d,d)
        for c = 1:d
            ind = qnk2ind(n+1,c-1,d)
            if mod(c,2) == 1
                for r = 2:2:d
                    l = qnk2ind(n,r-1,d)
                    Bx[r,c] = C[l,Cinds[l,1]:Cinds[l,2]]'*X[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
                end
            else
                for r = 1:2:d
                    l = qnk2ind(n,r-1,d)
                    Bx[r,c] = C[l,Cinds[l,1]:Cinds[l,2]]'*X[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
                end
            end
        end

        Jx[Block(n+1,n+2)] = Bx
        Jx[Block(n+2,n+1)] = Bx'

        Ax = zeros(d,d)
        # compute the diagonal elements
        for k = 0:d-1
            r = qnk2ind(n+1,k,d)
            cols = Cinds[r,1]:Cinds[r,2]
            Ax[k+1,k+1] = C[r,cols]'*X[cols,cols]*C[r,cols]
        end
        # off-diagonal elements
        for r = 1:d-2
            row = qnk2ind(n+1,r-1,d)
            for c = r+2:2:d
                Ax[r,c] = Ax[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*X[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end

        Jx[Block(n+2,n+2)] = Ax

        Ay = zeros(d,d)
        for r = 1:d-1
            row = qnk2ind(n+1,r-1,d)
            for c = r+1:2:d
                Ay[r,c] = Ay[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*Y[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end
        Jy[Block(n+2,n+2)] = Ay;

    end

    C, Jx, Jy, Cinds

end

function qnk2ind(n,k,d)
    if n <=  d-1
        ind = (n+1)*n ÷ 2 + k + 1
    else
        ind = (d+1)*d ÷ 2 + (n-d)*d + k + 1
    end
    ind
end

function pnk2ind(n,k)
    if n == 0
        ind = 1
    elseif k == 0
        ind = 2n
    else
        ind = 2n + 1
    end
    ind
end

function YP(Cϕ,n,d)
    # Construct the matrix representing multiplication of P_{k,0} = p_k(w) and P_{k,1} = yp_{k-1}(ϕw) by y, 0 <= k <= nmax
    # Cϕ = Pϕ\P, where P is a quasimatrix of OPs wrt w and Pϕ wrt ϕw
    # nmax is determined by the dimension of the space of OPs on y² = ϕ(x) of degree <= n, where ϕ is a polynomial of degree d

    dim = qnk2ind(n,d-1,d) # number of OPs on y² = ϕ(x) of degree <= n
    # Number of rows of the output matrix is 2nmax + 1 >= dim
    if mod(dim,2) == 0
        nmax = Int64(dim/2)
    else
        nmax = Int64((dim-1)/2)
    end

    l,u = d-1,d-1
    rows = vcat(1,fill(2,nmax))
    cols = vcat(1,fill(2,nmax-1+d))
    Y = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))

    r = pnk2ind(0,0)
    c = pnk2ind(1,1)
    Y[r,c] = Y[c,r] = Cϕ[1,1]

    for k = 1:nmax
        r = pnk2ind(k,0)
        c = pnk2ind(k,1)
        if c <= 2nmax+1
            Y[r,c] = Y[c,r] = Cϕ[k,k+1]
        else
            Y[r,c] = Cϕ[k,k+1]
        end
        c = pnk2ind(k+1,1)
        if c <= 2nmax+1
            Y[r,c] = Y[c,r] = Cϕ[k+1,k+1]
        else
            Y[r,c] = Cϕ[k+1,k+1]
        end

        r = pnk2ind(k,1)
        for m = 1:d-1
            c = pnk2ind(k+m,0)
            if c <= 2nmax+1
                Y[r,c] = Y[c,r] = Cϕ[k,k+1+m]
            else
                Y[r,c] = Cϕ[k,k+1+m]
            end
        end

    end
    Y
end

function XP(Jp,Jpϕ,n,d)
    # Construct the matrix representing multiplication of P_{k,0} = p_k(w) and P_{k,1} = yp_{k-1}(ϕw) by x, 0 <= k <= nmax
    # Jp and Jpϕ are the Jacobi operators of the OP families p_k(w) and p_{k-1}(ϕw)
    # nmax is determined by the dimension of the space of OPs on y² = ϕ(x) of degree <= n, where ϕ is a polynomial of degree d

    dim = qnk2ind(n,d-1,d) # number of OPs on y² = ϕ(x) of degree <= n
    # Number of rows of the output matrix is 2nmax + 1 >= dim
    if mod(dim,2) == 0
        nmax = Int64(dim/2)
    else
        nmax = Int64((dim-1)/2)
    end

    l,u = 1,1
    rows = vcat(1,fill(2,nmax))
    cols = vcat(1,fill(2,nmax+1))
    X = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))

    r = pnk2ind(0,0)
    X[r,r] = Jp[1,1]
    c = pnk2ind(1,0)
    X[r,c] = X[c,r] = Jp[1,2]

    for k = 1:nmax
        r = pnk2ind(k,0)
        X[r,r] = Jp[k+1,k+1]
        c = pnk2ind(k+1,0)
        if k < nmax
            X[r,c] = X[c,r] = Jp[k+1,k+2]
        else
            X[r,c] = Jp[k+1,k+2]
        end
        r = pnk2ind(k,1)
        X[r,r] = Jpϕ[k,k]
        c = pnk2ind(k+1,1)
        if k < nmax
            X[r,c] = X[c,r] = Jpϕ[k,k+1]
        else
            X[r,c] = Jpϕ[k,k+1]
        end
    end
    X
end
