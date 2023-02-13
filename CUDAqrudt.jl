using CUDA
using Adapt
using LinearAlgebra

mutable struct CUDAUDT{Float32}
	U::CuArray{Float32}
	D::CuArray{Float32}
	T::CuArray{Float32}
end

# iteration for destructuring into components
Base.iterate(S::CUDAUDT) = (S.U, Val(:D))
Base.iterate(S::CUDAUDT, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::CUDAUDT, ::Val{:T}) = (S.T, Val(:done))
Base.copy(S::CUDAUDT) = CUDAUDT(S.U, S.D, S.T)

function CUDAudt(A::Matrix{Float32})
    CuA = adapt(CuArray,A)
    U, R = qr(CuA)
    D = diag(R)
    CUDAUDT(U, D, Diagonal(1 ./ D) * R)
end

function CUDAudt(CuA::CuMatrix{Float32})
    U, R = qr(CuA)
    D = diag(R)
    CUDAUDT(CuMatrix(U), D, Diagonal(1 ./ D) * R)
end

function CUDAudtMult(A::CUDAUDT{Float32},B::CUDAUDT{Float32})
    CuMat = A.T * B.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(B.D))
    F = CUDAudt(CuMat)
    CUDAUDT(A.U * F.U, F.D, F.T * B.T)
end

function CUDArmuleTeV(A::CUDAUDT{Float32},eT::CUDAUDT{Float32},eV::CuVector{Float32})
    inveV = 1 ./ eV
    CuMat = A.T * eT.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(eT.D .* eV))
    F = CUDAudt(CuMat)
    CUDAUDT(A.U * F.U, F.D, F.T * Diagonal(inveV) * eT.T * Diagonal(eV))
end

function CUDArmuleTeV!(A::CUDAUDT{Float32},eT::CUDAUDT{Float32},eV::CuVector{Float32},MultB::CUDAUDT{Float32})
    inveV = 1 ./ eV
    CuMat = A.T * eT.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(eT.D .* eV))
    MultB.U, MultB.D, MultB.T = CUDAudt(CuMat)
    MultB.U = A.U * MultB.U
    MultB.T = MultB.T * Diagonal(inveV) * eT.T * Diagonal(eV)
    nothing
end

function CUDAlmuleTeV(eT::CUDAUDT{Float32},eV::CuVector{Float32},A::CUDAUDT{Float32})
    inveV = 1 ./ eV
    CuMat = Diagonal(inveV) * eT.T * Diagonal(eV) * A.U
    lmul!(Diagonal(eT.D .* eV), CuMat)
    rmul!(CuMat, Diagonal(A.D))
    F = CUDAudt(CuMat)
    CUDAUDT(eT.U * F.U, F.D, F.T * A.T)
end

function cuinv!(A::CuMatrix{Float32})
    cuItmp = CuArray(Matrix{Float32}(I(size(A,1))))
    tmp, ipiv = CUDA.CUSOLVER.getrf!(A)
    return CUDA.CUSOLVER.getrs!('N', tmp, ipiv, cuItmp)
end

function greens!(A::CUDAUDT{Float32},B::CUDAUDT{Float32},G::CuMatrix{Float32})
    CuMat = A.T * B.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(B.D))
    F = CUDAudt(CuMat)
    invXinvDp = cuinv!(F.T*B.T) * Diagonal(1 ./ max.(F.D,1))
    Dm = Diagonal(min.(F.D,1))
    tmp = invXinvDp+A.U*F.U*Dm
    G[:,:] = invXinvDp * cuinv!(tmp)
end