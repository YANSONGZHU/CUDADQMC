using CUDA
using Adapt
using LinearAlgebra

mutable struct CUDAUDT{Type <: Real}
	U::CuMatrix{Type}
	D::CuVector{Type}
	T::CuMatrix{Type}
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
    CUDAUDT(CuMatrix(U), D, Diagonal(1 ./ D) * R)
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

function greens!(A::CUDAUDT{Float32},B::CUDAUDT{Float32},G::CuMatrix{Float32})
    CuMat = A.T * B.U
    lmul!(Diagonal(A.D), CuMat)
    rmul!(CuMat, Diagonal(B.D))
    F = CUDAudt(CuMat)
    m = F.U' / F.T
    m[diagind(m)] .+= F.D
    utmp, d, ttmp = CUDAudt(m)
    u = similar(A.U)
    t = similar(A.T)
    mul!(u, F.U, utmp)
    mul!(t, ttmp, F.T)
    G[:,:] =  inv(t)*Diagonal(1 ./ d)*u'
end