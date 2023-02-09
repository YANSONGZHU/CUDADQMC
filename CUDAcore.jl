include("CUDAqrudt.jl")
using CUDA
using Adapt
using LinearAlgebra
using Test

struct lattice
    L::Int
    Ns::Int
    U::Float32
    μ::Float32
    Temp::Float32
    Nt::Int
    Δτ::Float32
    λ::Float32
    Tmatrix::Matrix{Int}
    expmΔτT::CUDAUDT{Float32}

    function lattice(L::Int,U::Float32,μ::Float32,Temp::Float32,Nt::Int)
        Ns = L^3
        Δτ = 1/(Temp*Nt)
        λ = acosh(exp(abs(U)*Δτ/2))
        Tmatrix = adapt(CuArray,initT(L))
        expmΔτT = CUDAudt(exp(-Δτ * initT(L)))
        new(L,Ns,U,μ,Temp,Nt,Δτ,λ,Tmatrix,expmΔτT)
    end
end

function initT(L::Int)
    Tmatrix = zeros(Int,L^3,L^3)
    index::Int = 1
    for z = 1:L, y = 1:L, x = 1:L
        Tmatrix[index, x == 1 ? index+L-1 : index-1] = -1
        Tmatrix[index, x == L ? index-L+1 : index+1] = -1
        Tmatrix[index, y == 1 ? index+L*(L-1) : index-L] = -1
        Tmatrix[index, y == L ? index-L*(L-1) : index+L] = -1
        Tmatrix[index, z == 1 ? index+L^2*(L-1) : index-L^2] = -1
        Tmatrix[index, z == L ? index-L^2*(L-1) : index+L^2] = -1
        index += 1
    end
    Tmatrix
end

function CUDAeTeV(eT::CUDAUDT,eV::CuVector{Float32})
    inveV = 1 ./ eV
    CUDAUDT(eT.U,eT.D .* eV,Diagonal(inveV) * eT.T * Diagonal(eV))
end


function initMultBudt(l::lattice,AuxField::Matrix{Int})
    MultBup = Vector{CUDAUDT}(undef,l.Nt+2)
    MultBdn = Vector{CUDAUDT}(undef,l.Nt+2)
    UDTI = CUDAudt(CuMatrix(Diagonal(ones(Float32,l.Ns))))
    MultBup[1] = copy(UDTI)
    MultBdn[1] = copy(UDTI)
    MultBup[l.Nt+2] = copy(UDTI)
    MultBdn[l.Nt+2] = copy(UDTI)
    for i = 2:l.Nt+1
        MultBup[i] = CUDAudtMult(MultBup[i-1],CUDAeTeV(l.expmΔτT,exp.(cu( AuxField[:,i-1])*l.λ .+ (l.μ - l.U/2)*l.Δτ)))
        MultBdn[i] = CUDAudtMult(MultBdn[i-1],CUDAeTeV(l.expmΔτT,exp.(cu(-AuxField[:,i-1])*l.λ .+ (l.μ - l.U/2)*l.Δτ)))
    end
    MultBup, MultBdn
end

function flipslice!(slice::Int,l::lattice,AuxField::Matrix{Int},Gup::CuArray{Float32},Gdn::CuArray{Float32})
    γup = exp.(-2*l.λ*AuxField[:,slice]).-1
    γdn = exp.( 2*l.λ*AuxField[:,slice]).-1
    Rup = 0
    Rdn = 0
    @inbounds for site = 1:l.Ns
        CUDA.@allowscalar Rup = 1+(1-(Gup[site,site]))*γup[site]
        CUDA.@allowscalar Rdn = 1+(1-(Gdn[site,site]))*γdn[site]
        P = Rup * Rdn
        if P > 1 || rand() < P
            AuxField[site,slice] *= -1
            updateg!(site,γup[site]/Rup,Gup)
            updateg!(site,γdn[site]/Rdn,Gdn)
        end
    end
    nothing
end

function updateg!(site::Int,prop::Float32,g::CuArray{Float32})
    gtmp = -g[site,:]
    CUDA.@allowscalar gtmp[site] += 1
    @views g[:,:] -= - prop * g[:,site] * transpose(gtmp)
    nothing
end

function updateRight!(slice::Int,l::lattice,AuxField::Matrix{Int},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT},gup::CuArray{Float32},gdn::CuArray{Float32})
    MultBup[l.Nt-slice+2] = CUDAudtMult(CUDAeTeV(l.expmΔτT,
        exp.(cu( AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ)), MultBup[l.Nt-slice+3])
    greens!(MultBup[l.Nt-slice+2],MultBup[l.Nt-slice+1],gup)
    MultBdn[l.Nt-slice+2] = CUDAudtMult(CUDAeTeV(l.expmΔτT,
        exp.(cu(-AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ)), MultBdn[l.Nt-slice+3])
    greens!(MultBdn[l.Nt-slice+2],MultBdn[l.Nt-slice+1],gdn)
    nothing
end

function updateLeft!(slice::Int,l::lattice,AuxField::Matrix{Int},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT},gup::CuArray{Float32},gdn::CuArray{Float32})
    MultBup[l.Nt-slice+2] = CUDAudtMult(MultBup[l.Nt-slice+1],
        CUDAeTeV(l.expmΔτT, exp.(cu( AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ)))
    greens!(MultBup[l.Nt-slice+2],MultBup[l.Nt-slice+3],gup)
    MultBdn[l.Nt-slice+2] = CUDAudtMult(MultBdn[l.Nt-slice+1],
        CUDAeTeV(l.expmΔτT, exp.(cu(-AuxField[:,slice])*l.λ .+ (l.μ - l.U/2)*l.Δτ)))
    greens!(MultBdn[l.Nt-slice+2],MultBdn[l.Nt-slice+3],gdn)
    nothing
end

function sweep!(l::lattice,AuxField::Matrix{Int},Gup::CuArray{Float32},Gdn::CuArray{Float32},
    MultBup::Vector{CUDAUDT},MultBdn::Vector{CUDAUDT})
    for slice = 1:l.Nt-1
        flipslice!(slice,l,AuxField,Gup,Gdn)
        updateRight!(slice,l,AuxField,MultBup,MultBdn,Gup,Gdn)
    end
    for slice = l.Nt:-1:2
        flipslice!(slice,l,AuxField,Gup,Gdn)
        updateLeft!(slice,l,AuxField,MultBup,MultBdn,Gup,Gdn)
    end
end