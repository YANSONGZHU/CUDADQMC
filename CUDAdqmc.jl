include("CUDAcore.jl")
include("measure.jl")

using ProgressBars

L = 10
U = 8.0f0
μ = 4.0f0
Temp = 0.5f0
Nt = 40
l = lattice(L,U,μ,Temp,Nt)
AuxField = rand([-1,1],l.Ns,Nt)
MultBup, MultBdn = initMultBudt(l,AuxField)
Gup = cu(zeros(Float32,L^3,L^3))
Gdn = cu(zeros(Float32,L^3,L^3))
Gup = greens!(MultBup[Nt+1],MultBup[Nt+2],Gup)
Gdn = greens!(MultBdn[Nt+1],MultBdn[Nt+2],Gdn)
Nsweep = 100
for sweep = 1:Nsweep
    @time sweep!(l,AuxField,Gup,Gdn,MultBup,MultBdn)
end