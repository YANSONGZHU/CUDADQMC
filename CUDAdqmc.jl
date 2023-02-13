include("CUDAcore.jl")
include("measure.jl")

using ProgressBars
using DelimitedFiles

L = 10
U = 8.0f0
μ = 4.0f0
Temp = 0.5f0
Nt = 40
l = lattice(L,U,Temp,Nt)
resultfile = "U8 T0.5 Spi.dat"
AuxField = rand([-1,1],l.Ns,Nt)
MultBup, MultBdn = initMultBudt(l,AuxField)
Gup = cu(zeros(Float32,L^3,L^3))
Gdn = cu(zeros(Float32,L^3,L^3))
Gup = greens!(MultBup[Nt+1],MultBup[Nt+2],Gup)
Gdn = greens!(MultBdn[Nt+1],MultBdn[Nt+2],Gdn)
Nsweep = 10000
SπDATA = zeros(Float32,10000,3)
for sweep = ProgressBar(1:Nsweep)
    sweep!(l,AuxField,Gup,Gdn,MultBup,MultBdn)
    SπDATA[sweep,1] = occupy(Gup,Gdn)
    SπDATA[sweep,2] = doubleoccupy(Gup,Gdn)
    SπDATA[sweep,3] = Sπ(Gup,Gdn,l)
end
io = open(resultfile, "w")
writedlm(io, SπDATA, '\t')
close(io)