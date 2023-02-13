function occupy(gup::CuArray{Float32},gdn::CuArray{Float32})
    Ns = size(gup,1)
    occupy = 2 - (sum(diag(gup)) + sum(diag(gdn)))/Ns
end

function doubleoccupy(gup::CuArray{Float32},gdn::CuArray{Float32})
    Ns = size(gup,1)
    doubleoccupy = sum((1 .- diag(gup)).*(1 .- diag(gdn)))/Ns
end

function kinetic(gup::CuArray{Float32},gdn::CuArray{Float32},Tmatrix::Matrix{Int})
    Ns = size(gup,1)
    gup = Matrix(gup)
    gdn = Matrix(gdn)
    gtildeup = Diagonal(Vector(ones(Ns))) - transpose(gup)
    gtildedn = Diagonal(Vector(ones(Ns))) - transpose(gdn)
    sumk = sum(gtildeup .* Tmatrix) + sum(gtildedn .* Tmatrix)
    sumk / (2 * Ns)
end

function energy(gup::CuArray{Float32},gdn::CuArray{Float32},l::lattice)
    Ek = kinetic(gup,gdn,l.Tmatrix)
    EU = 0.25f0 * l.U *doubleoccupy(gup,gdn)
    Ek - EU
end

function index2xyz(index::Int, L::Int)
	n = index - 1
	xyz = zeros(Int,3)
	for i = 1:3
		xyz[i] = n % L
		n = n ÷ L
	end
	xyz .+ 1
end

function Sπ(CuGup::CuArray{Float32},CuGdn::CuArray{Float32},l::lattice)
    Sπ = 0
    gup = Matrix(CuGup)
    gdn = Matrix(CuGdn)
    gtildeup = Diagonal(Vector(ones(l.Ns))) - transpose(gup)
    gtildedn = Diagonal(Vector(ones(l.Ns))) - transpose(gdn)
    Q = [pi, pi, pi]
    for i = 1:l.Ns
        for j = 1:l.Ns
            c = real(exp(im*sum(Q.*(index2xyz(i,l.L)-index2xyz(j,l.L)))))
            Sπ += c * (gtildeup[i,i] * gtildeup[j,j] + gtildeup[i,j] * gup[i,j] +
                gtildedn[i,i] * gtildedn[j,j] + gtildedn[i,j] * gdn[i,j] -
                gtildedn[i,i] * gtildeup[j,j] - gtildeup[i,i] * gtildedn[j,j])
        end
    end
    Sπ / l.Ns
end