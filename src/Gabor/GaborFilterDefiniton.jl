function gaborfilter(n, λ, θ, ψ, σ, γ)
    w = zeros(n, n)
    for x in 1:n
        for y in 1:n
            x̂ = x*cos(θ) + y*sin(θ)
            ŷ = -x*sin(θ) + y*cos(θ)
            w[x, y] =  exp((x̂^2 + (γ^2)*(ŷ^2))/(σ^2) * cos(2*π)*(x̂/λ + ψ))
        end
    end
    w
end

function randMNISTgaborfilterBayesOpt(n, λrange, ψupperbound, σrange, γrange, amplitude)
    w = gaborfilter(n, λrange[1] + (λrange[2] - λrange[1]) * rand(),
                    2π*rand(),
                    ψupperbound * rand(),
                    σrange[1] + (σrange[2] - σrange[1]) * rand(),
                    1 - γrange * rand())
    w .* amplitude / sum(abs.(w))
end

function set_connectivity_gabor!(net, inputsize, outputsize, patchsize,
    λrange = [patchsize/4,2*patchsize], 
    ψupperbound = 2π,
    σrange = [patchsize/8,patchsize], 
    γrange = 1, 
    amplitude = 1)
    weights = zeros(outputsize, inputsize)
    input_dim = Int(sqrt(inputsize))
    for i in 1:hiddensize
        mask = zeros(input_dim,input_dim)
        mask[1:patchsize,1:patchsize] = randMNISTgaborfilterBayesOpt(patchsize, λrange, ψupperbound, σrange, γrange, amplitude)
        shifts = rand(0:input_dim-patchsize,2)
        weights[i,:] = 20 * 2 .* circshift(mask,shifts)[:]
        weights[i,:] .*= 1. / norm(weights[i,:], Inf)
    end
    net.weights[1:hiddensize * inputsize] = weights[:]
end
