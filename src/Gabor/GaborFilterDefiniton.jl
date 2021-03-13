
#Given the necessary parameters, returns a Gabor filter
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

#Given the parameters, returns a normalized random Gabor Filter
function randgabor(n, λrange, ψupperbound, σrange, γrange, amplitude)
    w = gaborfilter(n, λrange[1] + (λrange[2] - λrange[1]) * rand(),
                    2π*rand(),
                    ψupperbound * rand(),
                    σrange[1] + (σrange[2] - σrange[1]) * rand(),
                    1 - γrange * rand())
    w .* amplitude / sum(abs.(w))
end

function establishConnectionGabor(dataset, nGabor, n, λrange, ψupperbound, σrange, γrange, amplitude, connectionMode):
    #Inicialize empty Gabor filter bank
    gaborBank = Matrix{Float64}[]
    featVectors = Matrix{Float64}[]
    #Create new filters and push to bank
    for i in 1:nGabor
        push!(gaborBank, randgabor(n, λrange, ψupperbound, σrange, γrange, amplitude))
    end

    for j in 1:size(dataset)[2]
        #Perform convolution operation
        if (connectionMode == "winnerTakesAll")
            push!(featVectors, winnerConv(j, gaborBank))
        elseif (connectionMode == "avgDown")
            push!(featVectors, avgConv(j, gaborBank))
        end
    end

    return featVectors
end


