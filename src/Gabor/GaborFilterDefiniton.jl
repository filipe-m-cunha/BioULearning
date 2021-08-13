
include("Convolutions.jl")

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

function establishConnectionGabor(dataset, nGabor, n, λrange, ψupperbound, σrange, γrange, amplitude, connectionMode, stride::Int64=1, padding::Int64=0)
    #Inicialize empty Gabor filter bank
    gaborBank = zeros(nGabor + 1, n, n)
    #finalDim = calcFinalSize(size(dataset)[2], stride, n, "zeros")
    finalDim = convert(Int64, (size(dataset)[2] + 2*padding - n)/stride)+1
    featVectors = zeros(size(dataset)[3], finalDim, finalDim)
    #Create new filters and push to bank
    for i in 1:nGabor
        gaborBank[i, :, :] = randgabor(n, λrange, ψupperbound, σrange, γrange, amplitude)
    end
    gaborBank[nGabor + 1, :, :] =  dataset[1].*ones(n, n)
    println("Starting Convolutions")
    for j in 1:size(dataset)[3]
        if (j%100==0)
            acc = j/size(dataset)[3]
            @printf "Now %.2f%%\n" acc * 100
        end
        #Perform convolution operation
        if (connectionMode == "winnerTakesAll")
            newImage= conv_forward(dataset[:, :, j], gaborBank, stride, padding)
            featVectors[j, :, :] = newImage
            #=
        elseif (connectionMode == "avgDown")
            featVectors[j, :, :] = avgConv(dataset[:, :, j], gaborBank)
        =#
        end
    end
    println("Ended Convolutions")
    return featVectors, gaborBank
end


function establishConnectionGaborAlt(dataset, nGabor, n, λrange, ψupperbound, σrange, γrange, amplitude, connectionMode, stride::Int64=1, padding::Int64=0)
    #Inicialize empty Gabor filter bank
    gaborBank = zeros(nGabor + 1, n, n)
    #finalDim = calcFinalSize(size(dataset)[2], stride, n, "zeros")
    finalDim = convert(Int64, (size(dataset)[2] + 2*padding - n)/stride)+1
    featVectors = zeros(size(dataset)[3], finalDim, finalDim)
    #Create new filters and push to bank
    for i in 1:nGabor
        gaborBank[i, :, :] = randgabor(n, λrange, ψupperbound, σrange, γrange, amplitude)
    end
    gaborBank[nGabor + 1, :, :] =  dataset[1].*ones(n, n)
    println("Starting Convolutions")
    for j in 1:size(dataset)[3]
        if (j%100==0)
            acc = j/size(dataset)[3]
            @printf "Now %.2f%%\n" acc * 100
        end
        #Perform convolution operation
        if (connectionMode == "winnerTakesAll")
            newImage= conv_forwardAlt(dataset[:, :, j], gaborBank, stride, padding)
            featVectors[j, :, :] = newImage
        elseif (connectionMode == "avgDown")
            featVectors[j, :, :] = avgConv(dataset[:, :, j], gaborBank)
        end
    end
    println("Ended Convolutions")
    return featVectors, gaborBank
end

