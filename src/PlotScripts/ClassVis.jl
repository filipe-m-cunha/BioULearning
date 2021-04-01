using DataFrames;
using StatsPlots;
using VegaLite;

include("../ModelError/ErrFunction.jl");

function get_Dict_Classes(placement)
    Dictionary = Dict(placement[1, :] => 1)
    j = 2
    for i in 1:length(placement)
        if placement[i, :] ∉ keys(Dictionary)
            Dictionary[placement[i, :]] = j
            j = j+1
        end
    end

    return Dictionary
end

function plot_Fake_Classes(dataFrame, wX, θ)

    placement = placeDataset(dataFrame, wX, θ)
    dictionary = get_Dict_Classes(dataFrame)

    z = Int[]
    for i in 1:length(placement)
        if get(dictionary, placement[i, :], 0)
            println("Error")
            break
        else
            append!(z, get(dictionary, placement[i, :], 0))
        end
    end

    @vlplot(:point, x=:placement[:, 1], y=:placement[:,2] color=:z)
end



