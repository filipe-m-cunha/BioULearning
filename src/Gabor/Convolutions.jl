using PaddedViews
using PyCall

@pyimport image-similarity-measures

#Computes final image size
function calcFinalSize(ninput::Int, stride::Int, gabSize::Int, padding::String)
    p = 0
    if (padding=="one")
        p = 1
    elseif (padding =="zeros")
        p = 1
    elseif (padding == "none")
    else
        @assert false "padding only supported to be none or same"
    end

    return floor((ninput + 2*p - gabSize)/(stride)) + 1
end

#Adds padding to an image
function addPadding(image, padding)
    if (padding=="none")
    elseif (padding == "one")
        image = PaddedView(1, image, (size(image)[1], size(image)[1]))
    elseif (padding == "zeros")
        image = PaddedView(0, image, (size(image)[1], size(image)[1]))
    else
        @assert false "padding only supported to be none or same"
    end

    return image
end

#Perfoms winnerTakesAll convolution, given a gabor filter bank and an image
function winnerConv(image, gaborBank, stride::Int=1, padding::String="full")

    sizeBank, filter_r, filter_c = size(gaborBank)

    if (filter_r != filter_c)
        throw(DomainError(gaborBank, "Filter row and column should be the same"))
    end

    outSize = calcFinalSize(input_r, stride, filter_r, padding)
    result = zeros(outSize, outSize)
    image = addPadding(image, padding)
    input_r, input_c = size(image)

    for i in 1:input_r
        for j in 1:input_c



