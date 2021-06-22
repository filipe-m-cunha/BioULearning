using PaddedViews
using Images
using Infinity


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

    return Int(floor((ninput + 2*p - gabSize)/(stride)) + 1)
end

#Adds padding to an image
function addPadding(image, padding)
    if (padding=="none")
    elseif (padding == "one")
        image = PaddedView(1, image, (size(image)[1], size(image)[1]))
    elseif (padding == "zeros")
        image = PaddedView(0, image, (size(image)[1], size(image)[1]))
    else
        @assert false "padding only supported to be none or one or zeros"
    end

    return image
end

#Perfoms winnerTakesAll convolution, given a gabor filter bank and an image
function winnerConv(image, gaborBank, stride::Int=2, padding::String="zeros")

    sizeBank, filter_r, filter_c = size(gaborBank)
    
    if (filter_r != filter_c)
        throw(DomainError(gaborBank, "Filter row and column should be the same"))
    end
    input_r, input_c = size(image)
    outSize = calcFinalSize(input_r, stride, filter_r, padding)
    result = zeros(outSize, outSize)
    image = addPadding(image, padding)
    input_r, input_c = size(image)
    start = 1
    if (padding == "none")
        start = 0
    end

    for i in 1:stride:(input_r - size(gaborBank)[2])
        for j in 1:stride:(input_c - size(gaborBank)[2])
            measures = [0, -∞]
            imageToCompare = image[i+start: i+filter_r, j+start:j+filter_c]
            for k in 1:sizeBank
                val = assess_ssim(imageToCompare, gaborBank[k, :, :])
                if val > measures[2]
                    measures = [k, val]
                end
            end
            result[i÷stride + 1, j÷stride + 1] = measures[1]
        end
    end

    return result
end



#Perfoms AverageConv convolution, given a gabor filter bank and an image
function avgConv(image, gaborBank, stride::Int=2, padding::String="full")

    sizeBank, filter_r, filter_c = size(gaborBank)

    if (filter_r != filter_c)
        throw(DomainError(gaborBank, "Filter row and column should be the same"))
    end

    outSize = calcFinalSize(input_r, stride, filter_r, padding)
    result = zeros(outSize*sizeBank, outSize*sizeBank)
    image = addPadding(image, padding)
    input_r, input_c = size(image)
    start = 1
    if (padding == "none")
        start = 0
    end

    for i in 1:stride:(input_r - size(gaborBank)[2])
        for j in 1:stride:(input_c - size(gaborBank)[2])
            measures = zeros(sizeBank)
            imageToCompare = image[i+start: i+filter_r, j+start:j+filter_c]
            for k in 1:sizeBank
                val = assess_ssim(imageToCompare, gaborBank[k, :, :])
                measures[k] = val
            end
            result[(i-1)*sizeBank: i*sizeBank, (j-1)*sizeBank:j*sizeBank] = measures
        end
    end

    return result
end
            

function inverseConv(gaborBank, x, dim1, dim2, stride)
    xres = reshape(x, dim1[1], dim1[2])
    xret = zeros(dim2[1], dim2[2])
    for i in 1:stride:(dim2[1] - size(gaborBank)[2])
        for j in:stride:(dim2[2]- size(gaborBank[2]))
            xret[i:i + size(gaborBank[2]), j:j+size(gaborBank[2])] = gaborBank[xres[i,j]]
        end
    end
    return xret
end


