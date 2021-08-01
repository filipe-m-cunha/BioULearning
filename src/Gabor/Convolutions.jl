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
function winnerConv(image, gaborBank, stride::Int=1, padding::String="zeros")

    sizeBank, filter_r, filter_c = size(gaborBank)
    
    if (filter_r != filter_c)
        throw(DomainError(gaborBank, "Filter row and column should be the same"))
    end
    input_r, input_c = size(image)
    #outSize = calcFinalSize(input_r, stride, filter_r, padding)
    outSize = input_r - filter_r + 1
    result = zeros(outSize, outSize)
    #image = addPadding(image, padding)
    input_r, input_c = size(image)
    start = 1
    #if (padding == "none")
    #    start = 0
    #end

    for i in 1:outSize
        for j in 1:outSize
            measures = [0, -∞]
            imageToCompare = image[i: i+filter_r-1, j:j+filter_c-1]
            for k in 1:sizeBank
                val = assess_ssim(imageToCompare, gaborBank[k, :, :])
                if val > measures[2]
                    measures = [k, val]
                end
            end
            result[i, j] = measures[1]
        end
    end

    return result
end

function deconvolutionWinnerTAll(featVec, gaborBank)
    x = size(gaborBank[1, :, :])[1]
    y = size(Xtemp[1, :, :])[1]
    imgSize = x + y - 1
    img = zeros(imgSize, imgSize)
    for i in 1:imgSize
        for j in 1:imgSize
            for k1 in max(1, i+x-imgSize):min(x, i)
                for k2 in max(1, j+x-imgSize):min(x, j)
                    img[i, j] += (1/(min(x, j) - max(1, j+x-imgSize)+1))*(1/(min(x, i-x, i) - max(1, i+x-imgSize)+1))*gaborBank[convert(Int64, featVec[i-k1+1, j-k2+1]), :, :][k1, k2]
                    #=if(isnan(img[i,j]))
                        println("i: ", i, " j: ", j, " k1: ", k1, " k2: ", k2)
                        println("min1: ", min(x, j))
                        println("max1: ", max(1, j+x-imgSize)+1)
                        println("min2: ", min(x, i-x, i))
                        println("max2: ", max(1, i+x-imgSize)+1)
                        println("val: ", gaborBank[convert(Int64, featVec[i-k1+1, j-k2+1]), :, :][k1, k2])
                    end=#
                end
            end
        end
    end
    return img
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


