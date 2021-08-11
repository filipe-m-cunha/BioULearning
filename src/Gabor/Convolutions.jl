using PaddedViews
using Images
using Infinity


#Perfoms winnerTakesAll convolution, given a gabor filter bank and an image
function conv_forward(img, gaborBank, stride=1, padding=0)
    img_H, img_W = size(img)
    sizeBank ,filter_H, filter_W = size(gaborBank)
    
    n_H = convert(Int64, (img_H + 2*padding - filter_H)/stride)+1
    n_W = convert(Int64, (img_W + 2*padding - filter_W)/stride)+1
    
    z = zeros(n_H, n_W)
    padded_img = PaddedView(0, img, (1:img_H+2*padding,1:img_W + 2*padding), (1+padding:img_H+padding, 1 + padding : img_W + padding))
    
    for h in 1:n_H
        for w in 1:n_W
            vert_start = h*stride
            vert_end = h*stride + filter_H
            horiz_start = w*stride
            horiz_end = w*stride + filter_W
            
            img_slice = padded_img[vert_start:vert_end-1, horiz_start:horiz_end-1]
            
            measures = [0, -∞]

            for k in 1:sizeBank
                val = assess_ssim(img_slice, gaborBank[k, :, :])
                if val > measures[2]
                    measures = [k, val]
                end
            end
            
            z[h, w] = measures[1]
            
        end
    end
    
    return z
    
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
                end
            end
        end
    end
    return img
end  


#=Perfoms AverageConv convolution, given a gabor filter bank and an image
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
=#


