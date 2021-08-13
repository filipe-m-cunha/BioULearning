using PaddedViews
using Images
using Infinity


function elementWiseMult(w1, w2)
    sum = 0
    for i in size(w1)[1]
        for j in size(w2)[2]
            sum += w1[i, j]*w2[i,j]
        end
    end
    return sum
end
    

#Perfoms winnerTakesAll convolution, given a gabor filter bank and an image
function conv_forward(img, gaborBank, stride::Int64=1, padding::Int64=0)
    img_H, img_W = size(img)
    sizeBank ,filter_H, filter_W = size(gaborBank)
    
    n_H = convert(Int64, (img_H + 2*padding - filter_H)/stride)+1
    n_W = convert(Int64, (img_W + 2*padding - filter_W)/stride)+1
    
    z = zeros(n_H, n_W)
    padded_img = PaddedView(0, img, (1:img_H+2*padding,1:img_W + 2*padding), (1+padding:img_H+padding, 1 + padding : img_W + padding))
    for h in 0:n_H-1
        for w in 0:n_W-1
            vert_start = h*stride+1
            vert_end = h*stride + filter_H+1
            horiz_start = w*stride+1
            horiz_end = w*stride +filter_W + 1
            #println(vert_end)
            #println(horiz_end)
            img_slice = padded_img[vert_start:vert_end-1, horiz_start:horiz_end-1]
            
            measures = [0, -∞]

            for k in 1:sizeBank
                val = assess_ssim(img_slice, gaborBank[k, :, :])
                if val > measures[2]
                    measures = [k, val]
                end
            end
            
            z[h+1, w+1] = measures[1]
            
        end
    end
    return z
    
end

function conv_forwardAlt(img, gaborBank, stride::Int64=1, padding::Int64=0)
    img_H, img_W = size(img)
    sizeBank ,filter_H, filter_W = size(gaborBank)
    
    n_H = convert(Int64, (img_H + 2*padding - filter_H)/stride)+1
    n_W = convert(Int64, (img_W + 2*padding - filter_W)/stride)+1
    
    z = zeros(n_H, n_W)
    padded_img = PaddedView(0, img, (1:img_H+2*padding,1:img_W + 2*padding), (1+padding:img_H+padding, 1 + padding : img_W + padding))
    for h in 0:n_H-1
        for w in 0:n_W-1
            vert_start = h*stride+1
            vert_end = h*stride + filter_H+1
            horiz_start = w*stride+1
            horiz_end = w*stride +filter_W + 1
            #println(vert_end)
            #println(horiz_end)
            img_slice = padded_img[vert_start:vert_end-1, horiz_start:horiz_end-1]
            
            measures = [0, -∞]

            for k in 1:sizeBank
                val = assess_ssim(img_slice, gaborBank[k, :, :])
                if val > measures[2]
                    measures = [k, val]
                end
            end
            
            z[h+1, w+1] = elementWiseMult(img_slice, gaborBank[convert(Int64, measures[1]), :, :])
            
        end
    end
    return z
    
end

function deconvolutionWinnerTAll(featVec, gaborBank, stride::Int64=1, padding::Int64=0)
    x = size(gaborBank[1, :, :])[1]
    y = size(featVec)[1]
    imgSize = y*stride - 2*padding + x - 1
    img = zeros(imgSize+2*padding, imgSize+2*padding)
    for w in 0:size(featVec)[1]-1
        for h in 0:size(featVec)[2]-1
            vert_start = h*stride+1
            vert_end = h*stride + x+1
            horiz_start = w*stride+1
            horiz_end = w*stride +x + 1
            img[vert_start:vert_end-1, horiz_start:horiz_end-1] += gaborBank[convert(Int64,featVec[h+1, w+1]), :, :]
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


