
-- TODO: turn this into a class
require 'image'

function loadImageList()
    
    --local allImages = getFileListRecursive('/home/mdfisher/ssd2/ImageNet/CLS-LOC/train/')
    --local allImages = getFileListRecursive('/home/mdfisher/ssd2/COCO/train2014/')
    --writeAllLines(opt.imageList, allImages)

    print('Initializing images from: ' .. opt.imageListSingleFrame)
    
    local imageList = util.readAllLines(opt.imageListSingleFrame)
    print('loaded ' .. #imageList .. ' images')
    return imageList
end

function loadImagePairsList()
    if not util.fileExists(opt.imageListFramePairs) then
        
    end

    print('Initializing image pairs from: ' .. opt.imageListFramePairs)
    
    local imagePairsList = util.readAllLines(opt.imageListFramePairs)
    print('loaded ' .. #imageListPairs .. ' images')
    return imagePairsList
end

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.cropSize, opt.cropSize}

local function loadAndResizeImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local function loadAndCropImage(path)
   collectgarbage()
   local input = loadAndResizeImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   if iH == oH then h1 = 0 end
   if iW == oW then w1 = 0 end
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end
   return out
end

function sampleBatchSingleFrame(imageList)
    -- pick an index of the datapoint to load next
    local reflectionPadding = 50
    local batchInputs = torch.FloatTensor(opt.batchSize, 3, opt.cropSize + reflectionPadding * 2, opt.cropSize + reflectionPadding * 2)
    local batchLabels = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    for i = 1, opt.batchSize do
        local imageFilename = imageList[ math.random( #imageList ) ]
        donkeys:addjob(
            function()
                local imgInput = loadAndCropImage(imageFilename)
                --local imgTarget = imgInput
                
                local imgTarget = caffePreprocess(imgInput:clone())
                
                imgInput:add(-0.5)
                imgInput = reflectionPadImage(imgInput, reflectionPadding)
                
                return imgInput, imgTarget
            end,
            function(imgInput, imgTarget)
                batchInputs[i] = imgInput
                batchLabels[i] = imgTarget
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.inputs = batchInputs
    batch.labels = batchLabels
    return batch
end

function sampleBatchFramePair(imageLoader)
    -- pick an index of the datapoint to load next
    local reflectionPadding = 50
    local batchInputs = torch.FloatTensor(opt.batchSize, 3, opt.cropSize + reflectionPadding * 2, opt.cropSize + reflectionPadding * 2)
    local batchLabels = torch.FloatTensor(opt.batchSize, 3, opt.cropSize, opt.cropSize)
    
    for i = 1, opt.batchSize do
        local imageFilename = imageLoader.imageList[ math.random( #imageLoader.imageList ) ]
        donkeys:addjob(
            function()
                local imgInput = loadAndCropImage(imageFilename)
                --local imgTarget = imgInput
                
                local imgTarget = caffePreprocess(imgInput:clone())
                
                imgInput:add(-0.5)
                imgInput = reflectionPadImage(imgInput, reflectionPadding)
                
                return imgInput, imgTarget
            end,
            function(imgInput, imgTarget)
                batchInputs[i] = imgInput
                batchLabels[i] = imgTarget
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.inputs = batchInputs
    batch.labels = batchLabels
    return batch
end
