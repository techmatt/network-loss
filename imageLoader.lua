
-- TODO: turn this into a class
require 'image'

function makeImageLoader()
    print('Initializing images from: ' .. opt.imageList)
    
    local result = {}
    result.imageList = readAllLines(opt.imageList)
    result.imageCount = #result.imageList
    print('loaded ' .. result.imageCount .. ' images')
    return result
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
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end
   return out
end

function sampleBatch(imageLoader)
    local batchSize = opt.batchSize
    
    -- pick an index of the datapoint to load next
    local batchInputs = torch.FloatTensor(batchSize, 3, opt.cropSize, opt.cropSize)
    local batchLabels = torch.FloatTensor(batchSize, 3, opt.cropSize, opt.cropSize)
    
    --[[for i = 1, batchSize do
        local imageFilename = imageLoader.imageList[ math.random( #imageLoader.imageList ) ]
        
        donkeys:addjob(
            function()
                local img = loadAndCropImage(imageFilename)
                img:add(-0.5)
                return img
            end,
            function(img)
                batchInputs[i] = imgA
                batchLabels[i] = imgB
            end)
        --local imgA = loadAndCropImage(imageFilename)
        --
        
        --local imgB = loadAndCropImage(imageFilename)
        local imgB = imgA
        batchInputs[i] = imgA
        batchLabels[i] = imgB
    end]]
    for i = 1, batchSize do
        local imageFilename = imageLoader.imageList[ math.random( #imageLoader.imageList ) ]
        
        donkeys:addjob(
            function()
                local img = loadAndCropImage(imageFilename)
                img:add(-0.5)
                return img
            end,
            function(img)
                batchInputs[i] = img
                batchLabels[i] = img
            end)
    end
    donkeys:synchronize()
    
    local batch = {}
    batch.inputs = batchInputs
    batch.labels = batchLabels
    return batch
end
