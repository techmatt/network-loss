
local imageLoaderClass = torch.class('imageLoader')

function imageLoaderClass:__init(opt)

    print('Initializing images from: ', opt.imageList)
    self.imageList = readAllLines(opt.imageList)
    self.imageCount = #self.imageList
    print('loaded ' .. self.imageCount .. ' images')
end

function imageLoaderClass:getBatch(opt)
    local batchSize = utils.getopt(opt, 'batchSize', 5)
    
    -- pick an index of the datapoint to load next
    local batchImages = torch.FloatTensor(batchSize, 3, 256, 256)
    local batchLabels = torch.FloatTensor(batchSize, 3, 256, 256)
    
    for i = 1, batchSize do
        local imageFilename = self.imageList[ math.random( #self.imageList ) ]
        local imgA = image.load(imageFilename,3,'float')
        local imgB = image.load(imageFilename,3,'float')
        batchImages[i] = imgA
        batchLabels[i] = imgB
    end
    
    local batch = {}
    batch.images = batchImages
    batch.labels = batchLabels
    return batch
end

return imageLoaderClass