
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Image style transfer using network loss')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network loss options ---------------
    cmd:option('-styleImage', 'examples/inputs/picassoA.jpg', 'TODO')
    cmd:option('-styleName', 'picassoA', 'TODO')
    cmd:option('-outDir', '/home/mdfisher/code/network-loss/out/picassoA/', 'TODO')
    cmd:option('-imageListSingleFrame', '/home/mdfisher/code/network-loss/data/imageListCOCO.txt', 'TODO')
    cmd:option('-imageListFramePairs', '/home/mdfisher/code/network-loss/data/imageListMovieClips.txt', 'TODO')
    cmd:option('-batchSize', 4, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 256, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-trainingMovieModel', false, 'TODO')
    cmd:option('-predecessorMatchWeight', 0.01, 'TODO')
    
    cmd:option('-movieInDir', '/home/mdfisher/raid/datasets/videos/', 'TODO')
    cmd:option('-movieClipDir', '/home/mdfisher/movieClips/', 'TODO')
    cmd:option('-movieClipLength', 8, 'TODO')
    cmd:option('-clipsPerMovie', 600, 'TODO')
    
    cmd:option('-contentWeight', 2.0, 'TODO')
    cmd:option('-styleWeight', 0.5, 'TODO')
    cmd:option('-TVWeight', 1e-4, 'TODO')
    
    cmd:option('-manualSeed', 2, 'Manually set RNG seed')
    cmd:option('-GPU', 1, 'Default preferred GPU')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',         20,    'Number of total epochs to run')
    cmd:option('-epochSize',       1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    
    local opt = cmd:parse(arg or {})
    
    opt.printModel = false
    opt.describeNets = false
    opt.useResidualBlock = false
    
    return opt
end

return M
