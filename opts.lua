
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network loss options ---------------
    cmd:option('-outDir', '/home/mdfisher/code/network-loss/out/', 'TODO')
    cmd:option('-imageList', '/home/mdfisher/code/network-loss/data/imageListCOCO.txt', 'TODO')
    cmd:option('-batchSize', 4, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 256, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-contentWeight', 5.0, 'TODO')
    cmd:option('-styleWeight', 2.0, 'TODO')
    cmd:option('-TVWeight', 1e-3, 'TODO')
    cmd:option('-styleImage', 'examples/inputs/wave_crop.jpg', 'TODO')
    
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     0.0, 'weight decay')
    
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    
    
    ------------ General options --------------------
    --[[cmd:option('-cache', './imagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './imagenet/imagenet_raw_images/256', 'Home of ImageNet dataset')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'alexnetowtbn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')]]
    cmd:text()

    local opt = cmd:parse(arg or {})
    return opt
end

return M
