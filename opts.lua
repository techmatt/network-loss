--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    
    ------------ Network loss options ---------------
    cmd:option('-outDir', '/home/mdfisher/code/network-loss/out/', 'TODO')
    cmd:option('-imageList', '/home/mdfisher/code/network-loss/data/imageList.txt', 'TODO')
    cmd:option('-batchSize', 16, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-imageSize', 256, 'Smallest side of the resized image')
    cmd:option('-cropSize', 250, 'Height and Width of image crop to be used as input layer')
    
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    
    ------------- Training options --------------------
    cmd:option('-epochCount',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       200, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    
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
