
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     10,   1e-2,   5e-4, },
        { 11,     29,   1e-5,   5e-4  },
        { 30,     43,   1e-6,   0 },
        { 44,     52,   5e-7,   0 },
        { 53,    1e8,   1e-7,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.outDir, 'train.log'))
local batchNumber
local totalBatchCount = 0
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train(imageLoader)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimState = {
        learningRate = params.learningRate,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        dampening = 0.0,
        weightDecay = params.weightDecay
        }
    end
    batchNumber = 0
    cutorch.synchronize()

    -- set the dropouts to training mode
    fullNetwork:training()

    local tm = torch.Timer()
    lossEpoch = 0
    for i = 1, opt.epochSize do
        local batch = sampleBatch(imageLoader)
        trainBatch(batch.inputs, batch.labels)
    end
    
    --[[for i = 1, opt.epochSize do
        -- queue jobs to data-workers
        donkeys:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local batch = sampleBatch(imageLoader)
                return batch.inputs, batch.labels
            end,
            -- the end callback (runs in the main thread)
            trainBatch
        )
    end
    donkeys:synchronize()]]

    cutorch.synchronize()

    lossEpoch = lossEpoch / (opt.batchSize * opt.epochSize)

    trainLogger:add{
    ['avg loss (train set)'] = lossEpoch
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t',
        epoch, tm:time().real, lossEpoch, lossEpoch))
    print('\n')

    -- save model
    collectgarbage()

    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    --model:clearState()
    --torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end

-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = fullNetwork:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    if totalBatchCount == 0 then
        --dumpNet(fullNetwork, inputs, opt.outDir .. 'full/')
        --dumpNet(vggContentNetwork, labels, opt.outDir .. 'content/')
    end
        
    local err, contentOutputs, contentTargets
    feval = function(x)
        contentTargets = vggContentNetwork:forward(labels):clone()
        
        if totalBatchCount % 100 == 0 then
            local inClone = inputs[1]:clone()
            inClone:add(0.5)
            
            local outClone = transformNetwork:forward(inputs)[1]:clone()
            outClone = caffeDeprocess(outClone)
            --outClone:add(0.5)
            
            image.save(opt.outDir .. 'sample' .. totalBatchCount .. '_in.png', inClone)
            image.save(opt.outDir .. 'sample' .. totalBatchCount .. '_out.png', outClone)
        end
        
        fullNetwork:zeroGradParameters()
        contentOutputs = fullNetwork:forward(inputs)
        
        --outputs = transformNetwork:forward(inputs)
        
        if totalBatchCount == 0 then
            --saveTensor(inputs, opt.outDir .. 'inputs.csv')
            --saveTensor(labels, opt.outDir .. 'labels.csv')
            --saveTensor(contentTargets, opt.outDir .. 'contentTargets.csv')
            --saveTensor(contentOutputs, opt.outDir .. 'contentOutputs.csv')
        end
        
        err = contentCriterion:forward(contentOutputs, contentTargets)
        local gradOutputs = contentCriterion:backward(contentOutputs, contentTargets)
        fullNetwork:backward(inputs, gradOutputs)
        vggContentNetwork:zeroGradParameters()
        
        return err, gradParameters
    end
    optim.adam(feval, parameters, optimState)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    lossEpoch = lossEpoch + err

    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err,
        optimState.learningRate, dataLoadingTime))

    dataTimer:reset()
    totalBatchCount = totalBatchCount + 1
end

--[[
    local err, contentOutputs, contentTargets
    feval = function(x)
        --contentTargets = vggContentNetwork:forward(labels):clone()
        
        fullNetwork:zeroGradParameters()
        contentOutputs = fullNetwork:forward(inputs)

        err = contentCriterion:forward(contentOutputs, labels)
        local gradOutputs = contentCriterion:backward(contentOutputs, labels)
        fullNetwork:backward(inputs, gradOutputs)
        return err, gradParameters
        
        --[[err = contentCriterion:forward(contentOutputs, contentTargets)
        local gradOutputs = contentCriterion:backward(contentOutputs, contentTargets)
        fullNetwork:backward(inputs, gradOutputs)
        --vggContentNetwork:zeroGradParameters()
        
        return err, gradParameters
    end]]
