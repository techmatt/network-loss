--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

function addConvElement(network,iChannels,oChannels,size,stride,padding)
   network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
   network:add(nn.SpatialBatchNormalization(oChannels,1e-3))
   network:add(nn.ReLU(true))
end

function addUpConvElement(network,iChannels,oChannels,size,stride,padding)
   network:add(nn.SpatialDilatedConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,2,2))
   network:add(nn.SpatialBatchNormalization(oChannels,1e-3))
   network:add(nn.ReLU(true))
end

function createModel()
    print('Creating model')
   
   local transformNetwork = nn.Sequential()
   
   addConvElement(transformNetwork, 3, 32, 9, 1, 1)
   addConvElement(transformNetwork, 32, 64, 3, 2, 1)
   addConvElement(transformNetwork, 64, 128, 3, 2, 1)
   
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   
   addUpConvElement(transformNetwork, 128, 64, 3, 1, 1)
   addUpConvElement(transformNetwork, 128, 32, 3, 1, 1)
   addConvElement(transformNetwork, 128, 3, 9, 1, 1)
   
   transformNetwork.imageSize = 256
   transformNetwork.imageCrop = 224
   
   --local model = nn.Sequential()
   --model:add(features):add(classifier)

   return transformNetwork
end

model = createModel()
cudnn.convert(model, cudnn)

-- 2. Create Criterion
criterion = nn.MSECriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()
