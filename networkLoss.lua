--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

paths.dofile('util.lua')
paths.dofile('torchUtil.lua')
paths.dofile('loadModel.lua')
paths.dofile('imageLoader.lua')

--print(opt)

--local allImages = getFileListRecursive('/home/mdfisher/ssd2/ImageNet/CLS-LOC/train/')
--writeAllLines(opt.imageList, stuff)

model = createModel()
cudnn.convert(model, cudnn)

-- 2. Create Criterion
criterion = nn.MSECriterion()

--[[print('=> Model')
print(model)

print('=> Criterion')
print(criterion)]]

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()
print('imagelist: ', opt.imageList)
local imageLoader = makeImageLoader()

--local batch = sampleBatch(imageLoader)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.outDir)
os.execute('mkdir -p ' .. opt.outDir)

paths.dofile('threadPool.lua')
paths.dofile('train.lua')
--paths.dofile('test.lua')

epoch = 1

for i=1,opt.epochCount do
   train(imageLoader)
   --test()
   epoch = epoch + 1
end
--]]
