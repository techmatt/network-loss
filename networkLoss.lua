
--
-- debug coonfig options
--
local printModel = false

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

fullNetwork, transformNetwork, vggTotalNetwork, vggContentNetwork, contentLossModule, styleLossModules = createModel()
cudnn.convert(fullNetwork, cudnn)
cudnn.convert(vggContentNetwork, cudnn)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
fullNetwork = fullNetwork:cuda()
vggContentNetwork = vggContentNetwork:cuda()

if printModel then
    print('=> Model')
    print(fullNetwork)
end

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

