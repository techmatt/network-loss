
paths.dofile('common.lua')

util = paths.dofile('util.lua')
paths.dofile('torchUtil.lua')
paths.dofile('nnModules.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

--local testImg = image.load(opt.styleImage)
--local paddedImg = reflectionPadImage(testImg, 100)
--image.save(opt.outDir .. 'padded.png', paddedImg)

--local allImages = getFileListRecursive('/home/mdfisher/ssd2/ImageNet/CLS-LOC/train/')
--local allImages = getFileListRecursive('/home/mdfisher/ssd2/COCO/train2014/')
--writeAllLines(opt.imageList, allImages)

--paths.dofile('movieProcessor.lua')
--extractAllClips()
--transformAllClips('out/transformPaintingA.t7')
--transformImageDirectory(opt.outDir .. 'models/transform.t7', 'data/Sintel/framesIn/', opt.outDir .. 'movieOut256/', 267, 62, 360, 360, 256, 256)

paths.dofile('loadModel.lua')
paths.dofile('imageLoader.lua')
paths.dofile('threadPool.lua')

--print(opt)

model = createModel()

local imageLoader = makeImageLoader()

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.outDir)
lfs.mkdir(opt.outDir)
lfs.mkdir(opt.outDir .. 'models/')
lfs.mkdir(opt.outDir .. 'samples/')

paths.dofile('train.lua')

epoch = 1

for i=1,opt.epochCount do
   train(imageLoader)
   epoch = epoch + 1
end

