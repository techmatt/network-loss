
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'loadcaffe'

paths.dofile('nnModules.lua')

function addConvElement(network,iChannels,oChannels,size,stride,padding)
   network:add(nn.SpatialConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding))
   network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
   network:add(nn.ReLU(true))
end

function addUpConvElement(network,iChannels,oChannels,size,stride,padding,extra)
   network:add(nn.SpatialFullConvolution(iChannels,oChannels,size,size,stride,stride,padding,padding,extra,extra))
   network:add(cudnn.SpatialBatchNormalization(oChannels,1e-3))
   network:add(nn.ReLU(true))
end

function deleteNetwork(net)
    for i=1,#net.modules do
        local module = net.modules[i]
        if torch.type(module) == 'nn.SpatialConvolutionMM' then
            -- remove these, not used, but uses gpu memory
            --module.gradWeight = nil
            --module.gradBias = nil
        end
    end
end

function createVGGDebug()
    local styleLossModules = {}
    local vggContentOut = nn.Sequential()
    vggContentOut:add(nn.SpatialConvolution(3,3,1,1,1,1,0,0))
    --vggContentOut:add(nn.SpatialConvolution(32,3,3,3,1,1,0,0))
    
    collectgarbage()
    return vggContentOut, styleLossModules
end

function createVGG()
    --local contentImage = image.load(opt.contentImage, 3)
    --contentImage = image.scale(contentImage, opt.cropSize, 'bilinear')
    --local contentImageCaffe = caffePreprocess(contentImage):float()
    --contentImageCaffe:cuda()
  
    local styleLossModules = {}
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()
    --vggIn:cuda()
    local vggContentOut = nn.Sequential()
    --vgg:cuda()
    local vggDepth = 9
    local contentName = 'relu2_2'
    for i = 1, vggDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        print('layer ' .. i .. '; ' .. name)
        local layerType = torch.type(layer)
        vggContentOut:add(layer)
        --local isPooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
        
        -- style and loss are handled very differently
        if name == contentName then
            print("Setting up content layer" .. i .. ": " .. name)
            --[[
            local target = vggOut:forward(contentImageCaffe):clone()
            local norm = false
            local contentLossModule = nn.ContentLoss(opt.contentWeight, target, norm):float()
            contentLossModule:cuda()
            vggOut:add(contentLossModule)
            lossModules.content[#lossModules + 1] = contentLossModule]]
        end
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut, styleLossModules
end

function createModel()
    print('Creating model')
   
   local transformNetwork = nn.Sequential()
   local fullNetwork = nn.Sequential()
   
   --[[if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end]]
   addConvElement(transformNetwork, 3, 32, 9, 1, 4)
   addConvElement(transformNetwork, 32, 64, 3, 2, 2)
   addConvElement(transformNetwork, 64, 128, 3, 2, 2)
   
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   --[[addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)
   addConvElement(transformNetwork, 128, 128, 3, 1, 1)]]
   
   addUpConvElement(transformNetwork, 128, 64, 3, 2, 1, 0)
   addUpConvElement(transformNetwork, 64, 32, 3, 2, 2, 1)
   
   transformNetwork:add(nn.SpatialConvolution(32,3,3,3,1,1,0,0))
   
   local vggContentNetwork, styleLossModules = createVGG()
   --local vggContentNetwork, styleLossModules = createVGGDebug()
   
   fullNetwork:add(transformNetwork)
   fullNetwork:add(vggContentNetwork)
   
   return fullNetwork, transformNetwork, vggContentNetwork, styleLossModules
end
