
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
    local styleImage = image.load(opt.styleImage, 3)
    styleImage = image.scale(styleImage, opt.cropSize, 'bilinear')
    local styleImageCaffe = caffePreprocess(styleImage):float()
    --contentImageCaffe:cuda()
  
    local styleLossModules = {}
    local contentLossModule = {}
    local vggIn = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn'):float()
    --vggIn:cuda()
    local vggContentOut = nn.Sequential()
    local vggTotalOut = nn.Sequential()
    --vgg:cuda()
    local vggDepth = 9
    local contentDepth = 9
    local contentName = 'relu2_2'
    
    local styleNames = {}
    --styleNames['relu1_2'] = true
    --styleNames['relu2_2'] = true
    --styleNames['relu3_3'] = true
    --styleNames['relu4_2'] = true
    
    for i = 1, vggDepth do
        local layer = vggIn:get(i)
        local name = layer.name
        print('layer ' .. i .. '; ' .. name)
        local layerType = torch.type(layer)
        
        vggTotalOut:add(layer)
        --local isPooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

        if i <= contentDepth then
            vggContentOut:add(layer)
            if name == contentName then
                print("Setting up content layer" .. i .. ": " .. name)
                local contentTarget = nil
                local norm = false
                contentLossModule = nn.ContentLoss(opt.contentWeight, contentTarget, norm):float()
                vggTotalOut:add(contentLossModule)
            end
        end
        
        if styleNames[name] then
            print("Setting up style layer" .. i .. ": " .. name)
            local styleTarget = vggTotalOut:forward(styleImageCaffe):clone()
            local norm = false
            local styleLossModule = nn.StyleLoss(opt.styleWeight, styleTarget, norm):float()
            vggTotalOut:add(styleLossModule)
            styleLossModules[#styleLossModules + 1] = styleLossModule
        end
    end
    
    vggIn = nil
    collectgarbage()
    return vggContentOut, vggTotalOut, contentLossModule, styleLossModules
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
   
   local vggContentNetwork, vggTotalNetwork, contentLossModule, styleLossModules = createVGG()
   
   fullNetwork:add(transformNetwork)
   fullNetwork:add(vggTotalNetwork)
   
   return fullNetwork, transformNetwork, vggTotalNetwork, vggContentNetwork, contentLossModule, styleLossModules
end
