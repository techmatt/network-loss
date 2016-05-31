require 'torch'
require 'math'
require 'lfs'

function getSize(tensor)
    if #tensor:size() == 2 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ']'
    elseif #tensor:size() == 3 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ' ' ..
                      tostring(tensor:size()[3]) .. ']'
    elseif #tensor:size() == 4 then
        return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                      tostring(tensor:size()[2]) .. ' ' ..
                      tostring(tensor:size()[3]) .. ' ' ..
                      tostring(tensor:size()[4]) .. ']'
    else
        return '[unknown vector size]'
    end
end

function describeNet(network, inputs)
    print('dumping network, input size: ' .. getSize(inputs))
    local subnet = nn.Sequential()
    for i, module in ipairs(network:listModules()) do
        local moduleType = torch.type(module)
        --print('module ' .. i .. ': ' .. moduleType)
        if tostring(moduleType) ~= 'nn.Sequential' then
            subnet:add(module)
            local outputs = subnet:forward(inputs)
            print('module ' .. i .. ': ' .. getSize(outputs) .. ': ' .. tostring(module))
        end
    end
end

function dumpNet(network, inputs, dir)
    lfs.mkdir(dir)
    print('dumping network, input size: ' .. getSize(inputs))
    local subnet = nn.Sequential()
    for i, module in ipairs(network:listModules()) do
        local moduleType = torch.type(module)
        --print('module ' .. i .. ': ' .. moduleType)
        if tostring(moduleType) ~= 'nn.Sequential' then
            subnet:add(module)
            local outputs = subnet:forward(inputs)
            print('module ' .. i .. ': ' .. getSize(outputs) .. ': ' .. tostring(module))
            saveTensor(outputs, dir .. i .. '_' .. moduleType .. '.csv')
        end
    end
end

function saveTensor(tensor, filename)
    local out = assert(io.open(filename, "w"))
    out:write(getSize(tensor) .. '\n')
    
    local maxDim = 32
    
    local splitter = ","
    for a=1,math.min(maxDim,tensor:size(1)) do
        for b=1,math.min(maxDim,tensor:size(2)) do
            for c=1,math.min(maxDim,tensor:size(3)) do
                for d=1,math.min(maxDim,tensor:size(4)) do
                    out:write(tensor[a][b][c][d])
                    if d == tensor:size(4) or d == maxDim then
                        out:write("\n")
                    else
                        out:write(splitter)
                    end
                end
            end
        end
    end

    out:close()
end