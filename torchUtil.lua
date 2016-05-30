require 'torch'

function getSize(tensor)
    return '[' .. tostring(tensor:size()[1]) .. ' ' ..
                  tostring(tensor:size()[2]) .. ' ' ..
                  tostring(tensor:size()[3]) .. ' ' ..
                  tostring(tensor:size()[4]) .. ']'
end

function dumpNet(network, inputs)
    print('dumping network, input size: ' .. getSize(inputs))
    local subnet = nn.Sequential()
    for i, module in ipairs(network:listModules()) do
        if i >= 2 then
            subnet:add(module)
            local outputs = subnet:forward(inputs)
            print(getSize(outputs) .. ': ' .. tostring(module))
        end
    end
end
