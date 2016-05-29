--require 'cunn'
--local ffi=require 'ffi'
require 'lfs'

function getFileListRecursive(dir)
    result = {}
    for file in lfs.dir(dir) do
        if lfs.attributes(file,"mode") == "file" then
            result[#result] = file
        elseif lfs.attributes(file,"mode")== "directory" then
            --getFileListRecursive(file)
            --for l in lfs.dir("C:\\Program Files\\"..file) do
            --     print("",l)
            --end
        end
    end
    return result
end

function getDirList(dir)

end

function fileExists(file)
    local f = io.open(file, "rb")
    if f then f:close() end
    return f ~= nil
end

function readAllLines(file)
    if not fileExists(file) then 
        print('file not found: ', file)
        return {}
    end
    lines = {}
    for line in io.lines(file) do 
        lines[#lines + 1] = line
    end
    return lines
end
