
paths.dofile('common.lua')

local movieList = {
        -- filename, outFolder, cropW, cropH
        { 'Big Hero 6.mkv', 'BigHero6', 512, 512 },
        { 'Despicable Me 2.mkv', 'DM2', 512, 512 },
        { 'Her.mkv', 'Her', 512, 512 },
        { 'LesMiserables.mkv', 'LesMiserables', 512, 512 },
        { 'Maleficient.mp4', 'Maleficient', 512, 512 },
        { 'Minions-2015.mkv', 'Minions', 512, 512 },
        { 'Office Space.mkv', 'OfficeSpace', 512, 512 },
        { 'Painted skin.mkv', 'PaintedSkin', 512, 512 },
        { 'SummerWars.mkv', 'SummerWars', 512, 512 },
        { 'TheGrandBudapestHotel.mkv', 'TheGrandBudapestHotel', 512, 512 },
        { 'ThePrincessBride.mkv', 'ThePrincessBride', 512, 512 },
        { 'Tron.mkv', 'Tron', 512, 512 },
        { 'Onii.mp4', 'Onii', 512, 512 },
    }
    
--ffmpeg -ss 5 -i file.mp4 -y -an -f image2 -r 1/5 clip%03d.jpg
--ffmpeg -ss 5 -i file.mp4 -y -an -f image2 -frames 10 clip%03d.jpg
--ffmpeg -f image2 -start_number 543 -i movieOut256/image-%06d.png -r25 movie256.mp4

function extractRandomClip(movieInfo, outDir)
    local seek = math.random(360, 3600)
    
    local movieFile = opt.movieInDir .. movieInfo[1]
    local cropW = movieInfo[3]
    local cropH = movieInfo[4]
    
    if util.fileExists(outDir .. 'cropped/clip001.jpg') then
        print('skipping ' .. outDir .. 'cropped/clip001.jpg')
        return
    end
    
    lfs.mkdir('temp')
    lfs.mkdir(outDir)
    lfs.mkdir(outDir .. 'cropped/')
    
    os.remove('temp/clip001.png')
    os.execute('ffmpeg -ss ' .. seek .. ' -i \"' .. movieFile .. '\" -y -an -f image2 -frames ' .. opt.movieClipLength .. ' temp/clip%03d.png')
    print('extracted clip from ' .. movieInfo[1])
    if util.fileExists('temp/clip001.png') then
        for _,filename in ipairs(util.getFileListRecursive('temp/')) do
            local img = image.load(filename)
            local cropX = img:size()[3] / 2 - 256
            local cropY = img:size()[2] / 2 - 256
            img = image.crop(img, cropX, cropY, cropX + cropW, cropY + cropH)
            img = image.scale(img, 256, 256)
            local outFile = string.gsub(filename, 'temp/', outDir .. 'cropped/')
            outFile = string.gsub(outFile, '.png', '.jpg')
            image.save(outFile, img)
        end
    else
        print('Failed to create clip: ' .. movieInfo[1])
        error()
    end
end

function extractRandomClips(movieInfo, clipCount)
    local baseOutDir = opt.movieClipDir .. movieInfo[2] .. '/'
    lfs.mkdir(baseOutDir)
    for clip = 1, clipCount do
        local outDir = baseOutDir .. 'clip' .. clip .. '/'
        extractRandomClip(movieInfo, outDir)
    end
end

function extractAllClips()
    for _,movie in ipairs(movieList) do
        extractRandomClips(movie, opt.clipsPerMovie)
    end
end

function transformAllClips(modelFilename)
    print('loading model from ' .. modelFilename)
    transformNetwork = torch.load(modelFilename)
    transformNetwork = transformNetwork:cuda()
    transformNetwork:evaluate()
    
    local transformSize = 256
    local moviePairsList = {}
    
    for _,movie in ipairs(movieList) do
        local baseOutDir = opt.movieClipDir .. movie[2] .. '/'
        lfs.mkdir(baseOutDir)
        for clip = 1, opt.clipsPerMovie do
            local clipDir = baseOutDir .. 'clip' .. clip .. '/'
            local inDir = clipDir .. 'cropped/'
            local outDir = clipDir .. opt.styleName .. transformSize .. '/'
            transformClipDirectory(transformNetwork, inDir, outDir, transformSize, transformSize)
            for f = 1, opt.movieClipLength - 1 do
                local filenameA = outDir .. 'clip' .. util.zeroPad(f, 3) .. '.jpg'
                local filenameB = outDir .. 'clip' .. util.zeroPad(f + 1, 3) .. '.jpg'
                table.insert(moviePairsList, filenameA .. '|' .. filenameB)
            end
        end
    end
    
    writeAllLines(opt.imageListFramePairs, moviePairsList)
end

function transformClipDirectory(transformNetwork, dirIn, dirOut, resizeW, resizeH)
    local reflectionPadding = 50
    
    lfs.mkdir(dirOut)
    
    local imageFilenames = util.getFileListRecursive(dirIn)
    for _,filename in ipairs(imageFilenames) do
        local outFilename = string.gsub(filename, dirIn, dirOut)
        if util.fileExists(outFilename) then
            print('skipping ' .. outFilename)
        else
            local img = image.load(filename)
            img = image.scale(img, resizeW, resizeH)
            imgSource = img:clone()
            
            img:add(-0.5)
            img = reflectionPadImage(img, reflectionPadding)
        
            local batchInput = torch.CudaTensor(1, 3, img:size()[2], img:size()[3])
            batchInput[1] = img:clone()
        
            img = img:cuda()
            imgStyled = transformNetwork:forward(batchInput)[1]
            imgStyled = caffeDeprocess(imgStyled)
            
            
            print('saving ' .. outFilename)
            image.save(outFilename, imgStyled)
        end
    end
end

function transformImageDirectory(modelFilename, dirIn, dirOut, cropX, cropY, cropW, cropH, resizeW, resizeH)
    local reflectionPadding = 50
    
    lfs.mkdir(dirOut)
    
    print('loading model from ' .. modelFilename)
    transformNetwork = torch.load(modelFilename)
    transformNetwork = transformNetwork:cuda()
    transformNetwork:evaluate()
    
    local imageFilenames = util.getFileListRecursive(dirIn)
    for _,filename in ipairs(imageFilenames) do
        local img = image.load(filename)
        img = image.crop(img, cropX, cropY, cropX + cropW, cropY + cropH)
        img = image.scale(img, resizeW, resizeH)
        imgSource = img:clone()
        
        img:add(-0.5)
        img = reflectionPadImage(img, reflectionPadding)
    
        local batchInput = torch.CudaTensor(1, 3, img:size()[2], img:size()[3])
        batchInput[1] = img:clone()
    
        img = img:cuda()
        --print('img size: ' .. getSize(img))
        --print(transformNetwork)
        imgStyled = transformNetwork:forward(batchInput)[1]
        imgStyled = caffeDeprocess(imgStyled)
        
        local imgFinal = torch.FloatTensor(3, imgSource:size()[2], imgSource:size()[3] * 2)
        local imgLeft = imgFinal:narrow(3, 1, resizeW)
        local imgRight = imgFinal:narrow(3, 1 + resizeW, resizeW)
        imgLeft:copy(imgStyled)
        imgRight:copy(imgSource)
        
        local outFilename = string.gsub(filename, dirIn, dirOut)
        print('saving ' .. outFilename)
        image.save(outFilename, imgFinal)
    end
end
