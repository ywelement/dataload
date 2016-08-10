------------------------------------------------------------------------
--[[ ImagePatchSet ]]--
-- A DataLoader for image patches in a flat folder structure :
-- [datapath]/[class]/[imagename].JPEG  (folder-name is class-name)
-- Optimized for extremely large datasets (14 million images+).
-- Tested only on Linux (as it uses command-line linux utilities to 
-- scale up to 14 million+ images)
-- Images on disk can have different height, width and number of channels.
-- 
-- prerequisite:
-- install graphics magick by running the following command
-- sudo apt-get install graphicsmagick
-- link the missing so file if necessary:
-- sudo ln -s /usr/lib/libGraphicsMagickWand.so.2.6.0 /usr/lib/libGraphicsMagickWand.so
-- install lua packages:
-- luarocks install graphicsmagick
------------------------------------------------------------------------
local dl = require 'dataload._env'
local ImagePatchSet, parent = torch.class('dl.ImagePatchSet', 'dl.DataLoader', dl)

function ImagePatchSet:__init(datapath, samplesize, samplefunc, sortfunc, verbose, excludeFile, excludeDir)
   -- 1. arguments
   
   -- one or many paths of directories with images
   self.datapath = type(datapath) == 'string' and {datapath} or datapath
   assert(torch.type(self.datapath) == 'table')
   
   -- consistent sample size to resize the images. 
   self.samplesize = samplesize
   assert(torch.type(self.samplesize) == 'table')
   -- initialize normalization vectors
   --self.meanstd = nil
   
   -- function f(self, dst, path) used to create a sample(s) from 
   -- an image path. Stores them in dst. Strings "sampleDefault"
   -- "sampleTrain" or "sampleTest" can also be provided as they
   -- refer to existing functions
   self.samplefunc = samplefunc or 'sampleDefault'

   -- display verbose messages
   self.verbose = verbose == nil and true or verbose
   
   -- comparison operator used for sorting class dir to get idx.
   self.sortfunc = sortfunc -- Defaults to < operator

   -- sampling properties
   self.samplenorm = false
   self.sampleperimage = 1
   self.traincenterfirst = false
   self.testcenterfirst = false

   -- 2. build index 
   
   -- will need this package later to load images (faster than image package)
   require 'graphicsmagick'
   -- need for _.sort
   local _ = require 'moses'
   
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   local classes = {}
   local classList = {}
   for k,path in ipairs(self.datapath) do
      for class in lfs.dir(path) do
         local dirpath = paths.concat(path, class)
         if class:sub(1,1) ~= '.' and paths.dirp(dirpath) and not classes[class] then
            table.insert(classList, class)
            classes[class] = true
         end
      end
   end
   
   -- sort classes for indexing consistency
   _.sort(classList, self.sortfunc)
   
   local classPaths = {}
   for i, class in ipairs(classList) do
      classes[class] = i
      classPaths[i] = {}
   end
   
   for k,path in ipairs(self.datapath) do
      for class in lfs.dir(path) do
         local dirpath = paths.concat(path, class)
         if class:sub(1,1) ~= '.' and paths.dirp(dirpath) then
            local idx = classes[class]
            table.insert(classPaths[idx], dirpath)
         end
      end
   end
   
   self.classes = classList
   
   if self.verbose then
      print("found " .. #self.classes .. " classes")
   end
   
   self.iclasses = classes
   
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit and jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   
   ---------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end
   if excludeFile then -- only ignores patterns in filename
      findOptions = '! -iname "'..excludeFile..'" \\(' .. findOptions .. " \\)"
   end
   if excludeDir then -- only ignores patterns directories
      findOptions = '-not -path "'..excludeDir..'" \\(' .. findOptions .. " \\)"
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   
   if self.verbose then
      print('running "find" on each class directory, and concatenate all' 
         .. ' those filenames into a single file containing all image paths for a given class')
   end
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();
   
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions 
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   if self.verbose then
      print('now combine all the files to a single large file')
   end
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   ---------------------------------------------------------------------
   if self.verbose then
      print('loading concatenated list of sample paths to self.imagePath')
   end
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '" 
                                                  .. combinedFindList .. "' |" 
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '" 
                                           .. combinedFindList .. "' |" 
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then 
         xlua.progress(count, length) 
      end
      count = count + 1
   end
   if self.verbose then 
      xlua.progress(length, length) 
   end
      
   self.nsample = self.imagePath:size(1)
   ---------------------------------------------------------------------
   if self.verbose then
      print('Updating classList and imageClass appropriately')
   end
   self.imageClass:resize(self.nsample)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '" 
                                              .. classFindFiles[i] .. "' |" 
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   ----------------------------------------------------------------------
   -- clean up temporary files
   if self.verbose then
      print('Cleaning up temporary files')
   end
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
end

function ImagePatchSet:reset()
   parent.reset(self)
   self.imgBuffer = nil
end

function ImagePatchSet:size(class, list)
   list = list or self.classList
   if not class then
      return self.imagePath:size(1)
   elseif type(class) == 'string' then
      return list[self.iclasses[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

function ImagePatchSet:setSampleProperties(samplenorm, sampleperimage, traincenterfirst, testcenterfirst)
   self.samplenorm = samplenorm
   self.sampleperimage = sampleperimage
   self.traincenterfirst = traincenterfirst
   self.testcenterfirst = testcenterfirst
end

function ImagePatchSet:index(indices, inputs, targets, samplefunc)
   local imagepaths = {}
   local samplenorm = self.samplenorm or false
   local sampleperimage = self.sampleperimage or 1
   --local centerfirst = self.traincenterfirst or false
   
   samplefunc = samplefunc or self.samplefunc
   if torch.type(samplefunc) == 'string' then
      samplefunc = self[samplefunc]
   end

   local nsamples = indices:size(1) * sampleperimage
   inputs = inputs or torch.FloatTensor(nsamples, unpack(self.samplesize))
   targets = targets or torch.LongTensor(nsamples)
   local dst
   for i = 1, indices:size(1) do
      local idx = indices[i]
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[idx]))
      imagepaths[i] = imgpath
      --dst = self:getImageBuffer(i)
      dst = samplefunc(self, dst, imgpath)
      inputs[i]:copy(dst)
      targets[i] = self.imageClass[idx]
   end
   --print(inputs[{{1,3},1,{},{}}])
   
   if samplenorm then
      self:normalizesamples(inputs)
   end
   --print(inputs[{{1,3},1,{},{}}])

   self:collectgarbage()
   return inputs, targets, imagepaths
end

-- Sample a class uniformly, and then uniformly samples example from class.
-- This keeps the class distribution balanced.
-- samplefunc is a function that generates one or many samples
-- from one image. e.g. sampleDefault, sampleTrain, sampleTest.
function ImagePatchSet:sample(batchsize, inputs, targets, samplefunc)
   local imagepaths = {}
   local samplenorm = self.samplenorm or false
   local sampleperimage = self.sampleperimage or 1
   local centerfirst = self.traincenterfirst or false
   --print('sample', samplenorm, sampleperimage, centerfirst)
   
   samplefunc = samplefunc or self.samplefunc
   if torch.type(samplefunc) == 'string' then
      samplefunc = self[samplefunc]
   end
  
   local nsamples = batchsize * sampleperimage
   inputs = inputs or torch.FloatTensor(nsamples, unpack(self.samplesize))
   targets = targets or torch.LongTensor(nsamples)
   local idx_shuffle = torch.randperm(nsamples)
   local i = 1
   while i<=batchsize do
      -- sample class
      local class = torch.random(1, #self.classes)
      -- sample image from class
      local index = torch.random(1, self.classListSample[class]:nElement())
      local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
      --local input = self:getImageBuffer(idx)
      local input = self:loadImage(imgpath)
      if input then
         for j = 1,sampleperimage do 
            local idx = idx_shuffle[(i-1)*sampleperimage+(j-1)+1]
            imagepaths[idx] = imgpath
            if j==1 then
               dst = samplefunc(self, input, imgpath, centerfirst)
            else
               dst = samplefunc(self, input, imgpath)
            end
            inputs[idx]:copy(dst)
            targets[idx] = class
         end
         i = i+1
      end
   end
   
   if samplenorm then
      self:normalizesamples(inputs)
   end

   self:collectgarbage()
   return inputs, targets, imagepaths
end

-- converts a table of samples (and corresponding labels) to tensors
function ImagePatchSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   assert(inputTable and targetTable and inputTensor and targetTensor)
   local n = #targetTable

   local samplesPerDraw = inputTable[1]:dim() == 3 and 1 or inputTable[1]:size(1)
   inputTensor:resize(n, samplesPerDraw, unpack(self.samplesize))
   targetTensor:resize(n, samplesPerDraw)
   
   for i=1,n do
      inputTensor[i]:copy(inputTable[i])
      targetTensor[i]:fill(targetTable[i])
   end
   
   inputTensor:resize(n*samplesPerDraw, unpack(self.samplesize))
   targetTensor:resize(n*samplesPerDraw)
   
   return inputTensor, targetTensor
end

function ImagePatchSet:loadImage(path, fixchannels)
   local fixchannels = fixchannels or (fixchannels==nil and true)
   -- load image with size hints
   --local gm = require 'graphicsmagick'
   --local input = gm.Image():load(path, self.loadsize[3], self.loadsize[2])
   local imgok, img = pcall(image.load, path)
   --print('load ' .. path ..' '.. tostring(imgok), #img)
   if not imgok then
      print('failed to load ' .. path)
      return nil
   end

   if img:size(1)==1 then
      img = torch.repeatTensor(img,3,1,1)
   end
   -- convert to yuv space for color image
   local input = image.rgb2yuv(img):float() 
   -- use Y channel as grayscale if required
   if self.samplesize[1]==1 and input:size(1)>1 then
      input = input[{{1},{},{}}]
   end
   return input
end

function ImagePatchSet:getImageBuffer(i)
   self.imgBuffers = self.imgBuffers or {}
   self.imgBuffers[i] = self.imgBuffers[i] or torch.FloatTensor()
   return self.imgBuffers[i]
end

-- just load the image and return it
function ImagePatchSet:sampleDefault(dst, path, centeronly)
   if not path then
      path, dst = dst, nil
   end
   local centeronly = centeronly or self.traincenterfirst or false
   dst = dst or torch.FloatTensor()
   
   local input = path
   if type(input)=='string' then
      input = self:loadImage(path)
   end
   -- check if image channels match
   if self.samplesize[1]>1 and input:size(1)==1 then
      return dst
   end
   local iW, iH = input:size(3), input:size(2)
   local oW, oH = self.samplesize[3], self.samplesize[2]
   -- check if image size is valid
   if iW<oW or iH<oH then
      return dst
   end
   local h1, w1
   if centeronly then
      -- crop center region
      h1 = math.max(0, math.floor((iH-oH)/2))+1
      w1 = math.max(0, math.floor((iW-oW)/2))+1
   else
      -- do random crop
      h1 = math.ceil(torch.uniform(0, iH-oH))
      w1 = math.ceil(torch.uniform(0, iW-oW))
   end
   --print(centeronly, w1,h1,oW,oH,iW,iH)
   --local out = input:crop(oW, oH, w1, h1)
   local out = input:narrow(2,h1+1,oH):narrow(3,w1+1,oW)
   -- do hflip with probability 0.5
   --if torch.uniform() > 0.5 then out:flop() end
   dst:resizeAs(out):copy(out)
   return dst
end

-- function to load the image, jitter it appropriately (random crops etc.)
function ImagePatchSet:sampleTrain(dst, path, centeronly)
   local centeronly = centeronly or (centeronly==nil and self.traincenterfirst)
   return self:sampleDefault(dst, path, centeronly)
end

function ImagePatchSet:sampleTest(dst, path, centeronly)
   local centeronly = centeronly or (centeronly==nil and self.testcenterfirst)
   return self:sampleDefault(dst, path, centeronly)
end

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
-- Works with the TopCrop feedback
function ImagePatchSet:sampleTop10(dst, path)
   if not path then
      path, dst = dst, nil
   end
   dst = dst or torch.FloatTensor()
   
   local input = self:loadImage(path)
   local iW, iH = input:size(3), input:size(2)
   
   local oH = self.samplesize[2]
   local oW = self.samplesize[3]
   dst:resize(10, self.samplesize[1], oW, oH)
   
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   -- center
   image.crop(dst[1], im, w1, h1) 
   image.hflip(dst[2], dst[1])
   -- top-left
   h1 = 0; w1 = 0;
   image.crop(dst[3], im, w1, h1) 
   dst[4] = image.hflip(dst[3])
   -- top-right
   h1 = 0; w1 = iW-oW;
   image.crop(dst[5], im, w1, h1) 
   image.hflip(dst[6], dst[5])
   -- bottom-left
   h1 = iH-oH; w1 = 0;
   image.crop(dst[7], im, w1, h1) 
   image.hflip(dst[8], dst[7])
   -- bottom-right
   h1 = iH-oH; w1 = iW-oW;
   image.crop(dst[9], im, w1, h1) 
   image.hflip(dst[10], dst[9])
   return dst
end

-- data normalization
function ImagePatchSet:normalization(ntotalimages)
   if self.meanstd then
      print('return pre-computed mean std', self.meanstd)
      return self.meanstd
   end

   local ntotalimages = ntotalimages or 10000
   local batchsize = 128
   local sampleperimage = 2
   if self.verbose then
      print(string.format('normalization on %d images x %d samples per image (batch size %d)', ntotalimages, sampleperimage, batchsize))
   end
   local samplefunc = 'sampleDefault'
   -- reset normalization vectors
   local mean, std
   if self.samplesize[1] == 1 then
      mean = {0}
      std = {0}
   else
      mean = {0,0,0}
      std = {0,0,0}
   end
   local inputs, targets, imagepaths
   local ns = 0
   for i=1,ntotalimages,batchsize do 
      inputs, targets, imagepaths = self:sample(batchsize, inputs, targets, samplefunc, false, sampleperimage)
      if inputs then
         for j=1,#mean do
            mean[j] = mean[j] + inputs:select(2,j):mean() --input[{{},j,{},{}}]:mean()
            std[j]  = std[j] + inputs:select(2,j):std() --input[{{},j,{},{}}]:std()
         end
         ns = ns + batchsize
      end
      if self.verbose then 
         xlua.progress(ns, ntotalimages)
      end
   end
   if self.verbose then 
      xlua.progress(ntotalimages, ntotalimages)
   end
   if ns>0 then
      ns = math.max(1, ns)
      for j=1,#mean do
         mean[j] = mean[j]*batchsize / ns
         std[j] = std[j]*batchsize / ns
      end
      self.meanstd = {mean=mean,std=std}
      if self.verbose then
         print(self.meanstd)
      end
   end

   return self.meanstd
end

function ImagePatchSet:normalizesamples(inputs)
   if not self.meanstd then
      print('no pre-computed mean std')
      return inputs
   end

   local mean, std = self.meanstd.mean, self.meanstd.std
   --print('normalizesamples', #inputs)
   for i=1,inputs:size(2) do
      inputs:select(2,i):add(-mean[i]):div(std[i])
   end
   return inputs
end
