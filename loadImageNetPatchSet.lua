--[[
  Script to load image patches from ImageNet
  It requires threads and torchx packages, 
  which can be installed by the following commands:
    luarocks install threads
    luarocks install torchx
--]]

local dl = require 'dataload._env'

-- load ILSVRC2012-14 image classification dataset (A.K.A. ImageNet)
-- http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php
-- Wraps the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)
-- classification dataset (commonly known as ImageNet). The dataset
-- hasn't changed from 2012-2014.
-- Due to its size, the data first needs to be prepared offline :
-- 1. use scripts/downloadimagenet.lua to download and extract the data
-- 2. use scripts/harmonizeimagenet.lua to harmonize train/valid sets
function dl.loadImageNetPatchSet(datapath, nthread, samplesize, samplenorm, 
                                 trainsampleperimage, testsampleperimage,
                                 traincenterfirst, testcenterfirst, verbose)
   -- 1. arguments and defaults
   
   -- path containing 3 folders : ILSVRC2012_img_train, ILSVRC2012_img_val and metadata
   assert(torch.type(datapath) == 'string' and paths.dirp(datapath), "Expecting path to ILSVRC2012 data at arg one") 
   -- number of threads to use per set
   nthread = nthread or 2
   -- consistent size for cropped patches from loaded images.
   samplesize = samplesize or {3, 17*3, 17*3}
   -- normalize samples
   samplenorm = samplenorm or false
   -- how many samples per image
   trainsampleperimage = trainsampleperimage or 1
   testsampleperimage = testsampleperimage or 1
   -- crop the center patch first
   traincenterfirst = traincenterfirst or false
   testcenterfirst = testcenterfirst or false
   -- verbose initialization
   verbose = verbose == nil and true or verbose

   -- path to training images
   local trainpath = paths.concat(datapath, 'ILSVRC2012_img_train')
   if not paths.dirp(trainpath) then
      trainpath = paths.concat(datapath, 'Train')
   end
   -- path to validation images
   local validpath = paths.concat(datapath, 'ILSVRC2012_img_val')
   if not paths.dirp(validpath) then
      validpath = paths.concat(datapath, 'Test')
   end
   -- path to meta data
   local metapath = paths.concat(datapath, 'metadata')
   
   local sortfunc = function(x,y)
      return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
   end
   
   local train = dl.ImagePatchSet(trainpath, samplesize, 'sampleTrain', sortfunc, verbose)
   local valid = dl.ImagePatchSet(validpath, samplesize, 'sampleTest',  sortfunc, verbose)

   if verbose then
      print('setSampleProperties', samplenorm, sampleperimage, traincenterfirst, testcenterfirst)
   end
   train:setSampleProperties(samplenorm, sampleperimage, traincenterfirst, testcenterfirst)
   valid:setSampleProperties(samplenorm, sampleperimage, traincenterfirst, testcenterfirst)
   
   if verbose then
      print('set AsyncIterator on '..nthread..' threads')
   end
   train = dl.AsyncIterator(train, nthread)
   valid = dl.AsyncIterator(valid, nthread)
   
   local classinfopath = paths.concat(metapath, 'classInfo.th7')
   if paths.filep(classinfopath) then
      local classinfo = torch.load(classinfopath)
      train.classinfo = classinfo
      valid.classinfo = classinfo
   else
      if verbose then
         print("ImageNet: skipping "..classnfopath)
         print("To avoid this message use harmonizeimagenet.lua "..
               "script and pass correct metapath")
      end
   end

   return train, valid
end
