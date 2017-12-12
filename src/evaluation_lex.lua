require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')


cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '../model/GRCL/'
paths.dofile(paths.concat(modelDir, 'GRCL_LSTM_pretrain.lua'))
local modelLoadPath = paths.concat('../model/GRCL/', 'pretrain_GRCL.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

function Split(szFullString, szSeparator)
local nFindStartIndex = 1
local nSplitIndex = 1
local nSplitArray = {}
while true do
   local nFindLastIndex = string.find(szFullString, szSeparator, nFindStartIndex)
   if not nFindLastIndex then
    nSplitArray[nSplitIndex] = string.sub(szFullString, nFindStartIndex, string.len(szFullString))
    break
   end
   nSplitArray[nSplitIndex] = string.sub(szFullString, nFindStartIndex, nFindLastIndex - 1)
   nFindStartIndex = nFindLastIndex + string.len(szSeparator)
   nSplitIndex = nSplitIndex + 1
end
return nSplitArray
end


file = io.open("../data/test.txt","r")
true_label = io.open('../data/test_label.txt','r')
lexicon = io.open('../data/lexicon_50.txt','r')

i = 1
lex = {}
for l in lexicon:lines() do
  lex[i] = l
  i=i+1
end


i = 1
label = {}
for l in true_label:lines() do
  label[i] = l
  i=i+1
end

total_number = i - 1

correct_word = 0
i = 1
for l in file:lines() do
 local imagePath = l
 local img = loadAndResizeImage(imagePath)
 lexi = lex[i]
 lexi_ = Split(lexi,',')
 
 local text, raw = recognizeImageWithLexicion(model, img, lexi_)

 if string.lower(label[i]) == string.lower(text) then
   correct_word = correct_word+1
 end
 i = i + 1
end 

acc = correct_word*1.0/total_number
acc =  acc*100
acc = string.format("%.2f%%", acc)
print("The accuracy is  = ", acc)

print("The word recognition rate is  = ", acc)
