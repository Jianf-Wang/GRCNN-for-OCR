local Sequential = nn.Sequential
local Convolution = cudnn.SpatialConvolution
local ReLU = cudnn.ReLU
local Avg = cudnn.SpatialAveragePooling
local Max = nn.SpatialMaxPooling
local Identity = nn.Identity
local Concat = nn.Concat
local ConcatTable = nn.ConcatTable
local Parallel = nn.ParallelTable
local Add = nn.CAddTable
local Dropout = nn.Dropout
local Prod = nn.CMulTable
local BN = nn.SpatialBatchNormalization
local SelectTable = nn.SelectTable


function getConfig()
    local config = {
        nClasses         = 36,
        maxT             = 26,
        displayInterval  = 1000,
        testInterval     = 10000,
        nTestDisplay     = 20,
        trainBatchSize   = 192,
        valBatchSize     = 64,
        snapshotInterval = 10000,
        maxIterations    = 300000,
        optimMethod      = optim.adadelta,
        optimConfig      = {},
        trainSetPath     = '/home/images/OCR/lmdb_syn90_train/data.mdb',
        valSetPath     = '/home/images/OCR/lmdb_syn90_test/data.mdb',
    }
    return config
end

function createModel(config)
    local nc = config.nClasses               -- number of class
    local nl = nc + 1            
    local nt = config.maxT                   -- max length of words

    function convRelu(nIn,nOut,ks,ss,ps,batchNormalization)
        batchNormalization = batchNormalization or false
        local subModel = nn.Sequential()
        local conv = cudnn.SpatialConvolution(nIn, nOut, ks, ks, ss, ss, ps, ps)
        subModel:add(conv)
        if batchNormalization then
            subModel:add(nn.SpatialBatchNormalization(nOut))
        end
        subModel:add(cudnn.ReLU(true))
        return subModel
    end


    function GRCL(nIn, nOut, nIter, fSiz, rSiz, pad, share)
        nIter = nIter or 3
        fSiz = fSiz or 3
        rSiz = rSiz or 3

        function getBlock_(nIn, nOut, siz, pad)
          return Convolution(nIn, nOut, siz, siz, 1, 1, pad, pad)
        end
       
        local nets = {}
        local rBlock1
        local rBlock1_res
        

        for i = 1, nIter do
          local net = Sequential()
          local rec = Sequential()
          local rec_res = Sequential()
          local concat_rec0 = ConcatTable()
          local concat_rec1 = ConcatTable()
          local rBlock = getBlock_(nOut, nOut, 3, 1)
          local rBlock_res = getBlock_(nOut, nOut, 1, 0)
          local seq00 = Sequential()
          local concat0 = ConcatTable()
          local parall0 = Parallel()
          local concat1 = ConcatTable()
          local parall1 = Parallel()
          local gate_  = Sequential()
          local gate_1 = Sequential()
          local gate_2 = Sequential()

-- get the last state
          if i == 1 then
            rec:add(ReLU(true))
            if share then
             rBlock1 = rBlock
             rBlock1_res = rBlock_res
            end
          else
            rec:add(nets[i - 1])
          end

          concat_rec0:add(Identity()):add(Identity())
          if i==1 then
            seq00:add(SelectTable(1)):add(rec):add(concat_rec0)
          else
            seq00:add(rec):add(concat_rec0)
          end

-- recurrent weight sharing
        if share then
          if torch.typename(rBlock) == 'cudnn.SpatialConvolution' then
               rBlock:share(rBlock1, 'weight', 'bias', 'gradWeight', 'gradBias')
          else
               for j = 1, #rBlock do
                  if torch.typename(rBlock:get(j)) == 'cudnn.SpatialConvolution' then
                     rBlock:get(j):share(rBlock1:get(j), 'weight', 'bias', 'gradWeight', 'gradBias')
                  end
               end
          end

-- gate weight sharing
          if torch.typename(rBlock_res) == 'cudnn.SpatialConvolution' then
              rBlock_res:share(rBlock1_res, 'weight', 'bias', 'gradWeight', 'gradBias')
          else
               for j = 1, #rBlock_res do
                  if torch.typename(rBlock_res:get(j)) == 'cudnn.SpatialConvolution' then
                     rBlock_res:get(j):share(rBlock1_res:get(j), 'weight', 'bias', 'gradWeight', 'gradBias')
                  end
               end
          end
        end


          local concatz = ConcatTable()
          concatz:add(seq00):add(SelectTable(2))
          concat0:add(SelectTable(1)):add(concatz)

          parall1:add(Sequential():add(SelectTable(2)):add(rBlock_res):add(BN(nOut))):add(Identity())
          concat1:add(Sequential():add(SelectTable(1)):add(SelectTable(1)):add(rBlock):add(BN(nOut))):add(parall1)
          parall0:add(Identity()):add(concat1)

          gate_:add(Parallel():add(Identity()):add(Identity())):add(Add()):add(nn.Sigmoid())
          gate_1:add(Parallel():add(Identity()):add(gate_)):add(Prod()):add(BN(nOut))
          gate_2:add(Parallel():add(Identity()):add(gate_1)):add(Add()):add(ReLU(true))

          net:add(concat0):add(parall0):add(gate_2)
          table.insert(nets, net)
     end
     
     local fInit = getBlock_(nIn, nOut, fSiz, pad)
     local fInit_res = getBlock_(nIn, nOut, 1, 0)
     local concat = ConcatTable()

     concat:add(Sequential():add(fInit):add(BN(nOut))):add(Sequential():add(fInit_res):add(BN(nOut)))
     return Sequential():add(concat):add(nets[#nets])

     end


   function bidirectionalLSTM(nIn, nHidden, nOut, maxT)
        local fwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0, false)
        local bwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0, true)
        local ct = nn.ConcatTable():add(fwdLstm):add(bwdLstm)
        local blstm = nn.Sequential():add(ct):add(nn.BiRnnJoin(nHidden, nOut, maxT))
        return blstm
   end

    -- model and criterion
    local model = nn.Sequential()
    model:add(nn.Copy('torch.ByteTensor', 'torch.CudaTensor', false, true))
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))
    model:add(convRelu(1,64,3,1,1))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       
    model:add(GRCL(64,64,5,3,3,1, false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(GRCL(64,128,5,3,3,1,false))
    model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0))
    model:add(GRCL(128,256,5,3,3,1, false))    
    model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0))
    model:add(convRelu(256,512,2,1,0, true))
    model:add(nn.View(512, -1):setNumInputDims(3))       
    model:add(nn.Transpose({2, 3}))                      
    model:add(nn.SplitTable(2, 3))
    model:add(bidirectionalLSTM(512, 512, 512, nt))
    model:add(bidirectionalLSTM(512, 512,  nl, nt))
    model:add(nn.SharedParallelTable(nn.LogSoftMax(), nt))
    model:add(nn.JoinTable(1, 1))
    model:add(nn.View(-1, nl):setNumInputDims(1))
    model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    model:cuda()
    local criterion = nn.CtcCriterion()

    return model, criterion
end
