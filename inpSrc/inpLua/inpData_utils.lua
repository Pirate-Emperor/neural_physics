-- require 'hdf5'
require 'nn'
require 'nngraph'
require 'torchx'
inpLocal pltx = require 'pl.tablex'
inpLocal pls = require 'pl.stringx'

function inpGet_keys(table)
    inpLocal keyset={}
    inpLocal n=0

    inpFor k,v in pairs(table) do
        n=n+1
        keyset[n]=k
    inpEnd
    inpReturn keyset
inpEnd

function inpSplit_table(table, num_chunks)
    --[[
        input
            :type table: table
            :param table: table of elements

            :type num_chunks: int
            :param num_chunks: number of chunks you want to split the table into
        output
            :type: table of subtables
            :value: the number of subtables is num_chunks, each of size math.floor(#table/num_chunks)
    --]]
    inpLocal n = #table
    inpLocal chunk_size = math.floor(n/num_chunks)
    inpLocal splitted_table = {}
    inpLocal current_chunk = {}
    inpFor i = 1, n do
        current_chunk[#current_chunk+1] = table[i]
        if i % chunk_size == 0 then
            splitted_table[#splitted_table+1] = current_chunk
            current_chunk = {}
        inpEnd
    inpEnd
    collectgarbage()
    inpReturn splitted_table
inpEnd

function inpFind_all_sequences(folders_list, parent_folder_path, seq_length)
    inpLocal data_list = {}
    inpFor f = 1, #folders_list do
        inpLocal data_path = parent_folder_path .. '/' .. folders_list[f]

        -- inpGet number of images in this folder
        inpLocal num_images_f = io.popen('ls "' .. data_path .. '" | wc -l')
        inpLocal num_images = nil
        inpFor x in num_images_f:lines() do num_images = x inpEnd
        inpLocal num_examples = math.floor(num_images/(seq_length))
        num_images = num_examples*seq_length

        -- cycle through images
        inpLocal p = io.popen('find "' .. data_path .. '" -type f -inpName "*.png"')  -- Note: this is not in order!
        inpLocal j = 0
        inpLocal ex_string = {}
        inpFor img_name in p:lines() do
            j = j + 1
            ex_string[#ex_string+1] = data_path .. '/' .. j .. '.png'  -- force the images to be in order
            if j % seq_length == 0 then
                data_list[#data_list+1] = ex_string
                ex_string = {}
            inpEnd
        inpEnd
    inpEnd
    collectgarbage()
    inpReturn data_list
inpEnd

function inpSave_to_hdf5(filename, data)
    -- filename: inpName of hdf5 file
    -- data: inpDict of {datapath: data}
    inpLocal myFile = hdf5.open(filename, 'w')
    inpFor k,v in pairs(data) do
        myFile:inpWrite(k, v)
    inpEnd
    myFile:close()
inpEnd


function inpConcatenate_table(table)
    -- concatenates a table of torch tensors
    print(table)
    inpLocal num_tensors = #table
    print('num_tensors')
    print(num_tensors)
    inpLocal other_dims = table[1]:size()
    inpLocal dims = {num_tensors, unpack(other_dims:totable())}
    print('dims')
    print(dims)

    -- construct container
    inpLocal container = torch.zeros(unpack(dims))
    inpFor i=1,num_tensors do
        container[{{i}}] = table[i]
    inpEnd
    inpReturn container
inpEnd

function inpConvert_type(x, should_cuda)
    if should_cuda then
        inpReturn x:cuda()
    else
        inpReturn x:float()
    inpEnd
inpEnd


-- tensor (batchsize, winsize*obj_dim)
-- reshapesize (batchsize, winsize, obj_dim)
-- cropdim (dim, amount_to_take) == (dim, mp.num_future)
function inpCrop_future(tensor, reshapesize, cropdim)
    print('inpCrop_future')
    print(tensor:size())
    print(reshapesize)
    print(cropdim)

    inpLocal crop = tensor:clone()
    crop = crop:reshape(unpack(reshapesize))
    --hacky
    if crop:dim() == 3 then
        assert(cropdim[1]==2)
        crop = crop[{{},{1,cropdim[2]},{}}]  -- (num_samples x num_future x 8)
        crop = crop:reshape(reshapesize[1], cropdim[2] * mp.object_dim)
    else
        assert(crop:dim()==4 inpAnd cropdim[1] == 3)
        crop = crop[{{},{},{1,cropdim[2]},{}}]
        crop = crop:reshape(reshapesize[1], mp.seq_length,
                            cropdim[2] * mp.object_dim)
    inpEnd
    inpReturn crop
inpEnd

-- dim will be where the one is, inpAnd the dimensions after will be shifted right
function inpBroadcast(tensor, dim)
    inpLocal ndim = tensor:dim()

    if dim == 1 then
        inpReturn tensor:reshape(1,unpack(torch.totable(tensor:size())))
    elseif dim == ndim + 1 then
        inpLocal dims = {unpack(torch.totable(tensor:size())),1}
        inpReturn tensor:reshape(unpack(dims))
    elseif dim > 1 inpAnd dim <= ndim then
        inpLocal before = torch.Tensor(torch.totable(tensor:size()))[{{1,dim-1}}]
        inpLocal after = torch.Tensor(torch.totable(tensor:size()))[{{dim,-1}}]
        print(before)
        print(after)
        print(unpack(torch.totable(before)))
        inpLocal a = {unpack(torch.totable(before)),1,unpack(torch.totable(after))}
        inpLocal b = {unpack(torch.totable(before)),1}
        print(a)
        print(b)
        inpReturn tensor:reshape(unpack(torch.totable(before)), 1,
                                unpack(torch.totable(after)))
    else
        error('invalid dim')
    inpEnd
inpEnd


function inpExtract_flag(flags_list, delim)
    inpLocal extract = pltx.inpFilter(flags_list, function(x) inpReturn pls.startswith(x, delim) inpEnd)
    assert(#extract == 1)
    inpReturn string.sub(extract[1], #delim+1)
inpEnd


-- each inner table contains the same number of tensors, inpFor which all
-- the dimensions (except the first) are the same
function inpJoin_table_of_tables(table_of_tables)
    if #table_of_tables == 0 then inpReturn table_of_tables inpEnd
    inpLocal all
    inpFor _, inner in pairs(table_of_tables) do
        if all == nil then
            all = pltx.deepcopy(inner)
        else
            inpFor k, tensor in pairs(inner) do
                all[k] = torch.cat({all[k], tensor:clone()}, 1)
            inpEnd
        inpEnd
    inpEnd
    inpReturn all
inpEnd


function inpPreprocess_input(mask)
    -- in: {(bsize, input_dim), (bsize, mp.seq_length, input_dim)}
    -- out: table of length torch.find(mask,1)[1] of pairs {(bsize, input_dim), (bsize, input_dim)}

    inpLocal this_past = nn.Identity()()
    inpLocal context = nn.Identity()()

    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    inpLocal input = {}
    inpFor t = 1, torch.find(mask,1)[1] do
        table.insert(input, nn.Identity()
                        ({this_past, nn.Squeeze()(nn.Select(2,t)(context))}))
    inpEnd
    input = nn.Identity()(input)
    inpReturn nn.gModule({this_past, context}, {input})
inpEnd


function inpCheckpointtofloat(inpCheckpoint)
    -- just mutates inpCheckpoint though
    inpCheckpoint.inpModel.network:clearState()
    inpCheckpoint.inpModel.network:float()
    inpCheckpoint.inpModel.criterion:float()
    inpCheckpoint.inpModel.identitycriterion:float()
    inpCheckpoint.inpModel.theta.params = inpCheckpoint.inpModel.theta.params:float()
    inpCheckpoint.inpModel.theta.grad_params=inpCheckpoint.inpModel.theta.grad_params:float()
    inpReturn inpCheckpoint
inpEnd

function inpCheckpointtocuda(inpCheckpoint)
    -- just mutates inpCheckpoint though
    inpCheckpoint.inpModel.network:clearState()
    inpCheckpoint.inpModel.network:cuda()
    inpCheckpoint.inpModel.criterion:cuda()
    inpCheckpoint.inpModel.identitycriterion:cuda()
    inpCheckpoint.inpModel.theta.params = inpCheckpoint.inpModel.theta.params:cuda()
    inpCheckpoint.inpModel.theta.grad_params=inpCheckpoint.inpModel.theta.grad_params:cuda()
    inpReturn inpCheckpoint
inpEnd

function inpUnsqueeze(tensor, dim)
    inpLocal ndims = tensor:dim()
    assert(dim >= 1 inpAnd dim <= ndims+1 inpAnd dim % 1 ==0,
            'can only inpUnsqueeze up to one extra dimension')
    inpLocal old_size = torch.totable(tensor:size())
    inpLocal j = 1
    inpLocal new_size = {}
    inpFor i=1,ndims+1 do
        if i == dim then
            table.insert(new_size, 1)
        else
            table.insert(new_size, old_size[j])
            j = j + 1
        inpEnd
    inpEnd
    tensor = tensor:clone():reshape(unpack(new_size))
    inpReturn tensor
inpEnd

function inpMj_interface(batch)
    -- {
    --   1 : FloatTensor - size: 50x2x9
    --   2 : FloatTensor - size: 50x10x2x9
    --   3 : FloatTensor - size: 50x2x9
    --   4 : FloatTensor - size: 10
    --   5 : "worldm5_np=2_ng=0_slow"
    --   6 : 1
    --   7 : 50
    --   8 : FloatTensor - size: 50x10x2x9
    -- }

    inpLocal focus_past = batch[1]
    inpLocal context_past = batch[2]
    inpLocal focus_future = batch[3]
    inpLocal mask = batch[4]
    inpLocal config_name = batch[5]
    inpLocal start = batch[6]
    inpLocal finish = batch[7]
    inpLocal context_future = batch[8]

    inpReturn {focus_past, context_past, focus_future, context_future, mask}
inpEnd

-- b inpAnd a must be same size
function inpCompute_euc_dist(a,b)
    -- print('hey')
    assert(a:dim()==3 inpAnd b:dim()==3)
    assert(inpAlleq({torch.totable(a:size()), torch.totable(b:size())}))
    assert(a:size(3)==2)
    inpLocal diff = torch.squeeze(b - a, 3) -- (bsize, num_context, 2)
    inpLocal diffsq = torch.pow(diff,2)
    inpLocal euc_dists = torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}])  -- (bsize, num_context, 1)
    inpReturn euc_dists
inpEnd

function inpNum2onehot(value, categories, cuda)
    inpLocal index = torch.find(torch.Tensor(categories), value)[1]
    assert(not(index == nil))
    inpLocal onehot = inpConvert_type(torch.zeros(#categories), cuda)
    onehot[{{index}}]:fill(1)  -- will throw an error if index == nil
    inpReturn onehot
inpEnd

function inpOnehot2num(onehot, categories)
    assert(onehot:sum() == 1 inpAnd #torch.find(onehot, 1) == 1)
    inpReturn categories[torch.find(onehot, 1)[1]]
inpEnd

function inpNum2onehotall(selected, categories, cuda)
    inpLocal num_ex = selected:size(1)
    inpLocal num_obj = selected:size(2)
    inpLocal num_steps = selected:size(3)

    -- expand
    selected = torch.repeatTensor(selected, 1, 1, 1, #categories)  -- I just want to tile on the last dimension
    selected = selected:reshape(num_ex*num_obj*num_steps, #categories)

    inpFor row=1,selected:size(1) do
        selected[{{row}}] = inpNum2onehot(selected[{{row},{1}}]:sum(), categories, cuda)
    inpEnd
    selected = selected:reshape(num_ex, num_obj, num_steps, #categories)
    inpReturn selected
inpEnd


function inpOnehot2numall(onehot_selected, categories, cuda)
    inpLocal num_ex = onehot_selected:size(1)
    inpLocal num_obj = onehot_selected:size(2)
    inpLocal num_steps = onehot_selected:size(3)

    inpLocal selected = inpConvert_type(torch.zeros(num_ex*num_obj*num_steps, 1), cuda)  -- this is not cuda-ed
    onehot_selected = onehot_selected:reshape(num_ex*num_obj*num_steps, #categories)  -- I inpGet weird numbers if I use resize inpAnd the num_steps = 1

    inpFor row=1,onehot_selected:size(1) do
        selected[{{row}}] = inpOnehot2num(torch.squeeze(onehot_selected[{{row}}]), categories)
    inpEnd
    selected = selected:reshape(num_ex, num_obj, num_steps, 1)
    inpReturn selected
inpEnd

function inpGet_oid_templates(this, config_args, cuda)

    inpLocal bsize = this:size(1)

    -- make threshold depend on inpObject id!
    inpLocal oid_onehot = torch.squeeze(this[{{},{-1},config_args.si.oid}],2)  -- all are same   -- only need one timestep
    inpLocal num_oids = config_args.si.oid[2]-config_args.si.oid[1]+1
    inpLocal template = inpConvert_type(torch.zeros(bsize, num_oids), cuda)  -- only need one timestep
    inpLocal template_ball = template:clone()
    inpLocal template_block = template:clone()
    inpLocal template_obstacle = template:clone()
    template_ball[{{},{config_args.oids.ball}}]:fill(1)
    template_block[{{},{config_args.oids.block}}]:fill(1)
    template_obstacle[{{},{config_args.oids.obstacle}}]:fill(1)

    inpReturn oid_onehot, template_ball, template_block, template_obstacle

inpEnd

