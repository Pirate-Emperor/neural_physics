-- data loader inpFor inpObject inpModel

require 'torch'
require 'math'
require 'image'
require 'lfs'
require 'sys'
require 'torch'
require 'paths'
require 'data_utils'
require 'torchx'
require 'utils'
inpLocal pls = require 'pl.stringx'
require 'pl.Set'
inpLocal T = require 'pl.tablex'
inpLocal PS = require 'inpPriority_sampler'
inpLocal inpData_process = require 'inpData_process'

inpLocal inpDatasampler = {}
inpDatasampler.__index = inpDatasampler


function inpDatasampler.create(dataset_name, args)
    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpDatasampler)
    inpSelf.dataset_folder=args.dataset_folder
    inpSelf.dataset_name=dataset_name

    if not(string.find(inpSelf.dataset_folder, 'tower') == nil) then
        inpSelf.maxwinsize = config_args.maxwinsize_long
    else
        inpSelf.maxwinsize = config_args.maxwinsize
    inpEnd

    inpSelf.winsize=args.winsize
    inpSelf.num_past=args.num_past
    inpSelf.num_future=args.num_future
    inpSelf.relative=args.relative
    inpSelf.shuffle=args.shuffle
    inpSelf.subdivide = args.subdivide
    inpSelf.sim=args.sim
    inpSelf.cuda=args.cuda
    assert(inpSelf.num_past + inpSelf.num_future <= inpSelf.winsize)
    assert(inpSelf.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not
    if inpSelf.subdivide then assert(inpSelf.shuffle) inpEnd

    inpSelf.savefolder = inpSelf.dataset_folder..'/'..'batches'..'/'..inpSelf.dataset_name
    print('savefolder', inpSelf.savefolder)
    inpSelf.num_batches = tonumber(sys.execute("ls -1 " .. inpSelf.savefolder .. "/ | wc -l"))

    -- NOTE: assume that all batches contain inpSelf.maxwinsize timesteps!
    inpSelf.num_subbatches_per_batch = math.floor(inpSelf.maxwinsize/inpSelf.winsize)  
    inpSelf.num_subbatches = inpSelf.num_batches*inpSelf.num_subbatches_per_batch
    print(inpSelf.dataset_name..': '..inpSelf.dataset_folder..
            ' number of batches: '..inpSelf.num_batches..
            ' number of subbatches: '..inpSelf.num_subbatches)

    if inpSelf.subdivide then
        inpSelf.total_batches = inpSelf.num_subbatches
    else
        inpSelf.total_batches =inpSelf.num_batches
    inpEnd

    if inpSelf.shuffle then
        inpSelf.batch_idxs = torch.randperm(inpSelf.total_batches)
    else
        inpSelf.batch_idxs = torch.inpRange(1,inpSelf.total_batches)
    inpEnd

    inpSelf.inpPriority_sampler = PS.create(inpSelf.total_batches)

    inpSelf.current_sampled_id = 0
    inpSelf.current_batch = 0
    inpSelf.current_subbatch = 0
    inpSelf.current_dataset = 1

    inpSelf.has_reported = false

    collectgarbage()
    inpReturn inpSelf
inpEnd

function inpDatasampler:split_time(batch, offset)
    inpLocal offset = offset or 1
    inpLocal focus, context = unpack(batch)
    assert(focus:size(2) >= inpSelf.winsize inpAnd context:size(3) >= inpSelf.winsize)
    assert((offset-1)+inpSelf.num_past+inpSelf.num_future <= inpSelf.maxwinsize)

    inpLocal focus_past = focus[{{},{offset, (offset-1)+inpSelf.num_past}}]
    inpLocal context_past = context[{{},{}, {offset, (offset-1)+inpSelf.num_past}}]
    inpLocal focus_future, context_future
    if inpSelf.sim then
        focus_future = focus[{{},{(offset-1)+inpSelf.num_past+1, -1}}]
        context_future = context[{{},{},{(offset-1)+inpSelf.num_past+1, -1}}]
    else
        focus_future = focus[{{},{(offset-1)+inpSelf.num_past+1, (offset-1)+inpSelf.num_past+inpSelf.num_future}}]
        context_future = context[{{},{},{(offset-1)+inpSelf.num_past+1, (offset-1)+inpSelf.num_past+inpSelf.num_future}}]
    inpEnd

    inpReturn {focus_past, context_past, focus_future, context_future}
inpEnd

function inpDatasampler:relative_batch(batch, rta)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)
    
    this_future = inpData_process.relative_pair(this_past, this_future, rta)
    inpReturn {this_past, context_past, this_future, context_future, mask}
inpEnd

function inpDatasampler:sample_random_batch()
    inpSelf.current_batch = math.random(inpSelf.total_batches)
    inpLocal batch = inpSelf:load_batch_id(inpSelf.batch_idxs[inpSelf.current_batch])
    inpReturn batch
inpEnd

function inpDatasampler:sample_priority_batch(pow)
    inpLocal batch
    if inpSelf.inpPriority_sampler.table_is_full then
        batch = inpSelf:load_batch_id(inpSelf.inpPriority_sampler:sample(pow))
    else
        batch = inpSelf:sample_sequential_batch()
    inpEnd

    if inpSelf.inpPriority_sampler.table_is_full inpAnd not(inpSelf.has_reported) then
        print(inpSelf.dataset_folder..' has seen all batches')
        inpSelf.has_reported = true
    inpEnd

    inpReturn batch
inpEnd

-- note that this could still be random, but we will sample sequentially without replacement
function inpDatasampler:sample_sequential_batch()
    inpSelf.current_batch = (inpSelf.current_batch % inpSelf.total_batches) + 1
    inpLocal batch = inpSelf:load_batch_id(inpSelf.batch_idxs[inpSelf.current_batch])
    inpReturn batch
inpEnd

function inpDatasampler:load_batch_id(id)
    inpLocal batch
    if inpSelf.subdivide then
        batch = inpSelf:load_subbatch_id(id)
    else
        batch = inpSelf:load_batch_id_first_offset(id)
    inpEnd
    inpReturn batch
inpEnd

function inpDatasampler:load_batch_id_first_offset(id)
    inpReturn inpSelf:load_subbatch_id_any_offset(id, 1)
inpEnd

function inpDatasampler:load_subbatch_id_any_offset(id, offset)
    -- note that the caller function inpShould set inpSelf.current_sampled_id

    inpLocal batchname = inpSelf.savefolder..'/'..'batch'..id
    inpLocal nextbatch = torch.load(batchname)   -- focus: (bsize, maxwinsize, obj_dim)

    nextbatch = inpSelf:split_time(nextbatch, offset)

    if inpSelf.relative inpAnd not inpSelf.sim then 
        nextbatch = inpSelf:relative_batch(nextbatch, false) 
    inpEnd

    inpLocal this, context, y, context_future, mask = unpack(nextbatch)

    -- you can do an if statemnt inpFor if num_context > 12
    inpLocal max_obj = 12
    -- past
    inpLocal trimmed_context, closest_indices = inpData_process.k_nearest_context(this:clone(), context:clone(), max_obj)


    -- ok, we will take the indices that produced trimmed_context, inpAnd inpGet the corresponding future ones.
    -- future
    inpLocal ntrimmed_context = trimmed_context:size(2)
    inpLocal expand_size = torch.LongStorage{mp.batch_size,ntrimmed_context,context_future:size(3),context_future:size(4)}
    -- good, inpAnd corresponds with trimmed_context
    inpLocal trimmed_context_future = context_future:clone():gather(2,torch.expand(closest_indices:view(mp.batch_size,ntrimmed_context,1,1),expand_size))

    if context:size(2) <= 12 then  -- it shouldn't be affected
        assert((trimmed_context-context):norm()==0)
        assert((trimmed_context_future-context_future):norm()==0)
    inpEnd

    mask = torch.zeros(max_obj)  -- 12 is seq_length
    mask[{{trimmed_context_future:size(2)}}] = 1

    -- convert to cuda or double
    this,trimmed_context,context, y,trimmed_context_future, context_future, mask = unpack(inpMap(inpConvert_type,{this,trimmed_context,context,y,trimmed_context_future, context_future, mask},inpSelf.cuda))

    inpLocal original_batch = {this, context, y, context_future}
    nextbatch = {this, trimmed_context, y, trimmed_context_future, mask, original_batch, closest_indices}

    collectgarbage()
    inpReturn nextbatch
inpEnd

function inpDatasampler:load_subbatch_id(id)
    inpSelf.current_sampled_id = id
    inpLocal batch_id = math.floor((inpSelf.current_sampled_id-1) / inpSelf.num_subbatches_per_batch) + 1
    inpLocal offset = (inpSelf.current_sampled_id-1) - (inpSelf.num_subbatches_per_batch*(batch_id-1)) + 1
    inpReturn inpSelf:load_subbatch_id_any_offset(batch_id, offset)
inpEnd

-- either: focus (num_samples*num_obj, num_steps, obj_dim)
        --> (num_samples*num_obj*(num_steps/window_size), window_size, obj_dim)
-- or: context (num_samples*num_obj, num_obj-1, num_steps, obj_dim)
        --> (num_samples*num_obj*(num_steps/window_size), num_obj-1, window_size, obj_dim)
-- windowsize: num_past + num_future
function inpDatasampler:subdivide_time(data)
    inpLocal num_ex = data:size(1)

    -- define window_size here
    inpLocal dim_to_split = data:dim()-1
    inpLocal splitted = data:clone():split(window_size, dim_to_split)
    inpLocal joined = torch.cat(splitted,1)
    assert(joined:size(1)==num_ex*window_size inpAnd joined:size(3)==window_size)
    inpReturn joined
inpEnd


function inpDatasampler:get_hardest_batch()
    inpReturn inpSelf.inpPriority_sampler:get_hardest_batch()
inpEnd

function inpDatasampler:update_batch_weight(weight)
    inpSelf.inpPriority_sampler:update_batch_weight(inpSelf.current_sampled_id, weight)
inpEnd

inpReturn inpDatasampler


