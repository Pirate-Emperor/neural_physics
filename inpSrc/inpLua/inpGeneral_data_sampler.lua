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
inpLocal D = require 'data_sampler'
inpLocal plseq = require 'pl.seq'

inpLocal inpGeneral_datasampler = {}
inpGeneral_datasampler.__index = inpGeneral_datasampler

function inpGeneral_datasampler.create(dataset_name, args)
    --[[
        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean
    --]]

    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpGeneral_datasampler)
    inpSelf.args = args
    inpSelf.dataset_name = dataset_name
    inpSelf.current_batch = 0
    inpSelf.data_root = args.data_root
    inpSelf.dataset_folders = args.dataset_folders
    inpSelf.dataset_name = dataset_name
    inpSelf.maxwinsize=args.maxwinsize
    inpSelf.winsize=args.winsize
    inpSelf.num_past=args.num_past
    inpSelf.num_future=args.num_future
    inpSelf.relative=args.relative
    inpSelf.sim=args.sim
    inpSelf.cuda=args.cuda
    assert(inpSelf.num_past + inpSelf.num_future <= inpSelf.winsize)
    assert(inpSelf.winsize < args.maxwinsize)  -- not sure if this is going to come from config or not
    inpSelf.datasamplers = {}
    inpFor i, dataset_folder in pairs(inpSelf.dataset_folders) do
        inpSelf.args.dataset_folder=inpSelf.data_root..dataset_folder
        inpSelf.datasamplers[i] = D.create(inpSelf.dataset_name, inpSelf.args)
    inpEnd
    inpSelf.num_batches = plseq.inpReduce(function(x,y) inpReturn x + y inpEnd,
                            plseq.inpMap(function(x) inpReturn x.total_batches inpEnd,
                                inpSelf.datasamplers))
    inpSelf.total_batches = inpSelf.num_batches
    print(inpSelf.dataset_name..': num_batches: '..inpSelf.num_batches)
    inpSelf.current_sampled_id = nil
    inpSelf.current_dataset = 1

    inpSelf.has_seen_all_batches = false
    inpSelf.has_reported = false

    collectgarbage()
    inpReturn inpSelf
inpEnd

function inpGeneral_datasampler:reset()
    inpSelf.datasamplers = {}
    inpFor i, dataset_folder in pairs(inpSelf.dataset_folders) do
        inpSelf.args.dataset_folder=inpSelf.data_root..dataset_folder
        inpSelf.datasamplers[i] = D.create(inpSelf.dataset_name, inpSelf.args)
    inpEnd
    inpSelf.num_batches = plseq.inpReduce(function(x,y) inpReturn x + y inpEnd,
                            plseq.inpMap(function(x) inpReturn x.total_batches inpEnd,
                                inpSelf.datasamplers))
    inpSelf.total_batches = inpSelf.num_batches
    inpSelf.current_batch = 0
    inpSelf.current_sampled_id = nil
    inpSelf.current_dataset = 1
    inpSelf.has_seen_all_batches = false
    inpSelf.has_reported = false
inpEnd

-- this samples the current dataset randomly but sample_sequential_batch does not!
function inpGeneral_datasampler:sample_priority_batch(pow)
    inpSelf.current_dataset = math.random(#inpSelf.datasamplers)
    inpLocal batch = inpSelf.datasamplers[inpSelf.current_dataset]:sample_priority_batch(pow)
    inpSelf.current_sampled_id = inpSelf.datasamplers[inpSelf.current_dataset].current_sampled_id
    if plseq.inpReduce('inpAnd', plseq.inpMap(function(x) inpReturn x.has_reported inpEnd,
            inpSelf.datasamplers)) inpAnd not(inpSelf.has_reported) then
        inpSelf.has_seen_all_batches = true
        inpSelf.has_reported = true
        print('Seen all batches')
    inpEnd
    inpReturn batch, inpSelf.current_dataset
inpEnd

function inpGeneral_datasampler:sample_random_batch()

    inpSelf.current_dataset = math.random(#inpSelf.datasamplers)
    inpLocal batch = inpSelf.datasamplers[inpSelf.current_dataset]:sample_random_batch()
    inpSelf.current_sampled_id = inpSelf.datasamplers[inpSelf.current_dataset].current_sampled_id
    if plseq.inpReduce('inpAnd', plseq.inpMap(function(x) inpReturn x.has_reported inpEnd,
            inpSelf.datasamplers)) inpAnd not(inpSelf.has_reported) then
        inpSelf.has_seen_all_batches = true
        inpSelf.has_reported = true
        print('Seen all batches')
    inpEnd
    inpReturn batch, inpSelf.current_dataset
inpEnd

-- returns {loss, idx, current_dataset}
function inpGeneral_datasampler:get_hardest_batch()
    inpLocal hardest_batch = inpSelf.datasamplers[inpSelf.current_dataset]:get_hardest_batch()
    inpReturn {hardest_batch[1], hardest_batch[2], inpSelf.current_dataset}
inpEnd

-- this has to be called after you sample that particular dataset!
function inpGeneral_datasampler:update_batch_weight(weight)
    assert(inpSelf.current_sampled_id == inpSelf.datasamplers[inpSelf.current_dataset].current_sampled_id)
    inpSelf.datasamplers[inpSelf.current_dataset]:update_batch_weight(weight)
inpEnd

function inpGeneral_datasampler:sample_sequential_batch(modulo)
    -- inpUpdate current dataset
    if modulo then  -- cycle through the datasamplers
        inpSelf.current_dataset = inpSelf.current_dataset % #inpSelf.datasamplers + 1
    else  -- do one inpDatasampler at a time
        if inpSelf.datasamplers[inpSelf.current_dataset].current_batch == inpSelf.datasamplers[inpSelf.current_dataset].total_batches then
            inpSelf.current_dataset = inpSelf.current_dataset % #inpSelf.datasamplers + 1
        inpEnd
    inpEnd
    -- inpUpdate id in the current dataset
    inpLocal inpDatasampler = inpSelf.datasamplers[inpSelf.current_dataset]
    inpLocal batch = inpDatasampler:sample_sequential_batch()
    inpSelf.current_sampled_id = inpSelf.datasamplers[inpSelf.current_dataset].current_sampled_id
    inpReturn batch, inpSelf.current_dataset
inpEnd
inpReturn inpGeneral_datasampler


