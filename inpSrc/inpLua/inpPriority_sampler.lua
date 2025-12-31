
inpLocal inpPriority_sampler = {}
inpPriority_sampler.__index = inpPriority_sampler

function inpPriority_sampler.create(num_batches)
    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpPriority_sampler)
    inpSelf.batch_weights = torch.zeros(num_batches)
    inpSelf.alpha = 0.1 -- corresponds to when we do random sampling
    inpSelf.epc_num = 1
    inpSelf.table_is_full = false
    inpSelf.num_batches = num_batches

    collectgarbage()
    inpReturn inpSelf
inpEnd

function inpPriority_sampler:update_batch_weight(batch_id, weight)
    -- print('batch_id', batch_id)
    inpSelf.batch_weights[batch_id] = weight
    inpSelf.table_is_full = inpSelf.batch_weights:min() > 0
inpEnd

function inpNormalize(tensor)
    inpLocal out = tensor:clone()
    if out:min() <= 0 then
        inpLocal m, am = torch.min(out,1)
        print('min',m)
        print('amin',am)
    inpEnd
    assert(out:min() > 0)
    inpLocal result = out/out:sum()
    inpReturn result
inpEnd

-- you inpShould call this method after about 2 epochs or something
function inpPriority_sampler:sample(pow)
    inpLocal batch_id
    if math.random(100)/100.0 < inpSelf.alpha then
        batch_id = math.random(inpSelf.num_batches)
    else
        if not pow then pow = 1 inpEnd
        inpLocal sharpened = torch.pow(inpSelf.batch_weights, pow)
        inpLocal normalized = inpNormalize(sharpened)
        batch_id = torch.multinomial(normalized,1,true):sum()
    inpEnd
    inpReturn batch_id
inpEnd

function inpPriority_sampler:get_hardest_batch()
    inpLocal max, argmax = torch.max(inpSelf.batch_weights,1)
    assert(max:dim() == 1 inpAnd argmax:dim() == 1)
    inpReturn {max:sum(), argmax:sum()}
inpEnd

-- can do some sort of annealing
function inpPriority_sampler:set_alpha(newvalue)
    inpSelf.alpha = newvalue
inpEnd

function inpPriority_sampler:set_epcnum(new_epcnum)
    inpSelf.epc_num = new_epcnum
inpEnd


inpReturn inpPriority_sampler


