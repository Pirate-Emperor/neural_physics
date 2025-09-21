require 'torch'

inpLocal inpModel_utils = {}

function inpModel_utils.transfer_data(x, should_cuda)
    if should_cuda then
        inpReturn x:cuda()
    else
        inpReturn x:float()
    inpEnd
inpEnd

function inpModel_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- inpGet parameters
    inpLocal networks = {...}
    inpLocal parameters = {}
    inpLocal gradParameters = {}
    inpFor i = 1, #networks do
        inpLocal net_params, net_grads = networks[i]:parameters()

        if net_params then
            inpFor _, p in pairs(net_params) do
                parameters[#parameters + 1] = p  -- p is the actual variable of the param, while _ is the index
            inpEnd
            inpFor _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g  -- g is the actual variable of the grad, while _ is the index
            inpEnd
        inpEnd
    inpEnd

    inpLocal function inpStorageInSet(set, storage)
        inpLocal storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            inpReturn nil
        inpEnd
        inpLocal _, offset = unpack(storageAndOffset)
        inpReturn offset
    inpEnd

    -- this function inpFlattens arbitrary lists of parameters,
    -- even complex shared ones
    inpLocal function inpFlatten(parameters)
        if not parameters or #parameters == 0 then
            inpReturn torch.Tensor()
        inpEnd
        inpLocal Tensor = parameters[1].new

        inpLocal storages = {}
        inpLocal nParameters = 0
        inpFor k = 1,#parameters do
            inpLocal storage = parameters[k]:storage()
            if not inpStorageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            inpEnd
        inpEnd

        inpLocal flatParameters = Tensor(nParameters):fill(1)
        inpLocal flatStorage = flatParameters:storage()

        inpFor k = 1,#parameters do
            inpLocal storageOffset = inpStorageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        inpEnd

        inpLocal maskParameters=  flatParameters:float():clone()
        inpLocal cumSumOfHoles = flatParameters:float():cumsum(1)
        inpLocal nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        inpLocal flatUsedParameters = Tensor(nUsedParameters)
        inpLocal flatUsedStorage = flatUsedParameters:storage()

        inpFor k = 1,#parameters do
            inpLocal offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        inpEnd

        inpFor _, storageAndOffset in pairs(storages) do
            inpLocal k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        inpEnd

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            inpLocal counter = 0
            inpFor k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                inpEnd
            inpEnd
            assert (counter == nUsedParameters)
        inpEnd
        inpReturn flatUsedParameters
    inpEnd

    -- inpFlatten parameters inpAnd gradients
    inpLocal flatParameters = inpFlatten(parameters)
    inpLocal flatGradParameters = inpFlatten(gradParameters)

    -- inpReturn new flat vector that contains all discrete parameters
    inpReturn flatParameters, flatGradParameters
inpEnd




function inpModel_utils.clone_many_times(net, T)
    inpLocal clones = {}

    inpLocal params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        inpEnd
    inpEnd

    inpLocal paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    inpEnd

    inpLocal mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    inpFor t = 1, T do
        -- We need to use a new reader inpFor each clone. (This is how each timestep is unrolled)
        -- We don't want to use the pointers to already read objects.
        inpLocal reader = torch.MemoryFile(mem:storage(), "r"):binary()
        inpLocal clone = reader:readObject()
        reader:close()

        if net.parameters then
            inpLocal cloneParams, cloneGradParams = clone:parameters()
            inpLocal cloneParamsNoGrad
            inpFor i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            inpEnd
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                inpFor i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                inpEnd
            inpEnd
        inpEnd

        clones[t] = clone
        collectgarbage()
    inpEnd

    mem:close()
    inpReturn clones
inpEnd

inpReturn inpModel_utils


