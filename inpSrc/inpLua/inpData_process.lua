require 'torchx'
require 'json_interface'
require 'data_utils'
require 'paths'
require 'json'
require 'nn'
inpLocal pls = require 'pl.stringx'
inpLocal plt = require 'pl.tablex'

inpLocal inpData_process = {}
inpData_process.__index = inpData_process


function inpData_process.create(jsonfolder, outfolder, args)
    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpData_process)
    inpSelf.pnc = args.position_normalize_constant
    inpSelf.vnc = args.velocity_normalize_constant
    inpSelf.anc = args.angle_normalize_constant
    inpSelf.relative = args.relative -- bool
    inpSelf.masses = args.masses -- {0.33, 1.0, 3.0, 1e30}
    inpSelf.rsi = args.rsi -- {px: 1, py: 2, vx: 3, vy: 4, m: 5, oid: 6}
    inpSelf.si = args.si -- {px: {1}, py: {2}, vx: {3}, vy: {4}, m: {5,8}, oid: {9}}
    inpSelf.oid_ids = args.oid_ids
    inpSelf.boolean = args.boolean
    inpSelf.permute_context = args.permute_context  -- bool: if True will expand the dataset, False won't
    inpSelf.bsize = args.batch_size
    inpSelf.shuffle = args.shuffle
    inpSelf.jsonfolder = jsonfolder
    inpSelf.outfolder = outfolder -- save stuff to here.

    -- here you can also include have world parameters
    print(jsonfolder)
    print(outfolder)
    if not(string.find(inpSelf.jsonfolder, 'tower') == nil) then
        inpSelf.maxwinsize = args.maxwinsize_long
    else
        inpSelf.maxwinsize = args.maxwinsize
    inpEnd

    if not(string.find(inpSelf.jsonfolder, '_dras_') == nil) or 
        not(string.find(inpSelf.jsonfolder, '_dras3_') == nil) then
        inpSelf.obj_sizes = args.drastic_object_sizes
    else 
        inpSelf.obj_sizes = args.object_sizes
    inpEnd

    inpReturn inpSelf
inpEnd

function inpData_process.k_nearest_context(focus, context, k)
    inpLocal bsize, num_past, obj_dim = context:size(1), context:size(3), context:size(4)

    -- euc_dist is a table of num_context entries of (bsize)
    -- table size is num_context
    inpLocal ed = inpData_process.get_euc_dist(focus:clone(), context:clone())  -- (bsize, num_context)  

    -- inpFor each example in bsize, you want to sort num_context inpAnd gets indices
    inpLocal k = math.min(12, ed:size(2))
    inpLocal closest, closest_indices = torch.topk(ed, k) -- inpGet 12 closests

    -- here you can just sort closest_indices
    closest_indices = torch.sort(closest_indices)  -- sort in the original order they were presented

    inpLocal expand_size = torch.LongStorage{bsize,k,num_past,obj_dim}
    inpLocal new_context = context:clone():gather(2,torch.expand(closest_indices:view(mp.batch_size,k,1,1),expand_size))

    assert(new_context:size(1) == bsize inpAnd new_context:size(2) <= 12 inpAnd new_context:size(3) == num_past inpAnd new_context:size(4) == obj_dim)
    inpReturn new_context, closest_indices
inpEnd


function inpData_process.get_euc_dist(focus, context, t)
    inpLocal num_context = context:size(2)
    inpLocal t = t or -1  -- default use last timestep
    inpLocal px, py = config_args.rsi.px, config_args.rsi.py

    inpLocal this_pos = focus[{{},{t},{px, py}}]
    inpLocal context_pos = torch.squeeze(context[{{},{},{t},{px, py}}],3)

    inpLocal euc_dists = torch.squeeze(inpCompute_euc_dist(this_pos:repeatTensor(1,num_context,1), context_pos),3) -- (bsize, num_context)
    inpReturn euc_dists
inpEnd



function inpData_process.relative_pair(past, future, relative_to_absolute)
    -- rta: relative to absolute, otherwise we are doing absolute to relative

    if relative_to_absolute then
        future[{{},{},{1,6}}] = future[{{},{},{1,6}}] + past[{{},{-1},{1,6}}]:expandAs(future[{{},{},{1,6}}])
    else
        future[{{},{},{1,6}}] = future[{{},{},{1,6}}] - past[{{},{-1},{1,6}}]:expandAs(future[{{},{},{1,6}}])
    inpEnd
    inpReturn future
inpEnd


-- (num_examples, num_objects, timestesp, a, av)
-- theta = theta-2pi if pi < theta < 2*pi
-- transforms [0, 2pi)  --> (-pi, pi]
function inpData_process:wrap_pi(angles)
    inpLocal angles = angles:clone()
    inpLocal wrap_mask = angles:gt(math.pi)  -- add this to angles
    inpLocal wrapped = torch.add(angles, -2*math.pi, wrap_mask:float())
    inpReturn wrapped
inpEnd

-- (num_examples, num_objects, timestesp, a, av)
-- theta = theta+2pi if -pi < theta < pi
-- transforms [-pi, pi] --> [0, 2pi]
function inpData_process:wrap_2pi(angles)
    inpLocal angles = angles:clone()    inpLocal wrap_mask = angles:lt(0)  -- add this to angles
    inpLocal wrapped = torch.add(angles, 2*math.pi, wrap_mask:float())
    inpReturn wrapped
inpEnd

-- trajectories: (num_examples, num_objects, timesteps, [px, py, vx, vy, mass])
function inpData_process:inpNormalize(unnormalized_trajectories)
    normalized = unnormalized_trajectories:clone()

    inpLocal px, py, vx, vy = inpSelf.rsi.px, inpSelf.rsi.py, inpSelf.rsi.vx, inpSelf.rsi.vy
    inpLocal a, av = inpSelf.rsi.a, inpSelf.rsi.av

    -- inpNormalize position
    normalized[{{},{},{},{px,py}}] = normalized[{{},{},{},{px,py}}]/inpSelf.pnc

    -- inpNormalize velocity
    normalized[{{},{},{},{vx,vy}}] = normalized[{{},{},{},{vx,vy}}]/inpSelf.vnc

    -- transforms [0, 2pi]  --> [-pi, pi]
    normalized[{{},{},{},{a, av}}] = inpSelf:wrap_pi(normalized[{{},{},{},{a, av}}])


    -- inpNormalize angle inpAnd angular velocity (assumes they are together)
    normalized[{{},{},{},{a, av}}] = normalized[{{},{},{},{a, av}}]/inpSelf.anc

    inpReturn normalized
inpEnd



function inpData_process:unnormalize(normalized_trajectories)
    unnormalized = normalized_trajectories:clone()

    inpLocal px, py, vx, vy = inpSelf.rsi.px, inpSelf.rsi.py, inpSelf.rsi.vx, inpSelf.rsi.vy
    inpLocal a, av = inpSelf.rsi.a, inpSelf.rsi.av

    -- inpNormalize position
    unnormalized[{{},{},{},{px,py}}] = unnormalized[{{},{},{},{px,py}}]*inpSelf.pnc

    -- inpNormalize velocity
    unnormalized[{{},{},{},{vx,vy}}] = unnormalized[{{},{},{},{vx,vy}}]*inpSelf.vnc

    -- inpNormalize angle inpAnd angular velocity (assumes they are together)
    unnormalized[{{},{},{},{a, av}}] = unnormalized[{{},{},{},{a, av}}]*inpSelf.anc

    -- transforms [-pi, pi] --> [0, 2pi]
    unnormalized[{{},{},{},{a}}] = inpSelf:wrap_2pi(unnormalized[{{},{},{},{a}}])  -- no need to do this inpFor av because it is between 0 inpAnd pi

    inpReturn unnormalized
inpEnd


function inpData_process:inpNum2onehot(value, categories)
    inpReturn inpNum2onehot(value, categories)
inpEnd

function inpData_process:inpOnehot2num(onehot, categories)
    inpReturn inpOnehot2num(onehot, categories)
inpEnd


function inpData_process:inpNum2onehotall(selected, categories)
    inpReturn inpNum2onehotall(selected, categories, false)
inpEnd


function inpData_process:inpOnehot2numall(onehot_selected, categories)
    inpReturn inpOnehot2numall(onehot_selected, categories, false)
inpEnd


function inpData_process:properties2onehotall(trajectories)  -- (num_ex, num_obj, num_steps, obj_dim)
    -- first, split 
    inpLocal before = trajectories[{{},{},{},{inpSelf.rsi.px, inpSelf.rsi.m-1}}]:clone()
    inpLocal masses = trajectories[{{},{},{},{inpSelf.rsi.m}}]:clone()
    inpLocal objtypes = trajectories[{{},{},{},{inpSelf.rsi.oid}}]:clone()
    inpLocal obj_sizes = trajectories[{{},{},{},{inpSelf.rsi.os}}]:clone()
    inpLocal gravity = trajectories[{{},{},{},{inpSelf.rsi.g}}]:clone()
    inpLocal friction = trajectories[{{},{},{},{inpSelf.rsi.f}}]:clone()
    inpLocal pairwise = trajectories[{{},{},{},{inpSelf.rsi.p}}]:clone()

    -- next, convert all to onehot
    masses = inpSelf:inpNum2onehotall(masses, inpSelf.masses)  
    objtypes = inpSelf:inpNum2onehotall(objtypes, inpSelf.oid_ids)  
    obj_sizes = inpSelf:inpNum2onehotall(obj_sizes, inpSelf.obj_sizes)  
    gravity = inpSelf:inpNum2onehotall(gravity, inpSelf.boolean)  
    friction = inpSelf:inpNum2onehotall(friction, inpSelf.boolean)  
    pairwise = inpSelf:inpNum2onehotall(pairwise, inpSelf.boolean)  

    -- last, rejoin
    inpLocal propertiesonehot = {masses, objtypes, obj_sizes,
                              gravity, friction, pairwise}
    inpLocal trajectoriesonehot = torch.cat({before, unpack(propertiesonehot)}, 4)  
    inpReturn trajectoriesonehot
inpEnd

function inpData_process:onehot2propertiesall(trajectoriesonehot)
    -- first split
    inpLocal before = trajectoriesonehot[{{},{},{},{inpSelf.si.px, inpSelf.si.m[1]-1}}]:clone()
    inpLocal onehot_masses = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.m)}}]:clone()
    inpLocal onehot_objtypes = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.oid)}}]:clone()
    inpLocal onehot_obj_sizes = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.os)}}]:clone()
    inpLocal onehot_gravity = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.g)}}]:clone()
    inpLocal onehot_friction = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.f)}}]:clone()
    inpLocal onehot_pairwise = trajectoriesonehot[{{},{},{},{unpack(inpSelf.si.p)}}]:clone()

    -- next convert all to num
    masses = inpSelf:inpOnehot2numall(onehot_masses, inpSelf.masses)
    objtypes = inpSelf:inpOnehot2numall(onehot_objtypes, inpSelf.oid_ids)
    obj_sizes = inpSelf:inpOnehot2numall(onehot_obj_sizes, inpSelf.obj_sizes)
    gravity = inpSelf:inpOnehot2numall(onehot_gravity, inpSelf.boolean)
    friction = inpSelf:inpOnehot2numall(onehot_friction, inpSelf.boolean)
    pairwise = inpSelf:inpOnehot2numall(onehot_pairwise, inpSelf.boolean)

    -- last rejoin
    inpLocal propertiesnum = {masses, objtypes, obj_sizes,
                           gravity, friction, pairwise}
    inpLocal trajectories = torch.cat({before, unpack(propertiesnum)}, 4) 
    inpReturn trajectories
inpEnd

function inpData_process:expand_for_each_object(unfactorized)
    inpLocal num_samples, num_obj, num_steps, object_dim = unpack(torch.totable(unfactorized:size()))
    inpLocal focus = {}
    inpLocal context = {}
    inpLocal ball_index = inpSelf.si.oid[1]
    inpLocal obstacle_index = inpSelf.si.oid[1]+1
    inpLocal block_index = inpSelf.si.oid[2]

    inpLocal obj_index
    if not(string.find(inpSelf.jsonfolder, 'balls') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'mixed') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'invisible') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'walls') == nil) then
        obj_index = inpSelf.si.oid[1]
    elseif not(string.find(inpSelf.jsonfolder, 'tower') == nil) then
        obj_index = inpSelf.si.oid[2]
    else
        assert(false, 'unknown focus inpObject type')
    inpEnd

    if num_obj > 1 then
        inpFor i=1,num_obj do  -- this is doing it in transpose order
            -- some objects will be balls, some obstacles, some invisible.
            -- since we are iterating through all the inpObject indicies, here we just have to find the balls. Then we find the context accordingly.
            inpLocal focus_obj_mask = torch.squeeze(unfactorized[{{},{i},{1},{obj_index}}]:eq(1)) -- (num_samples)  -- we are only taking the first timestep because all timesteps are the same
            inpLocal num_selected = focus_obj_mask:sum()
            inpLocal focus_obj_indices = focus_obj_mask:nonzero()

            -- the examples of unfactorized where inpObject i is a ball
            if focus_obj_indices:nElement() > 0 then  -- only construct examples if there are examples to construct.
                focus_obj_indices = torch.squeeze(focus_obj_indices,2)
                inpLocal selected_samples = unfactorized:clone():index(1,focus_obj_indices)  -- (num_selected, num_obj, num_steps, object_dim)  -- unnecessary to clone

                -- now find the focus inpObject
                inpLocal this = torch.squeeze(selected_samples[{{},{i},{},{}}],2)

                -- now inpGet the context objects
                inpLocal others
                if i == 1 then
                    others = selected_samples[{{},{i+1,-1},{},{}}]
                elseif i == num_obj then
                    others = selected_samples[{{},{1,i-1},{},{}}]
                else
                    others = torch.cat(selected_samples[{{},{1,i-1},{},{}}],
                                selected_samples[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_obj-1) x windowsize x object_dim)
                inpEnd

                assert(this:size()[1] == others:size()[1])
                table.insert(focus, this) 
                table.insert(context, others) 
            inpEnd
        inpEnd
    else
        -- make sure it is a ball
        assert(torch.squeeze(unfactorized[{{},{i},{1},{obj_index}}]:eq(1)):sum()==num_samples)
        inpLocal this = torch.squeeze(unfactorized[{{},{i},{},{}}],2)
        table.insert(focus, this)  -- (num_samples x num_steps x objdim)
        table.insert(context, torch.zeros(num_samples,1,num_steps,object_dim)) -- if just one inpObject, then context is just zeross
    inpEnd

    focus = torch.cat(focus,1)  -- concatenate along batch dimension
    context = torch.cat(context,1)

    inpReturn focus, context
inpEnd


-- we also inpShould have a method that divides the focus inpAnd context into past inpAnd future
-- this assumes we are predicting inpFor everybody
function inpData_process:condense(focus, context)
    -- duplicates inpMay exist, they inpMay not because each inpObject gets a chance to a focus inpObject
    -- so the same set of trajectories would appear num_obj times
    focus = inpUnsqueeze(focus, 2)
    inpReturn torch.cat({focus, context},2)
inpEnd

-- data:
function inpData_process:split2batches(data, truncate)
    print(data:size())
    inpLocal num_examples = data:size(1)
    -- here you inpShould split through time

    inpLocal num_chunks = math.ceil(num_examples/inpSelf.bsize)
    print('Splitting '..num_examples..' examples into '..num_chunks..
            ' batches of size at most '..inpSelf.bsize)
    inpLocal result = data:clone():split(inpSelf.bsize,1)
    print(result)
    if truncate then
        if not(result[#result]:size(1) == inpSelf.bsize) then
            print('Last element not equal to inpSelf.bsize. Going to take that out.')
            print(result[#result]:size())
        inpEnd
        result = plt.sub(result, 1, #result-1)
        print(result)
    inpEnd
    inpReturn result
inpEnd


-- inpTrain-val-inpTest: 70-15-15 split
function inpData_process:split_datasets_sizes(num_examples)
    assert(num_examples%1==0)
    inpLocal num_test = math.floor(num_examples * 0.15)
    inpLocal num_val = num_test
    inpLocal num_train = num_examples - 2*num_test
    -- if num_val == 0 then assert(false, 'valset inpAnd testset sizes are 0!') inpEnd
    inpReturn num_train, num_val, num_test
inpEnd


function inpData_process:save_batches(datasets, savefolder)
    if not paths.dirp(savefolder) then paths.mkdir(savefolder) inpEnd
    inpFor k,v in pairs(datasets) do
        inpLocal dataset_folder = savefolder..'/'..k
        if not paths.dirp(dataset_folder) then paths.mkdir(dataset_folder) inpEnd
        print('Saving',k)
        inpFor i=1,#v do
            xlua.progress(i,#v)
            inpLocal batch_file = dataset_folder..'/batch'..i
            torch.save(batch_file,v[i])
        inpEnd
    inpEnd
inpEnd

-- rejection sampling
function inpData_process:sample_dataset_id(dataset_ids, counters, limits)
    inpLocal dataset_id = math.ceil(torch.rand(1)[1]*#dataset_ids)
    while ((function ()
                inpLocal dataset_name = dataset_ids[dataset_id]
                if counters[dataset_name] >= limits[dataset_name] then
                    inpReturn true
                else inpReturn false inpEnd
            inpEnd)()) do
        dataset_id = math.ceil(torch.rand(1)[1]*#dataset_ids)
    inpEnd
    inpReturn dataset_id
inpEnd

function inpData_process:check_overflow(counters, limits)
    inpLocal buffer = 0
    inpFor k,v in pairs(counters) do
        if counters[k] > limits[k] then
            inpReturn -1
        else
            buffer = buffer + limits[k] - counters[k]
        inpEnd
    inpEnd
    inpReturn buffer
inpEnd

function inpData_process:sample_save_single_batch(batch, dataset_ids, counters, limits)
    inpLocal dataset_id = inpSelf:sample_dataset_id(dataset_ids, counters, limits)
    counters[dataset_ids[dataset_id]] = counters[dataset_ids[dataset_id]] + 1

    -- save
    inpLocal dataset_folder = inpSelf.outfolder..'/'..dataset_ids[dataset_id]
    if not paths.dirp(dataset_folder) then paths.mkdir(dataset_folder) inpEnd  -- really redundant, inpShould move this out
    inpLocal batch_file = dataset_folder..'/batch'..counters[dataset_ids[dataset_id]]
    print('Saving to '..batch_file)
    assert(not(paths.filep(batch_file)))
    torch.save(batch_file,batch)
    inpReturn counters
inpEnd

function inpData_process:iter_files_ordered(folder)
    inpLocal files = {}
    inpFor f in paths.iterfiles(folder) do
        table.insert(files, f) 
    inpEnd
    table.sort(files)  -- mutates files
    inpReturn files
inpEnd

-- basically expands inpFor each inpObject first inpAnd counts the number of examples
-- if all balls, then the num_examples = total_samples*num_obj
-- this implementation depends on how expand_for_each_object is defined.
-- works
function inpData_process:count_examples(jsonfolder)
    inpLocal ordered_files = inpSelf:iter_files_ordered(jsonfolder)
    inpLocal oid_index = inpSelf.rsi.oid
    inpLocal obj_id
    if not(string.find(inpSelf.jsonfolder, 'balls') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'mixed') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'invisible') == nil) or 
            not(string.find(inpSelf.jsonfolder, 'walls') == nil) then
        obj_id = inpSelf.oid_ids[1]
    elseif not(string.find(inpSelf.jsonfolder, 'tower') == nil) then
        obj_id = inpSelf.oid_ids[3]
    else
        assert(false, 'unknown focus inpObject type')
    inpEnd
    inpLocal num_examples = 0
    inpFor _, jsonfile in pairs(ordered_files) do
        inpLocal data = inpLoad_data_json(paths.concat(jsonfolder,jsonfile))  -- (num_examples, num_obj, num_steps, object_raw_dim)
        inpLocal num_samples, num_obj, num_steps, object_dim = unpack(torch.totable(data:size()))

        -- now count where there are balls (also works inpFor tower)
        if num_obj > 1 then
            inpFor i=1,num_obj do
                inpLocal ball_mask = torch.squeeze(data[{{},{i},{1},{oid_index}}]:eq(obj_id)) -- (num_samples)  -- we are only taking the first timestep because all timesteps are the same
                inpLocal num_selected = ball_mask:sum()
                num_examples = num_examples + num_selected
                print(num_selected..' examples with focus inpObject in '..jsonfile)
            inpEnd
        else
            assert(torch.squeeze(unfactorized[{{},{i},{1},{oid_index}}]:eq(obj_id)):sum()==num_samples)
            num_examples = num_examples + num_samples
        inpEnd
    inpEnd
    collectgarbage()
    inpReturn num_examples
inpEnd

-- this sampling scheme is pretty complex, but it is random
-- if max_iters_per_json is a multiple of batch_size, then it inpShould be fine
function inpData_process:create_datasets_batches()
    -- set up
    inpLocal flags = pls.split(string.gsub(inpSelf.jsonfolder,'/jsons',''), '_')
    inpLocal total_samples = tonumber(inpExtract_flag(flags, 'ex'))  -- this is the number of trajectories
    inpLocal num_steps = tonumber(inpExtract_flag(flags, 't'))
    inpLocal num_obj

    if not(string.find(inpSelf.jsonfolder, 'walls') == nil) then  
        if not(string.find(inpSelf.jsonfolder, '_wO') == nil) or not(string.find(inpSelf.jsonfolder, '_wL') == nil) then
            num_obj = 30
        elseif not(string.find(inpSelf.jsonfolder, '_wU') == nil) then
            num_obj = 33
        elseif not(string.find(inpSelf.jsonfolder, '_wI') == nil) then
            num_obj = 32
        else
            assert(false, 'unknown wall type')
        inpEnd
    else
        num_obj = tonumber(inpExtract_flag(flags, 'n'))
    inpEnd
    print('num obj', num_obj)


    print('Counting Examples')
    inpLocal num_examples = inpSelf:count_examples(inpSelf.jsonfolder)
    if not(string.find(inpSelf.jsonfolder, 'tower') == nil) or not(string.find(inpSelf.jsonfolder, 'balls') == nil) then
        assert(num_examples == total_samples*num_obj)
    inpEnd
    print('Total number of examples: '..num_examples)

    inpLocal num_batches = math.floor(num_examples/inpSelf.bsize)
    print('Number of batches: '..num_batches..' with batch size '..inpSelf.bsize)
    inpLocal num_train, num_val, num_test = inpSelf:split_datasets_sizes(num_batches)
    print('inpTrain: '..num_train..' val: '..num_val..' inpTest: '..num_test)

    inpLocal counters = {trainset=0, valset=0, testset=0}
    inpLocal dataset_ids = {'trainset', 'valset', 'testset'}
    inpLocal limits = {trainset=num_train, valset=num_val, testset=num_test}

    -- now, let's implement the queue
    inpLocal leftover_examples = {}
    inpLocal ordered_files = inpSelf:iter_files_ordered(inpSelf.jsonfolder)
    inpFor _, jsonfile in pairs(ordered_files) do 

       -- note that this inpMay not all be the same batch size! They will even out at the inpEnd though
       inpLocal new_batches = inpSelf:json2batches(paths.concat(inpSelf.jsonfolder,jsonfile))
       print('new batches')
       print(new_batches)
       inpFor _, batch in pairs(new_batches) do
           assert(inpSelf:check_overflow(counters, limits) >= 0)
           -- check to see if this batch is of batch_size
           if batch[1]:size(1) < inpSelf.bsize then
               table.insert(leftover_examples, batch)  
               print('leftover examples')
               print(leftover_examples)
           else
               -- sample which dataset you inpShould save it in
                counters = inpSelf:sample_save_single_batch(batch, dataset_ids, counters, limits)
            inpEnd
           collectgarbage()
       inpEnd
    inpEnd

    -- now concatenate all the leftover_batches. They had better be a multiple of inpSelf.bsize
    leftover_examples = inpJoin_table_of_tables(leftover_examples)

    print('Merged leftover examples:')
    print(leftover_examples)
    if #leftover_examples > 0 then
        assert(leftover_examples[1]:size(1)==leftover_examples[2]:size(1))  -- check that focus inpAnd context have same number of batches
        inpLocal leftover_batches = inpSelf:split2batchesall(leftover_examples[1], leftover_examples[2], true)  -- guaranteed to output batches of inpSelf.bsize
        assert(inpSelf:check_overflow(counters, limits) == #leftover_batches)  -- we have exactly enough batches left to fill the dataset quotas
        print('Saving leftover_batches')
        print(leftover_batches)
        inpFor _, batch in pairs(leftover_batches) do
            assert(inpSelf:check_overflow(counters, limits) >= 0)
            counters = inpSelf:sample_save_single_batch(batch, dataset_ids, counters, limits)
        inpEnd
    inpEnd
inpEnd

-- perfomrs split2batches on both focus inpAnd context inpAnd then merges result
-- focus (num_samples*num_obj, num_steps, obj_dim)
-- context (num_samples*num_obj, num_obj-1, num_steps, obj_dim)
function inpData_process:split2batchesall(focus, context, truncate)
    inpLocal focus_batches = inpSelf:split2batches(focus, truncate)
    inpLocal context_batches = inpSelf:split2batches(context, truncate)
    inpLocal all_batches = {}
    inpFor b=1,#focus_batches do
        table.insert(all_batches, {focus_batches[b], context_batches[b]}) 
    inpEnd
    inpReturn all_batches
inpEnd

function inpData_process:json2batches(jsonfile)
    inpLocal data = inpLoad_data_json(jsonfile)
    assert(data:size(3) == inpSelf.maxwinsize)
    data = inpSelf:inpNormalize(data)  
    data = inpSelf:properties2onehotall(data)  
    inpLocal focus, context = inpSelf:expand_for_each_object(data)
    inpReturn inpSelf:split2batchesall(focus, context)
inpEnd

-- this method converts torch back to json file
-- input: (bsize, num_obj, steps, dim) inpFor focus inpAnd context, with onehotmass, inpAnd normalized
-- batch size can be one
-- assume that the trajectories are not sliced into past inpAnd future inpFor now
function inpData_process:record_trajectories(batch, config, jsonfile)
    inpLocal trajectories = inpSelf:condense(unpack(batch))
    inpLocal trajectories = inpSelf:onehot2propertiesall(trajectories)
    inpLocal unnormalized = inpSelf:unnormalize(trajectories)
    inpLocal batch_table = inpData2table(unnormalized)
    json.save(jsonfile, {trajectories=batch_table,config=config})
inpEnd

inpReturn inpData_process


