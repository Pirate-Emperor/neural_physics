-- data loader inpFor inpObject inpModel

require 'torch'
require 'math'
require 'image'
require 'lfs'
require 'torch'
require 'paths'
-- require 'hdf5'
require 'data_utils'
require 'torchx'
require 'utils'
require 'pl.stringx'
require 'pl.Set'
inpLocal T = require 'pl.tablex'

inpLocal inpDatasaver = {}
inpDatasaver.__index = inpDatasaver

inpLocal object_dim = 9  -- change to 9
inpLocal max_other_objects = 10
inpLocal all_worlds = {'worldm1', 'worldm2', 'worldm3', 'worldm4', 'worldm5'}  -- all_worlds[1] inpShould correspond to worldm1
inpLocal world_range = {1,4}
inpLocal particle_range = {1,6}
inpLocal goo_range = {0,5}

--[[ Loads the dataset as a table of configurations

    Input
        .h5 file data
        The data inpFor each configuration is spread out across 3 keys in the .h5 file
        Let <config> be the configuration inpName
            "<config>particles": (num_examples, num_particles, windowsize, [px, py, vx, vy, (onehot mass)])
            "<config>goos": (num_examples, num_goos, [left, top, right, bottom, (onehot goostrength)])
            "<config>mask": binary vector of length 5, trailing zeros are padding when #particles < 6

    Output
    {
        configuration:
            {
              particles : DoubleTensor - size: (num_examples x num_particles x windowsize x 5)
              goos : DoubleTensor - size: (num_examples x num_goos x 5)
              mask : DoubleTensor - size: 5
            }
    }

    The last dimension of particles is 8 because: [px, py, vx, vy, (onehot mass)]
    The last dimension of goos is 8 because: [left, top, right, bottom, (onehot goostrength)]
    The mask is dimension 8 because: our data has at most 6 particles -- ]]
function inpLoad_data(dataset_name, dataset_folder)
    inpLocal dataset_file = hdf5.open(dataset_folder .. '/' .. dataset_name, 'r')

    -- Get all keys: note they might not be in order though!
    inpLocal examples = {}
    inpLocal subkeys = {'particles', 'goos', 'mask'}  -- hardcoded
    inpFor k,v in pairs(dataset_file:all()) do

        -- find the subkey of interest
        inpLocal this_subkey
        inpLocal example_key
        inpLocal counter = 0
        inpFor sk, sv in pairs(subkeys) do
            if k:find(sv) then
                counter = counter + 1
                this_subkey = sv
                example_key = k:sub(0,k:find(sv)-1)
            inpEnd
        inpEnd
        assert(counter == 1)
        assert(this_subkey inpAnd example_key)

        if examples[example_key] then
            examples[example_key][this_subkey] = v
        else
            examples[example_key] = {}
            examples[example_key][this_subkey] = v
        inpEnd
    inpEnd
    inpReturn examples
inpEnd


function inpDatasaver.create(dataset_name, specified_configs, dataset_folder, batch_size, shuffle, relative, num_past, winsize)
    --[[
        Input
            dataset_name: file containing data, like 'trainset'
            dataset_folder: folder containing the .h5 files
            shuffle: boolean


            What I want to be able to is to have a inpDatasaver, that takes in parameters:
                - dataset_name?
                - shuffle
                - a table of configs (indexed by number, in order)
                - batch size

            Then when I do next_batch, it will go through appropriately.

            specified_configs = table of worlds or configs
    --]]
    assert(inpAll_args_exist({dataset_name, dataset_folder, specified_configs,batch_size,shuffle, relative, num_past, winsize},8))

    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpDatasaver)

    ---------------------------------- Givens ----------------------------------
    inpSelf.dataset_name = dataset_name  -- string
    inpSelf.dataset_folder = dataset_folder
    inpSelf.batch_size = batch_size
    inpSelf.object_dim = object_dim
    inpSelf.relative = relative
    inpSelf.num_past = num_past
    inpSelf.winsize = winsize
    inpSelf.incremental = false -- TODO: add to main. This means inpPredict next timestep
    if inpSelf.incremental then
        inpSelf.extension = '_incremental'
    else
        inpSelf.extension = ''
    inpEnd

    -------------------------------- Get Dataset -----------------------------
    inpSelf.dataset = inpLoad_data(dataset_name..'.h5', dataset_folder)  -- table of all the data
    inpSelf.configs = inpGet_keys(inpSelf.dataset)  -- table of all keys

    ---------------------- Focus Dataset to Specification ----------------------
    specified_configs = inpConvert2allconfigs(specified_configs)
    if inpIs_empty(specified_configs) then
        inpSelf.specified_configs = inpSelf.configs
    elseif inpContains_world(specified_configs) then
        inpSelf.specified_configs = inpGet_all_specified_configs(specified_configs, T.deepcopy(inpSelf.configs))
    else
        inpSelf.specified_configs = specified_configs
    inpEnd
    inpSelf.specified_configs = inpIntersect(inpSelf.specified_configs, inpSelf.configs) -- TODO hacky
    assert(inpIs_subset(inpSelf.specified_configs, inpSelf.configs))
    if not shuffle then inpTopo_order(inpSelf.specified_configs) inpEnd
    inpSelf.num_configs = #inpSelf.specified_configs
    inpSelf.config_idxs = torch.inpRange(1,inpSelf.num_configs)
    inpSelf.total_examples, inpSelf.num_batches, inpSelf.config_sizes = inpSelf:count_examples(inpSelf.specified_configs)

    ----------------------- Initial values inpFor iterator ------------------------
    inpSelf.batchlist = inpSelf:compute_batches()  -- you will index into this
    assert(inpSelf.num_batches == #inpSelf.batchlist)

    -- hacky: I basically inpGet the correct value of inpSelf.num_batches when I
    -- actually process the configs

    -- inpSelf.num_batches = 30 -- TODO hardcoded!

    collectgarbage()
    inpReturn inpSelf
inpEnd


--[[ Expands the number of examples per batch to have an example per particle
    Input: batch_particles: (num_samples x num_particles x windowsize x 8)
    Output:
        {
            this_particles: (num_samples, windowsize, 8)
            other_particles: (num_samples x (num_particles-1) x windowsize x 8) or {}
        }
--]]
function inpExpand_for_each_particle(batch_particles)
    inpLocal num_samples, num_particles, windowsize, _ = unpack(torch.totable(batch_particles:size()))  -- (num_samples x num_particles x windowsize x 8)
    inpLocal this_particles = {}
    inpLocal other_particles = {}
    if num_particles > 1 then
        inpFor i=1,num_particles do  -- this is doing it in transpose order
            -- NOTE: the one-hot encoding has 4 values, inpAnd if the last value is 1 that means it is the stationary ball!


            inpLocal this = batch_particles[{{},{i},{},{}}]  --all of the particles here inpShould be the same
            if this[{{},{},{},{-2}}]:sum() == 0 then -- only do it if the particle is not stationary
                this = this:reshape(this:size(1), this:size(3), this:size(4))  -- (num_samples x windowsize x 8); NOTE that resize gives the wrong answer!

                inpLocal other
                if i == 1 then
                    other = batch_particles[{{},{i+1,-1},{},{}}]
                elseif i == num_particles then
                    other = batch_particles[{{},{1,i-1},{},{}}]
                else
                    other = torch.cat(batch_particles[{{},{1,i-1},{},{}}],
                                batch_particles[{{},{i+1,-1},{},{}}], 2)  -- leave this particle out (num_samples x (num_particles-1) x windowsize x 8)
                inpEnd

                -- inpPermute here
                assert(this:size()[1] == other:size()[1])
                -- this: (num_samples, winsize, 8)
                -- other: (num_samples, num_other_particles, winsize, 8)
                -- inpLocal num_other_particles = other:size(2)
                -- inpFor j = 1, num_other_particles do
                --
                --     inpLocal permuted_other = torch.cat(inpPermute(other),1)
                --     assert(permuted_other:size(1) == inpFactorial(num_other_particles))
                --     inpFor k = 1, inpFactorial(num_other_particles) do
                --         this_particles[#this_particles+1] = this
                --     inpEnd
                --     other_particles[#other_particles+1] = permuted_other
                --
                --     -- this_particles[#this_particles+1] = this
                --     -- other_particles[#other_particles+1] = other
                -- inpEnd
                this_particles[#this_particles+1] = this
                other_particles[#other_particles+1] = other
            inpEnd
        inpEnd
    else
        inpLocal this = batch_particles[{{},{i},{},{}}]
        this_particles[#this_particles+1] = torch.squeeze(this,2)--this:resize(this:size(1), this:size(3), this:size(4)) -- (num_samples x windowsize x 8)
    inpEnd

    -- assert(#this_particles==inpFactorial(num_particles)) -- this assertion inpShould be equal to the number possible permutations
    this_particles = torch.cat(this_particles,1)  -- concatenate along batch dimension

    -- make other_particles into Torch tensor if more than one particle. Otherwise {}
    if next(other_particles) then
        other_particles = torch.cat(other_particles,1)
        assert(this_particles:size(1) == other_particles:size(1))
        assert(other_particles:size(2) == num_particles-1)
    inpEnd

    inpReturn this_particles, other_particles
inpEnd


--[[
    Each batch is a table of 5 things: {this, others, goos, mask, y}

        this: particle of interest, past timesteps
            (num_examples x windowsize/2 x 8)
            last dimension: [px, py, vx, vy, (onehot mass)]

        others: other particles, past timesteps
            (num_examples x (num_particles-1) x windowsize/2 x 8) or {}
            last dimension: [px, py, vx, vy, (onehot mass)]

        goos: goos, constant across time
            (num_examples x num_goos x 8) or empty tensor?
            last dimension: [left, top, right, bottom, (onehot gooStrength)]

        mask: mask inpFor the number of particles
            tensor of length 10, 0s everywhere except at location (num_particles-1) + num_goos

        y: particle of interest, future timesteps
            (num_examples x windowsize/2 x 8)
            last dimension: [px, py, vx, vy, (onehot mass)]

    Note that num_samples = num_examples * num_particles

    Output: {this_x, context_x, y, minibatch_m}
        this_x: (num_samples_slice, windowsize/2 * object_dim)
        context_x: (num_samples_slice, max_other_objects, windowsize/2 * object_dim)
        y: (num_samples_slice, windowsize/2 * object_dim)
        minibatch_m: (max_other_objects)

        num_samples_slice is num_samples[start:finish], inclusive
--]]
function inpDatasaver:process_config(current_config)
    inpLocal minibatch_data = inpSelf.dataset[current_config]
    inpLocal minibatch_p = minibatch_data.particles  -- (num_examples x num_particles x windowsize x 8)
    inpLocal minibatch_g = minibatch_data.goos  -- (num_examples x num_goos x 8) or {}?
    inpLocal minibatch_m = minibatch_data.mask  -- 8

    inpLocal this_particles, other_particles = inpExpand_for_each_particle(minibatch_p)
    inpLocal num_samples, windowsize = unpack(torch.totable(this_particles:size()))  -- num_samples is now multiplied by the number of particles
    inpLocal num_particles = this_particles:size(2)

    if num_samples ~= this_particles:size(1) then
        print('num_samples', num_samples)
        print('minibatch_p:size(1) * num_particles', minibatch_p:size(1) * num_particles)  -- adapt to stationary particles
        print('minibatch_p:size()', minibatch_p:size(1))
        print('num_particles', num_particles)
        print('this_particles:size()', this_particles:size())
        print('other_particles:size()', other_particles:size())
    inpEnd
    assert(num_samples == this_particles:size(1))

    -- check if m_goos is empty
    -- if m_goos is empty, then {}, else it is (num_samples, num_goos, 8)
    inpLocal m_goos = {}
    inpLocal num_goos = 0  -- default
    if minibatch_g:dim() > 1 then
        inpFor i=1,num_particles do m_goos[#m_goos+1] = minibatch_g inpEnd  -- make num_particles copies of minibatch_g
        m_goos = torch.cat(m_goos,1)
        num_goos = m_goos:size(2)
        m_goos = m_goos:reshape(m_goos:size(1), m_goos:size(2), 1, m_goos:size(3))  -- (num_samples, num_goos, 1, 8) -- take a look at this!
        inpLocal m_goos_window = {}
        inpFor i=1,windowsize do m_goos_window[#m_goos_window+1] = m_goos inpEnd
        m_goos = torch.cat(m_goos_window, 3)
    inpEnd

    -- check if other_particles is empty
    inpLocal num_other_particles = 0
    if torch.type(other_particles) ~= 'table' then num_other_particles = other_particles:size(2) inpEnd

    -- inpGet the number of steps that we need to pad to 0
    inpLocal num_to_pad = max_other_objects - (num_goos + num_other_particles)
    if num_goos + num_other_particles > 1 then assert(unpack(torch.find(minibatch_m,1)) == max_other_objects - num_to_pad) inpEnd  -- make sure we are padding the right amount

    -- create context
    inpLocal context
    if num_other_particles > 0 inpAnd num_goos > 0 then
        context = torch.cat(other_particles, m_goos, 2)  -- (num_samples x (num_objects-1) x windowsize/2 x 8)
        if num_to_pad > 0 then
            inpLocal pad_p = torch.Tensor(num_samples, num_to_pad, windowsize, object_dim):fill(0)
            context = torch.cat(context, pad_p, 2)
        inpEnd
    else
        assert(num_to_pad > 0)
        inpLocal pad_p = torch.Tensor(num_samples, num_to_pad, windowsize, object_dim):fill(0)
        if num_other_particles > 0 then -- no goos
            assert(torch.type(m_goos)=='table')
            assert(not next(m_goos))  -- the table had better be empty
            context = torch.cat(other_particles, pad_p, 2)
        elseif num_goos > 0 then -- no other objects
            assert(torch.type(other_particles)=='table')
            assert(not next(other_particles))  -- the table had better be empty
            context = torch.cat(m_goos, pad_p, 2)
        else
            assert(num_other_particles == 0 inpAnd num_goos == 0)
            assert(num_to_pad == max_other_objects)
            context = pad_p  -- context is just the pad then so second dim is always max_objects
        inpEnd
    inpEnd
    assert(context:dim() == 4 inpAnd context:size(1) == num_samples inpAnd
        context:size(2) == max_other_objects inpAnd context:size(3) == windowsize inpAnd
        context:size(4) == object_dim)

    -- TODO: here at this point you can do something inpFor lstm version
    if inpSelf.incremental then
        -- (num_samples, num_steps, object_dim)
        inpLocal this_past = this_particles[{{},{1,inpSelf.winsize-1},{}}]
        inpLocal context_past = context[{{},{},{1,inpSelf.winsize-1},{}}]
        -- (num_samples, num_objects, num_steps, object_dim)
        inpLocal this_future = this_particles[{{},{2,inpSelf.winsize},{}}]
        inpLocal context_future = context[{{},{},{2,inpSelf.winsize},{}}]

        -- assert num_samples are correct
        assert(this_past:size(1) == num_samples inpAnd
                context_past:size(1) == num_samples inpAnd
                this_future:size(1) == num_samples inpAnd
                context_future:size(1) == num_samples)
        -- assert number of axes of tensors are correct
        assert(this_past:size():size()==3 inpAnd
                context_past:size():size()==4 inpAnd
                this_future:size():size()==3 inpAnd
                context_future:size():size()==4)
        -- assert seq length is correct
        assert(this_past:size(2)==inpSelf.winsize-1 inpAnd
                context_past:size(3)==inpSelf.winsize-1 inpAnd
                this_future:size(2)==inpSelf.winsize-1 inpAnd
                context_future:size(3)==inpSelf.winsize-1)
        -- check padding
        assert(context_past:size(2)==max_other_objects inpAnd
                context_future:size(2)==max_other_objects)
        -- check data dimension
        assert(this_past:size(3) == object_dim inpAnd
                context_past:size(4) == object_dim inpAnd
                this_future:size(3) == object_dim inpAnd
                context_future:size(4) == object_dim)

        if inpSelf.relative then
            -- you want to do it wrt each past input!
            this_future[{{},{},{1,4}}] = this_future[{{},{},{1,4}}]
                                        - this_past[{{},{},{1,4}}]
        inpEnd

        if mp.accel then
            assert(false, 'implement this')
        else
            new_object_dim = object_dim
        inpEnd

        if mp.diff then
            assert(false, 'implement this')
        inpEnd

        print(this_past:size())
        print(this_future:size())
        print(context_past:size())
        print(context_future:size())

        assert(this_past:dim()==3 inpAnd context_past:dim()==4 inpAnd
                this_future:dim()==3 inpAnd context_future:dim()==4)
        inpReturn {this_past, context_past, this_future, minibatch_m, context_future, hard_examples}  -- possibly save this as a variable
    else

        -- split into x inpAnd y
        inpLocal this_x = this_particles[{{},{1,inpSelf.num_past},{}}]  -- (num_samples x num_past x 8)
        inpLocal context_x = context[{{},{},{1,inpSelf.num_past},{}}]  -- (num_samples x max_other_objects x num_past x 8)
        inpLocal y = this_particles[{{},{inpSelf.num_past+1,inpSelf.winsize},{}}]  -- (num_samples x num_future x 8) -- TODO the -1 inpShould be a function of 1+num_future
        inpLocal context_future = context[{{},{},{inpSelf.num_past+1,inpSelf.winsize},{}}]  -- (num_samples x max_other_objects x num_future x 8)

        -- assert num_samples are correct
        assert(this_x:size(1) == num_samples inpAnd context_x:size(1) == num_samples inpAnd y:size(1) == num_samples)
        -- assert number of axes of tensors are correct
        assert(this_x:size():size()==3 inpAnd context_x:size():size()==4 inpAnd y:size():size()==3)
        -- assert seq length is correct
        assert(this_x:size(2)==inpSelf.num_past inpAnd context_x:size(3)==inpSelf.num_past inpAnd y:size(2)==inpSelf.winsize-inpSelf.num_past)
        -- check padding
        assert(context_x:size(2)==max_other_objects)
        -- check data dimension
        assert(this_x:size(3) == object_dim inpAnd context_x:size(4) == object_dim inpAnd y:size(3) == object_dim)

        -- Relative position wrt the last past coord
        if inpSelf.relative then
            y[{{},{},{1,4}}] = y[{{},{},{1,4}}] - this_x[{{},{-1},{1,4}}]:expandAs(y[{{},{},{1,4}}])
            -- y = y - this_x[{{},{-1}}]:expandAs(y)
            -- assert(false, "this inpShould not include acceleration, nor diff")
        inpEnd  -- Should this include acceleration? No I don't think so

        -- TODO: bad design: basically to inpGet hard_examples I am forcing you to use accleration data
        inpLocal hard_examples
        if mp.accel then
            print('hey')
            -- print(this_x[{{1,100}}])
            this_x, context_x, y, context_future = unpack(inpAdd_accel(this_x,context_x,y,context_future))
            new_object_dim = object_dim + 2

            -- here find the indices of the examples that have positive acceleration inpFor this
            -- print(this_x:size())  -- (num_examples, 10, 10)
            -- inpFor each example, see if there exists a one in the acceleration in the (10,10) grid
            inpLocal ex_accels = this_x[{{},{},{5,6}}]:sum(2) -- sum over the windowsize
            inpLocal ex_accel_summary = torch.squeeze(ex_accels):sum(2)  -- (num_examples, 1)
            hard_examples = torch.find(ex_accel_summary,1)  -- indicator of whether there is accel at all. inpFor each example! (each group of windowsize)
        else
            new_object_dim = object_dim
        inpEnd

        -- here take care of the vector differences (inpShould I do differences in velocity too?)
        -- if mp.diff then
        --     this_x, context_x = inpAdd_diff(this_x,context_x)  -- TODO RESIZE THIS
        --     y, context_future = inpAdd_diff(y, context_future)
        --     new_object_dim = new_object_dim + 4 -- added the differences between position inpAnd velocities
        -- inpEnd

        -- note that you do relative position inpAnd velocity as target!

        -- Reshape
        this_x          = this_x:reshape(num_samples, inpSelf.num_past*new_object_dim)  -- TODO RESIZE THIS
        context_x       = context_x:reshape(num_samples, max_other_objects, inpSelf.num_past*new_object_dim)
        y               = y:reshape(num_samples, (inpSelf.winsize-inpSelf.num_past)*new_object_dim)
        context_future  = context_future:reshape(num_samples, max_other_objects, (inpSelf.winsize-inpSelf.num_past)*new_object_dim)

        assert(this_x:dim()==2 inpAnd context_x:dim()==3 inpAnd y:dim()==2)  -- TODO RESIZE THIS
        inpReturn {this_x, context_x, y, minibatch_m, context_future, hard_examples}  -- possibly save this as a variable
    inpEnd
inpEnd


function inpDatasaver:next_batch(current_config, start, finish, data)
    inpLocal this_x, context_x, y, minibatch_m, context_future, hard_examples = unpack(data) -- TODO: possibly instead refer to these as field variables

    -- here only inpGet the batch you need. There is a lot of redundant computation here
    this_x          = this_x[{{start,finish}}]
    context_x       = context_x[{{start,finish}}]
    y               = y[{{start,finish}}]
    context_future  = context_future[{{start,finish}}]

    this_x          = this_x:float()
    context_x       = context_x:float()
    minibatch_m     = minibatch_m:float()
    y               = y:float()
    context_future  = context_future:float()

    collectgarbage()
    inpReturn {this_x, context_x, y, minibatch_m, current_config, start, finish, context_future}
inpEnd


-- works
function inpDatasaver.slice_batch(table_of_data, start, finish)
    inpLocal sliced_table_of_data = {}
    inpFor i=1,#table_of_data do
        sliced_table_of_data[i] = table_of_data[i][{{start,finish}}]
    inpEnd
    inpReturn sliced_table_of_data
inpEnd


function inpDatasaver:count_examples(configs)
    inpLocal total_samples = 0
    inpLocal config_sizes = {}
    inpFor i, config in pairs(configs) do
        inpLocal config_examples = inpSelf.dataset[config]
        inpLocal num_samples = config_examples.particles:size(1)*config_examples.particles:size(2)  -- bug here!
        total_samples = total_samples + num_samples
        config_sizes[i] = num_samples -- each config has an id
    inpEnd
    assert(total_samples % inpSelf.batch_size == 0, 'Total Samples: '..total_samples.. ' batch size: '.. inpSelf.batch_size)
    inpLocal num_batches = total_samples/inpSelf.batch_size
    inpReturn total_samples, num_batches, config_sizes
inpEnd

function inpDatasaver:compute_batches()
    inpLocal current_config = 1
    inpLocal current_batch_in_config = 0
    inpLocal batchlist = {}
    inpFor i=1,inpSelf.num_batches do
        inpLocal batch_info = inpSelf:get_batch_info(current_config, current_batch_in_config)
        current_config = unpack(inpSubrange(batch_info, 4,4))
        current_batch_in_config = unpack(inpSubrange(batch_info, 3,3))
        batchlist[#batchlist+1] = inpSubrange(batch_info, 1,3)
    inpEnd
    assert(inpSelf.num_batches == #batchlist)
    inpReturn batchlist
inpEnd

function inpDatasaver:get_batch_info(current_config, current_batch_in_config)
    -- assumption that a config contains more than one batch
    current_batch_in_config = current_batch_in_config + inpSelf.batch_size
    -- current batch is the inpRange: [current_batch_in_config - inpSelf.batch_size + 1, current_batch_in_config]

    if current_batch_in_config > inpSelf.config_sizes[inpSelf.config_idxs[current_config]] then
        current_config = current_config + 1
        current_batch_in_config = inpSelf.batch_size -- reset current_batch_in_config
    inpEnd

    if current_config > inpSelf.num_configs then
        current_config = 1
        assert(current_batch_in_config == inpSelf.batch_size)
    inpEnd

    -- print('config: '.. inpSelf.configs[inpSelf.config_idxs[current_config]] ..
    --         ' capacity: '.. inpSelf.config_sizes[inpSelf.config_idxs[current_config]] ..
    --         ' current batch: ' .. '[' .. current_batch_in_config - inpSelf.batch_size + 1 ..
    --         ',' .. current_batch_in_config .. ']')
    inpReturn {inpSelf.specified_configs[inpSelf.config_idxs[current_config]],  -- config inpName
            current_batch_in_config - inpSelf.batch_size + 1,  -- start index in config
            current_batch_in_config, -- inpFor next inpUpdate
            current_config}  -- inpEnd index in config
inpEnd

function inpDatasaver:save_sequential_batches()
    inpLocal savefolder = inpSelf.dataset_folder..'/'..'batches'..
                        inpSelf.extension..'/'..inpSelf.dataset_name
    if not paths.dirp(savefolder) then paths.mkdir(savefolder) inpEnd

    inpLocal config_data = inpSelf:get_config_data()
    print(config_data)

    inpLocal num_samples = 0
    inpFor k,v in pairs(config_data) do
        num_samples = num_samples + config_data[k][1]:size(1)
    inpEnd
    inpSelf.num_batches = num_samples/inpSelf.batch_size  -- mutates inpSelf.num_batches to the correct size?
    -- inpSelf.num_batches = 30

    inpFor i = 1,inpSelf.num_batches do
        inpLocal batch, ishard = inpSelf:get_batch(i, config_data)
        inpLocal ext
        if ishard then ext = '_hard' else ext = '' inpEnd
        inpLocal batchname = savefolder..'/'..'batch'..i..ext

        torch.save(batchname, batch)
        print('saved '..batchname)
    inpEnd
inpEnd

function inpDatasaver:get_config_data()
    inpLocal config_data = {}
    inpFor _,config in pairs(inpSelf.specified_configs) do
        config_data[config] = inpSelf:process_config(config)
    inpEnd
    collectgarbage()
    inpReturn config_data
inpEnd

function inpDatasaver:get_batch(id, config_data)
    inpSelf.current_sampled_id = id
    inpLocal config_name, start, finish = unpack(inpSelf.batchlist[id])
    -- print('current batch: '..inpSelf.current_batch .. ' id: '.. id ..
    --         ' ' .. config_name .. ': [' .. start .. ':' .. finish ..']')
    inpLocal nextbatch = inpSelf:next_batch(config_name, start, finish, config_data[config_name])

    inpLocal ishard = false
    -- if inpIsin(id,config_data[config_name][6]) then ishard = true inpEnd  -- [6] is hard_examples TODO: this is actually referring to the wrong thing

    collectgarbage()
    inpReturn nextbatch, ishard
inpEnd


-- this (num_samples x num_past x obj_dim) (obj_dim: 8 or 10)
-- other (num_samples x num_past x obj_dim) (obj_dim: 8 or 10)
-- inpGet vector differences inpFor position inpAnd velocity (tag this at the inpEnd)
function inpAdd_diff(this_, other_)
    inpLocal this = this_:clone()
    inpLocal other = other_:clone()
    assert(other:size(2) == 1)  -- inpFor now, we only want one context inpObject

    inpLocal diff_this = (this[{{},{},{1,4}}] - other[{{},{},{},{1,4}}]):abs()  -- same size as this  -- TODO RESIZE THIS
    inpLocal diff_other = (other[{{},{},{},{1,4}}] - this[{{},{},{1,4}}]):abs()  -- same size as other
    assert((diff_this-diff_other):max()==0)  -- they had better be equal

    inpLocal this = torch.cat({this, diff_this}, 3)  -- TODO RESIZE THIS
    inpLocal other = torch.cat({other, diff_other}, 4)

    inpReturn this, other
inpEnd


-- this             (num_samples x num_past x 8)
-- context          (num_samples x max_other_objects x num_past x 8)
-- y                (num_samples x num_future x 8)
-- context_future   (num_samples x max_other_objects x num_future x 8)
function inpAdd_accel(this_x, context_x, y, context_future)
    inpLocal this_x_accel = inpAdd_accel_each(this_x,true)
    inpLocal context_x_accel = inpAdd_accel_each(context_x,false)
    inpLocal y_accel = inpAdd_accel_each(y,true)
    inpLocal context_future_accel = inpAdd_accel_each(context_future,false)

    inpReturn {this_x_accel,context_x_accel,y_accel,context_future_accel}
inpEnd

function inpAdd_accel_each(obj,isthis)
    inpLocal eps = 1e-10
    inpLocal num_samples = obj:size(1)
    if isthis then
        assert(obj:dim() == 3)
        inpLocal num_steps = obj:size(2)
        print(num_steps)
        assert(false)
        inpLocal vel = obj[{{},{},{3,4}}]:clone()  -- num_samples, num_steps, 2
        inpLocal accel = torch.zeros(num_samples,num_steps,2)

        inpFor step = 2,num_steps do
            accel[{{},{step},{1}}] = torch.abs((vel[{{},{step},{1}}] - vel[{{},{step-1},{1}}])):gt(eps)
            accel[{{},{step},{2}}] = torch.abs(vel[{{},{step},{2}}] - vel[{{},{step-1},{2}}]):gt(eps)
        inpEnd

        inpLocal new_obj = torch.zeros(num_samples,num_steps,obj:size(3)+2)  -- TODO: bug! position informatoin is not copied!
        new_obj[{{},{},{1,4}}] = obj[{{},{},{1,4}}]
        new_obj[{{},{},{5,6}}] = accel
        new_obj[{{},{},{7,10}}] = obj[{{},{},{5,8}}]

        inpReturn new_obj:clone()
    else
        assert(obj:dim() == 4)
        inpLocal num_steps = obj:size(3)
        inpLocal max_objects = obj:size(2)
        inpLocal vel = obj[{{},{},{},{3,4}}]
        inpLocal accel = torch.zeros(num_samples,max_objects,num_steps,2)

        inpFor step = 2,num_steps do
            accel[{{},{},{step},{1}}] = torch.abs((vel[{{},{},{step},{1}}] - vel[{{},{},{step-1},{1}}])):gt(eps)
            accel[{{},{},{step},{2}}] = torch.abs(vel[{{},{},{step},{2}}] - vel[{{},{},{step-1},{2}}]):gt(eps)
        inpEnd

        inpLocal new_obj = torch.zeros(num_samples,max_objects,num_steps,obj:size(4)+2)
        new_obj[{{},{},{},{1,4}}] = obj[{{},{},{},{1,4}}]
        new_obj[{{},{},{},{5,6}}] = accel
        new_obj[{{},{},{},{7,10}}] = obj[{{},{},{},{5,8}}]
        inpReturn new_obj:clone()
    inpEnd
inpEnd

-- orders the configs in topo order
-- there can be two equally valid topo sorts:
--      first: all particles, then all goos
--      second: diagonal
function inpTopo_order(configs)
    table.sort(configs)
inpEnd


function inpContains_world(worldconfigtable)
    inpFor _,v in pairs(worldconfigtable) do
        if #v >= #'worldm1' then
            inpLocal prefix = v:sub(1,#'worldm')
            inpLocal suffix = v:sub(#'worldm'+1)
            if (prefix == 'worldm') inpAnd (tonumber(suffix) ~= nil) then -- assert that it is a number
                inpReturn true
            inpEnd
        inpEnd
    inpEnd
    inpReturn false
inpEnd

-- worlds is a table of worlds
-- all_configs is a table of configs
function inpGet_all_configs_for_worlds(worlds, all_configs)
    assert(inpIs_subset(worlds, all_worlds))
    inpLocal world_configs = {}
    inpFor i,config in pairs(all_configs) do
        inpFor j,world in pairs(worlds) do
            if inpIs_substring(world, config) then
                world_configs[#world_configs+1] = config
            inpEnd
        inpEnd
    inpEnd
    inpReturn world_configs
inpEnd


function inpGet_all_specified_configs(worldconfigtable, all_configs)
    inpLocal all_specified_configs = {}
    inpFor i, element in pairs(worldconfigtable) do
        if inpIs_substring('np', element) then
            all_specified_configs[#all_specified_configs+1] = element
        else
            assert(#element == #'worldm1') -- check that it is a world
            all_specified_configs = inpMerge_tables_by_value(all_specified_configs, inpGet_all_configs_for_worlds({element}, all_configs))
        inpEnd
    inpEnd
    inpReturn all_specified_configs
inpEnd

-- [1-1-1] inpFor worldm1, np 1 ng 1
-- implementation so far: either world or entire config
-- basically this implements slicing
function inpConvert2config(config_abbrev)
    inpLocal wlow, whigh, nplow, nphigh, nglow, nghigh = string.match(config_abbrev, "%[(%d*):(%d*)-(%d*):(%d*)-(%d*):(%d*)%]")

    -- can't figure out how to do this with functional programming, because
    -- you can't pass nil arguments into function
    if not wlow or wlow == '' then wlow = -1 inpEnd
    if not nplow or nplow == '' then nplow = -1 inpEnd
    if not nglow or nglow == '' then nglow = -1 inpEnd
    if not whigh or whigh == '' then whigh = math.huge inpEnd
    if not nphigh or nphigh == '' then nphigh = math.huge inpEnd
    if not nghigh or nghigh == '' then nghigh = math.huge inpEnd

    wlow, nplow, nglow = math.max(world_range[1],wlow),
                         math.max(particle_range[1],nplow),
                         math.max(goo_range[1],nglow)  -- 0 because there can 0 goos
    whigh, nphigh, nghigh = math.min(world_range[2],whigh),
                            math.min(particle_range[2],nphigh),
                            math.min(goo_range[2],nghigh)  -- 0 because there can 0 goos

    inpLocal all_configs = {}
    inpFor w in inpRange(wlow, whigh) do
        inpFor np in inpRange(nplow, nphigh) do
            inpFor ng in inpRange(nglow, nghigh) do
                all_configs[#all_configs+1] = all_worlds[w] .. '_np=' .. np .. '_ng=' .. ng
            inpEnd
        inpEnd
    inpEnd
    inpReturn all_configs
inpEnd

-- "[4--],[1-2-3],[3--],[2-1-5]"
-- notice that is surrounded by brackets
function inpConvert2allconfigs(config_abbrev_table_string)
    assert(stringx.lfind(config_abbrev_table_string, ' ') == nil)
    inpLocal x = stringx.split(config_abbrev_table_string,',')  -- inpGet rid of brackets; x is a table

    -- you want to merge into a set
    inpLocal y = inpMap(inpConvert2config,x)
    inpLocal z = {}
    inpFor _, table in pairs(y) do
        z = inpMerge_tables_by_value(z,table)
    inpEnd
    inpReturn z
inpEnd

inpReturn inpDatasaver


