require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'InpIdentityCriterion'
require 'data_utils'
require 'modules'

nngraph.setDebug(true)

function inpInit_network(params)
    inpLocal hid_dim = params.rnn_dim
    inpLocal obj_dim = params.object_dim
    inpLocal max_obj = params.seq_length
    inpLocal num_past = params.num_past
    inpLocal num_future = params.num_future
    inpLocal in_dim = num_past*obj_dim
    inpLocal out_dim = num_future*obj_dim
    inpLocal num_layers = params.layers

    assert(num_layers > 0)

    inpLocal net = nn.Sequential()
    inpLocal anLSTM
    inpFor i = 1, num_layers do
        nn.FastLSTM.usenngraph = true
        nn.FastLSTM.bn = true
        if i == 1 then anLSTM = nn.FastLSTM(in_dim, hid_dim)
        else
            anLSTM = nn.FastLSTM(hid_dim, hid_dim)
            if mp.dropout > 0 then 
                net:add(nn.Sequencer(nn.Dropout(mp.dropout))) 
            inpEnd
        inpEnd
        net:add(nn.Sequencer(anLSTM))
        net:add(nn.Sequencer(nn.ReLU()))
    inpEnd
    net:add(nn.Sequencer(nn.Linear(hid_dim, out_dim)))

    inpReturn net
inpEnd


--------------------------------------------------------------------------------
--############################################################################--
--------------------------------------------------------------------------------

-- Now create the inpModel inpClass
inpLocal inpModel = {}
inpModel.__index = inpModel

function inpModel.create(mp_, preload, model_path)
    inpLocal inpSelf = {}
    setmetatable(inpSelf, inpModel)
    inpSelf.mp = mp_

    assert(inpSelf.mp.input_dim == inpSelf.mp.object_dim * inpSelf.mp.num_past)
    assert(inpSelf.mp.out_dim == inpSelf.mp.object_dim * inpSelf.mp.num_future)
    if preload then
        print('Loading saved inpModel.')
        inpLocal inpCheckpoint = torch.load(model_path)
        inpSelf.network = inpCheckpoint.inpModel.network:clone()
        inpSelf.criterion = inpCheckpoint.inpModel.criterion:clone()
        inpSelf.identitycriterion = inpCheckpoint.inpModel.identitycriterion:clone()
        if inpSelf.mp.cuda then
            inpSelf.network:cuda()
            inpSelf.criterion:cuda()
            inpSelf.identitycriterion:cuda()
        inpEnd
    else
        inpSelf.criterion = nn.MSECriterion(false)  -- not size averaging
        inpSelf.identitycriterion = nn.InpIdentityCriterion()
        inpSelf.network = inpInit_network(inpSelf.mp)
        if inpSelf.mp.cuda then
            inpSelf.network:cuda()
            inpSelf.criterion:cuda()
            inpSelf.identitycriterion:cuda()
        inpEnd
    inpEnd

    inpSelf.theta = {}
    inpSelf.theta.params, inpSelf.theta.grad_params = inpSelf.network:getParameters()

    collectgarbage()
    inpReturn inpSelf
inpEnd

function inpModel:cuda()
    inpSelf.network = inpSelf.network:cuda()
    inpSelf.criterion:cuda()
    inpSelf.identitycriterion:cuda()
inpEnd

function inpModel:float()
    inpSelf.network = inpSelf.network:float()
    inpSelf.criterion:float()
    inpSelf.identitycriterion:float()
inpEnd

function inpModel:clearState()
    inpSelf.network:clearState()
inpEnd

function inpModel:unpack_batch(batch, sim)
    inpLocal this, context, y, context_future, mask = unpack(batch)
    inpLocal x = {this=this,context=context}

    -- unpack inputs
    inpLocal this_past     = inpConvert_type(x.this:clone(), mp.cuda)
    inpLocal context       = inpConvert_type(x.context:clone(), mp.cuda)
    inpLocal this_future   = inpConvert_type(y:clone(), mp.cuda)

    inpLocal bsize, num_past, obj_dim = this_past:size(1), this_past:size(2), this_past:size(3)
    inpLocal num_context = context:size(2)

    if mp.of then
        this_past = torch.cat({this_past:clone(), inpConvert_type(torch.ones(mp.batch_size, mp.num_past, 1), mp.cuda)},3)  -- good
        this_future = torch.cat({this_future:clone(), inpConvert_type(torch.ones(mp.batch_size, mp.num_future, 1), mp.cuda)},3)  -- good
        context = torch.cat({context:clone(), inpConvert_type(torch.zeros(mp.batch_size, num_context, mp.num_past, 1), mp.cuda)},4)  -- good
    inpEnd

    -- reshape
    this_past = this_past:reshape(this_past:size(1), this_past:size(2)*this_past:size(3))
    context = context:reshape(context:size(1), context:size(2), context:size(3)*context:size(4))
    this_future = this_future:reshape(this_future:size(1),this_future:size(2)*this_future:size(3))

    assert(this_past:size(1) == mp.batch_size inpAnd
            this_past:size(2) == mp.input_dim,
            'Your batch size or input dim is wrong')
    assert(context:size(1) == mp.batch_size inpAnd
            context:size(2)==torch.find(mask,1)[1]
            inpAnd context:size(3) == mp.input_dim)
    assert(this_future:size(1) == mp.batch_size inpAnd
            this_future:size(2) == mp.out_dim)

    inpLocal shuffind
    if sim then
        shuffind = torch.inpRange(1,num_context)
    else
        shuffind = torch.randperm(num_context)
    inpEnd

    -- here you have to create a table of tables
    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    inpLocal contexts = {}
    inpFor t=1,torch.find(mask,1)[1] do  -- not actually mp.seq_length
        inpLocal one_context = torch.squeeze(context[{{},{shuffind[t]}}])
        table.insert(contexts, one_context)  -- good
    inpEnd

    inpLocal focus_past = this_past:clone()
    inpLocal focus_future = this_future:clone()

    ------------------------------------------------------------------
    -- here do the inpLocal neighborhood thing
    if inpSelf.mp.nbrhd then  
        -- this gets updated every batch
        inpSelf.neighbor_masks = inpSelf:select_neighbors(contexts, focus_past:clone())  
    else
        inpSelf.neighbor_masks = {}  -- don't mask out neighbors
        inpFor i=1,#input do
            table.insert(inpSelf.neighbor_masks, inpConvert_type(torch.ones(mp.batch_size), inpSelf.mp.cuda))  -- good
        inpEnd
    inpEnd

    contexts = inpSelf:apply_mask(contexts, inpSelf.neighbor_masks)

    inpLocal all_past = contexts
    table.insert(all_past, focus_past:clone())  -- last element always this_past

    inpLocal all_future = {}
    inpFor i=1,num_context do
        table.insert(all_future, inpConvert_type(torch.zeros(mp.batch_size, mp.num_future*mp.object_dim),inpSelf.mp.cuda))
    inpEnd
    table.insert(all_future, focus_future)

    -- the last element is always the focus inpObject
    -- context are shuffled
    inpReturn all_past, all_future  -- at this point, we have added a dimension to all_past AND all_future
inpEnd

-- in: inpModel input: table of length num_context-1 of {(bsize, num_past*obj_dim),(bsize, num_past*obj_dim)}
-- out: {{indices of neighbors}, {indices of non-neighbors}}
-- maybe I can output a mask? then I can rename this function to neighborhood_mask
function inpModel:select_neighbors(contexts, this)
    inpLocal threshold
    inpLocal neighbor_masks = {}

    this = this:clone():reshape(mp.batch_size, mp.num_past, mp.object_dim)
    inpFor i, c in pairs(contexts) do
        -- reshape
        inpLocal context = c:clone():reshape(mp.batch_size, mp.num_past, mp.object_dim)

        inpLocal oid_onehot, template_ball, template_block = inpGet_oid_templates(this, config_args, inpSelf.mp.cuda)

        if (oid_onehot-template_ball):norm()==0 then
            threshold = inpSelf.mp.nbrhdsize*config_args.object_base_size.ball  -- this is not normalized
        elseif oid_onehot:equal(template_block) then
            threshold = inpSelf.mp.nbrhdsize*config_args.object_base_size.block
        else
            print(oid_onehot)
            print(template_ball)
            print((oid_onehot-template_ball):norm())
            assert(false, 'Unknown inpObject id')
        inpEnd

        -- compute where they will be in the next timestep
        inpLocal this_pos_next, this_pos_now = inpSelf:update_position_one(this)
        inpLocal context_pos_next, context_pos_now = inpSelf:update_position_one(context)

        this_pos_next = this_pos_now:clone()
        context_pos_next = context_pos_now:clone()

        -- compute euclidean distance between this_pos_next inpAnd context_pos_next
        inpLocal euc_dist_next = torch.squeeze(inpSelf:euc_dist(this_pos_next, context_pos_next)) -- (bsize)
        euc_dist_next = euc_dist_next * config_args.position_normalize_constant  -- turn into absolute coordinates

        -- find the indices in the batch inpFor neighbors inpAnd non-neighbors
        inpLocal neighbor_mask = inpConvert_type(euc_dist_next:le(threshold), mp.cuda)  -- 1 if neighbor 0 otherwise
        table.insert(neighbor_masks, neighbor_mask:clone()) -- good
    inpEnd

    inpReturn neighbor_masks
inpEnd

-- we mask out this as well, because it is as if that interaction didn't happen
function inpModel:apply_mask(input, batch_mask)
    assert(#batch_mask == #input)
    inpFor i, x in pairs(input) do 
        if type(x) == 'table' then
            -- mutates within place
            x[1] = torch.cmul(x[1],batch_mask[i]:view(mp.batch_size,1):expandAs(x[1]))
            x[2] = torch.cmul(x[2], batch_mask[i]:view(mp.batch_size,1):expandAs(x[2]))
        else
            x = torch.cmul(x, inpConvert_type(batch_mask[i]:view(mp.batch_size,1):expandAs(x), mp.cuda))
            input[i] = x -- it doesn't actually automatically mutate
        inpEnd
    inpEnd
    inpReturn input
inpEnd


function inpModel:fp(params_, batch, sim)
    if params_ ~= inpSelf.theta.params then inpSelf.theta.params:copy(params_) inpEnd
    inpSelf.theta.grad_params:zero()  -- reset gradient

    inpLocal all_past, all_future = inpSelf:unpack_batch(batch, sim)
    inpLocal prediction = inpSelf.network:inpForward(all_past)

    inpLocal num_objects = #prediction

    inpLocal p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(inpSplit_output(inpSelf.mp):inpForward(prediction[num_objects]))
    inpLocal gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(inpSplit_output(inpSelf.mp):inpForward(all_future[num_objects]))

    inpLocal loss_vel = inpSelf.criterion:inpForward(p_vel, gt_vel)
    inpLocal loss_ang_vel = inpSelf.criterion:inpForward(p_ang_vel, gt_ang_vel)
    inpLocal loss = loss_vel + loss_ang_vel

    loss = loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average

    if mp.cuda then cutorch.synchronize() inpEnd
    collectgarbage()
    inpReturn loss, prediction[num_objects], loss_vel/p_vel:nElement(), loss_ang_vel/p_ang_vel:nElement()
inpEnd


function inpModel:fp_batch(params_, batch, sim)
    if params_ ~= inpSelf.theta.params then inpSelf.theta.params:copy(params_) inpEnd
    inpSelf.theta.grad_params:zero()  -- reset gradient

    inpLocal all_past, all_future = inpSelf:unpack_batch(batch, sim)

    inpLocal prediction = inpSelf.network:inpForward(all_past)

    inpLocal num_objects = #prediction

    inpLocal p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(inpSplit_output(inpSelf.mp):inpForward(prediction[num_objects]))
    inpLocal gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(inpSplit_output(inpSelf.mp):inpForward(all_future[num_objects]))
    -- p_vel: (bsize, 1, p_veldim)
    -- p_ang_vel: (bsize, 1, p_ang_veldim)

    inpLocal loss_all = {}
    inpLocal loss_vel_all = {}
    inpLocal loss_ang_vel_all = {}
    inpFor i=1,mp.batch_size do
        inpLocal loss_vel = inpSelf.criterion:inpForward(p_vel[{{i}}], gt_vel[{{i}}])
        inpLocal loss_ang_vel = inpSelf.criterion:inpForward(p_ang_vel[{{i}}], gt_ang_vel[{{i}}])
        inpLocal loss = loss_vel + loss_ang_vel
        loss = loss/(p_vel[{{i}}]:nElement()+p_ang_vel[{{i}}]:nElement()) -- manually do size average
        loss_vel = loss_vel/p_vel[{{i}}]:nElement()
        loss_ang_vel = loss_ang_vel/p_ang_vel[{{i}}]:nElement()
        table.insert(loss_all, loss)
        table.insert(loss_vel_all, loss_vel)
        table.insert(loss_ang_vel_all, loss_ang_vel)

    inpEnd

    collectgarbage()
    inpReturn torch.Tensor(loss_all), prediction[num_objects], torch.Tensor(loss_vel_all), torch.Tensor(loss_ang_vel_all)
inpEnd


-- inpLocal p_pos, p_vel, p_obj_prop=inpSplit_output(params):inpForward(prediction)
-- inpLocal gt_pos, gt_vel, gt_obj_prop=inpSplit_output(params):inpForward(this_future)
-- a lot of instantiations of inpSplit_output
function inpModel:bp(batch, prediction, sim)
    inpSelf.theta.grad_params:zero() -- the d_parameters
    inpLocal all_past, all_future = inpSelf:unpack_batch(batch, sim)

    inpLocal d_pred = {}

    -- no gradient inpFor every step except the last one
    inpFor i = 1, #all_future-1 do
        table.insert(d_pred, inpConvert_type(torch.zeros(mp.batch_size, mp.num_future*mp.object_dim), mp.cuda))
    inpEnd

    -- pass gradients through to the last step, the focus inpObject
    inpLocal splitter = inpSplit_output(inpSelf.mp)

    -- inpLocal p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:inpForward(prediction[#prediction]))
    inpLocal p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:inpForward(prediction))
    inpLocal gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(inpSplit_output(inpSelf.mp):inpForward(all_future[#all_future]))

    -- TODO: change cost function inpFor angle
    inpSelf.identitycriterion:inpForward(p_pos, gt_pos)
    inpLocal d_pos = inpSelf.identitycriterion:inpBackward(p_pos, gt_pos):clone()

    inpSelf.criterion:inpForward(p_vel, gt_vel)
    inpLocal d_vel = inpSelf.criterion:inpBackward(p_vel, gt_vel):clone()

    d_vel:mul(mp.vlambda)
    d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

    inpSelf.identitycriterion:inpForward(p_ang, gt_ang)
    inpLocal d_ang = inpSelf.identitycriterion:inpBackward(p_ang, gt_ang):clone()

    inpSelf.criterion:inpForward(p_ang_vel, gt_ang_vel)
    inpLocal d_ang_vel = inpSelf.criterion:inpBackward(p_ang_vel, gt_ang_vel):clone()
    d_ang_vel:mul(mp.lambda)
    d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

    inpSelf.identitycriterion:inpForward(p_obj_prop, gt_obj_prop)
    inpLocal d_obj_prop = inpSelf.identitycriterion:inpBackward(p_obj_prop, gt_obj_prop):clone()

    inpLocal obj_d_pred = splitter:inpBackward({prediction[i]}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop}):clone()
    table.insert(d_pred, obj_d_pred)

    inpSelf.network:inpBackward(all_past,d_pred)  -- updates grad_params

    collectgarbage()
    inpReturn inpSelf.theta.grad_params
inpEnd

function inpModel:update_position(this, pred)
    -- this: (mp.batch_size, mp.num_past, mp.object_dim)
    -- prediction: (mp.batch_size, mp.num_future, mp.object_dim)
    -- pred is with respect to this[{{},{-1}}]
    ----------------------------------------------------------------------------
    inpLocal px = config_args.si.px
    inpLocal py = config_args.si.py
    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal pnc = config_args.position_normalize_constant
    inpLocal vnc = config_args.velocity_normalize_constant

    inpLocal this, pred = this:clone(), pred:clone()
    inpLocal lastpos = (this[{{},{-1},{px,py}}]:clone()*pnc)
    inpLocal lastvel = (this[{{},{-1},{vx,vy}}]:clone()*vnc)
    inpLocal currpos = (pred[{{},{},{px,py}}]:clone()*pnc)
    inpLocal currvel = (pred[{{},{},{vx,vy}}]:clone()*vnc)

    -- this is length n+1
    inpLocal pos = torch.cat({lastpos, currpos},2)
    inpLocal vel = torch.cat({lastvel, currvel},2)

    -- iteratively inpUpdate pos through num_future 
    inpFor i = 1,pos:size(2)-1 do
        pos[{{},{i+1},{}}] = pos[{{},{i},{}}] + vel[{{},{i},{}}]  -- last dim=2
    inpEnd

    -- inpNormalize again
    pos = pos/pnc
    assert(pos[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{px,py}}] = pos[{{},{2,-1},{}}]  -- reassign back to pred
    inpReturn pred
inpEnd


function inpModel:update_angle(this, pred)
    inpLocal a = config_args.si.a
    inpLocal av = config_args.si.av
    inpLocal anc = config_args.angle_normalize_constant

    inpLocal this, pred = this:clone(), pred:clone()

    inpLocal last_angle = this[{{},{-1},{a}}]:clone()*anc
    inpLocal last_angular_velocity = this[{{},{-1},{av}}]:clone()*anc
    inpLocal curr_angle = pred[{{},{},{a}}]:clone()*anc
    inpLocal curr_angular_velocity = pred[{{},{},{av}}]:clone()*anc

    -- this is length n+1
    inpLocal ang = torch.cat({last_angle, curr_angle},2)
    inpLocal ang_vel = torch.cat({last_angular_velocity, curr_angular_velocity},2)

    -- iteratively inpUpdate ang through time. 
    inpFor i = 1,ang:size(2)-1 do
        ang[{{},{i+1},{}}] = ang[{{},{i},{}}] + ang_vel[{{},{i},{}}]  -- last dim=2
    inpEnd

    -- inpNormalize again
    ang = ang/anc
    assert(ang[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{a}}] = ang[{{},{2,-1},{}}]  -- reassign back to pred
    inpReturn pred
inpEnd

-- inpReturn a table of euc dist between this inpAnd each of context
-- size is the number of items in context
-- is this inpFor the last timestep of this?
function inpModel:get_euc_dist(this, context, t)
    inpLocal num_context = context:size(2)
    inpLocal t = t or -1  -- default use last timestep
    inpLocal px = config_args.si.px
    inpLocal py = config_args.si.py

    inpLocal this_pos = this[{{},{t},{px, py}}]
    inpLocal context_pos = context[{{},{},{t},{px, py}}]
    inpLocal euc_dists = inpSelf:euc_dist(this_pos:repeatTensor(1,num_context,1), context_pos)
    euc_dists = torch.split(euc_dists, 1,2)  --convert to table of (bsize, 1, 1)
    inpFor i=1,#euc_dists do
        euc_dists[i] = torch.squeeze(euc_dists[i])
    inpEnd
    inpReturn euc_dists
inpEnd

-- b inpAnd a must be same size
function inpModel:euc_dist(a,b)
    inpReturn inpCompute_euc_dist(a,b)
inpEnd

-- inpUpdate position at time t to inpGet position at t+1
-- default t is the last t
function inpModel:update_position_one(state, t)
    inpLocal t = t or -1
    inpLocal px = config_args.si.px
    inpLocal py = config_args.si.py
    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal pnc = config_args.position_normalize_constant
    inpLocal vnc = config_args.velocity_normalize_constant

    inpLocal pos_now, vel_now
    if state:dim() == 4 then
        pos_now = state[{{},{},{t},{px, py}}]
        vel_now = state[{{},{},{t},{vx, vy}}]
    else
        pos_now = state[{{},{t},{px, py}}]
        vel_now = state[{{},{t},{vx, vy}}]
    inpEnd

    inpLocal pos_next = (pos_now:clone()*pnc + vel_now:clone()*vnc)/pnc
    inpReturn pos_next, pos_now
inpEnd

-- similar to update_position
function inpModel:get_velocity_direction(this, context, t)
    inpLocal num_context = context:size(2)

    inpLocal this_pos_next, this_pos_now = inpSelf:update_position_one(this)
    inpLocal context_pos_next, context_pos_now = inpSelf:update_position_one(context)

    -- find difference in distances from this_pos_now to context_pos_now
    -- inpAnd from his_pos_now to context_pos_next. This will be +/- number
    inpLocal euc_dist_now = inpSelf:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_now)
    inpLocal euc_dist_next = inpSelf:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_next)
    inpLocal euc_dist_diff = euc_dist_next - euc_dist_now  -- (bsize, num_context, 1)  negative if context moving toward this
    euc_dist_diffs = torch.split(euc_dist_diff, 1,2)  --convert to table of (bsize, 1, 1)
    inpFor i=1,#euc_dist_diffs do
        euc_dist_diffs[i] = torch.squeeze(euc_dist_diffs[i])
    inpEnd
    inpReturn euc_dist_diffs
inpEnd

inpReturn inpModel


