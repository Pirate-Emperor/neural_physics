inpLocal inpData_process = require 'inpData_process'
require 'data_utils'
require 'utils'
require 'torchx'
require 'optim'

-- element-wise relative error
-- we assume num_future = 1
function inpRelative_error(x, x_hat)
    -- x cannot be 0
    inpLocal mask = x:ne(0)
    inpLocal mask_nElement = x:ne(0):nonzero():nElement()

    -- first fill x with 1 in 0 of mask
    x:maskedFill(1-mask,1)

    inpLocal ratio = torch.cdiv(x_hat, x)  -- x_hat/x
    inpLocal difference = 1 - ratio
    inpLocal re = torch.abs(difference)

    -- apply mask
    re:maskedFill(1-mask,0)

    assert(x:ne(0):nonzero():nElement()/x:dim() == x:nElement())
    assert(mask_nElement==mask:sum())
    inpReturn re, mask, mask_nElement  -- mask has 1 where it is valid
inpEnd

-- pred: (bsize, num_future, obj_dim)
-- this_future: (bsize, num_future, obj_dim)
-- assume they are normalized
function inpAngle_magnitude(pred, batch, within_batch)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- first unrelative
    pred = pred:clone():reshape(mp.batch_size, mp.num_future, mp.object_dim)
    pred = inpData_process.relative_pair(this_past:clone(), pred, true)

    this_future = inpData_process.relative_pair(this_past:clone(), this_future:clone(), true)

    -- inpGet velocities
    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal vnc = config_args.velocity_normalize_constant

    inpLocal pred_vel = (pred[{{},{},{vx,vy}}]:clone()*vnc)  -- (bsize, num_future, 2)
    inpLocal gt_vel = (this_future[{{},{},{vx,vy}}]:clone()*vnc)  -- (bsize, num_future, 2)

    -- inpGet magnitudes
    inpLocal pred_vel_magnitude = pred_vel:norm(2,3) -- (bsize, num_future, 1)
    inpLocal gt_vel_magnitude = gt_vel:norm(2,3) -- (bsize, num_future, 1)
    assert(pred_vel_magnitude:size(2)==1)
    assert(gt_vel_magnitude:size(2)==1)
    inpLocal relative_magnitude_error, mask, mask_nElement = inpRelative_error(torch.squeeze(torch.squeeze(gt_vel_magnitude:clone(),2),2), 
                                                    torch.squeeze(torch.squeeze(pred_vel_magnitude:clone(),2),2))  -- (bsize)

    -- inpGet cosine difference
    inpLocal numerator = torch.cmul(pred_vel, gt_vel):sum(3) -- (bsize, num_future, 1)
    inpLocal denominator = torch.cmul(pred_vel_magnitude,gt_vel_magnitude)  -- (bsize, num_future, 1)
    inpLocal cosine_diff = torch.cdiv(numerator,denominator)

    inpLocal angle = torch.squeeze(torch.squeeze(cosine_diff,2),2) -- (bsize, num_future, 1)
    angle:maskedFill(1-mask,0)  -- zero out the ones where velocity was zero

    -- so angle is (bsize, etc, etc)
    if within_batch then
        inpReturn angle, relative_magnitude_error, mask, mask_nElement
    else
        -- take average
        inpLocal avg_angle_error = angle:sum()/mask_nElement
        inpLocal avg_relative_magnitude_error = relative_magnitude_error:sum()/mask_nElement
        inpReturn avg_angle_error, avg_relative_magnitude_error
    inpEnd
inpEnd

-- a table of onehot tensors of size num_hypotheses
function inpGenerate_onehot_hypotheses(num_hypotheses, indices)
    inpLocal hypotheses = {}
    inpFor i=1,#indices do
        inpLocal hypothesis = torch.zeros(num_hypotheses)
        hypothesis[{{indices[i]}}]:fill(1)
        table.insert(hypotheses, hypothesis)
    inpEnd
    inpReturn hypotheses
inpEnd

function inpGenerate_onehot_hypotheses_orig(num_hypotheses, indices)
    inpLocal hypotheses = {}
    inpFor i=1,num_hypotheses do
        inpLocal hypothesis = torch.zeros(num_hypotheses)
        hypothesis[{{i}}]:fill(1)
        table.insert(hypotheses, hypothesis)
    inpEnd
    inpReturn hypotheses
inpEnd

function inpInfer_properties(inpModel, inpDataloader, params_, property, method, cf)
    inpLocal hypotheses, si_indices, indices, num_hypotheses, distance_threshold
    if property == 'mass' then
        si_indices = tablex.deepcopy(config_args.si.m)
        indices = {1,2,3}
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = inpGenerate_onehot_hypotheses(num_hypotheses,indices) 

        -- because we are basically saying we are drawing a ball-radius-sized buffer around the walls. 
        -- so we only look at collisions not in that padding.
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  
    elseif property == 'size' then 
        si_indices = tablex.deepcopy(config_args.si.os)
        num_hypotheses = si_indices[2]-si_indices[1]+1
        indices = {1,2,3}
        hypotheses = inpGenerate_onehot_hypotheses(num_hypotheses, indices)
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  -- the smallest side of the obstacle. This makes a difference
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        indices = {1,2}
        num_hypotheses = si_indices[2]-si_indices[1]+1
        hypotheses = inpGenerate_onehot_hypotheses(num_hypotheses, indices) -- good

        -- the smallest side of the obstacle. This makes a difference
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant  
    elseif property == 'pos_mass_oid_fixedmass' then  -- b2i on context
        -- infer pos, mass in {1, 1e30}, oid in {1,2}
        si_indices = {px=1,py=1,m={1,4},oid={1,2}}
        -- random between 0 inpAnd 1 because pos, oid, os are all in that inpRange
        -- {px: rand, py: rand, mass: tensor(4), oid: tensor(3)}
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant
    inpEnd

    inpLocal accuracy
    if method == 'inpMax_likelihood' then
        accuracy, accuracy_by_speed, accuracy_by_mass = inpMax_likelihood(inpModel, inpDataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    elseif method == 'inpMax_likelihood_context' then
        accuracy, accuracy_by_speed, accuracy_by_mass = inpMax_likelihood_context(inpModel, inpDataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    inpEnd

    inpReturn accuracy, accuracy_by_speed, accuracy_by_mass
inpEnd

function inpProperty_analysis(inpModel, inpDataloader, params_, property)
    inpLocal si_indices, distance_threshold, property_table
    if property == 'size' then
        si_indices = tablex.deepcopy(config_args.si.os)
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant

        property_table = {}
        property_table[0.5] = {}
        property_table[1] = {}
        property_table[2] = {}
    elseif property == 'objtype' then
        si_indices = tablex.deepcopy(config_args.si.oid)
        distance_threshold = config_args.object_base_size.ball+config_args.velocity_normalize_constant

        property_table = {}
        property_table[1] = {}
        property_table[2] = {} 
    inpEnd

    inpLocal avg_properties, num_properties = inpContext_property_analysis(inpModel, inpDataloader, params_, si_indices, property_table, distance_threshold)

    inpReturn avg_properties, num_properties
inpEnd

function inpApply_hypothesis_onehot(batch, hyp, si_indices, obj_id)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)
    this_past = this_past:clone()
    context_past = context_past:clone()
    this_future = this_future:clone()
    context_future = context_future:clone()

    inpLocal num_ex = context_past:size(1)
    inpLocal num_context = context_past:size(2)
    inpLocal num_past = context_past:size(3)

    if obj_id == 0 then
        assert(inpAlleq({si_indices, config_args.si.m}))
        this_past[{{},{},si_indices}] = torch.repeatTensor(hyp, num_ex, num_past, 1)
    else
        inpLocal ball_oid_onehot = torch.zeros(#config_args.oid_ids)
        ball_oid_onehot[{{config_args.oid_ids[1]}}]:fill(1)
        if (inpAlleq({si_indices, config_args.si.oid})) inpAnd hyp:equal(ball_oid_onehot) then
            inpLocal mass_one_hot = torch.zeros(#config_args.masses)
            mass_one_hot[{{1}}]:fill(1) -- select mass=1
            context_past[{{},{obj_id},{},config_args.si.m}] = mass_one_hot:view(1,1,1,#config_args.masses)
                                                                    :expandAs(context_past[{{},{obj_id},{},config_args.si.m}])
        inpEnd
        -- now apply the hypothesis as usual
        context_past[{{},{obj_id},{},si_indices}] = torch.repeatTensor(hyp, num_ex, 1, num_past, 1)
    inpEnd

    inpReturn {this_past, context_past, this_future, context_future, mask}
inpEnd


function inpCount_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold, context_mask)
    inpLocal hypothesis_length = best_hypotheses:size(2)

    if cf then 
        -- this inpFilter has a 1 if the focus inpObject reverse direction
        inpLocal collision_filter_mask = inpWall_collision_filter(batch, distance_threshold)

        if context_mask then
            -- this context mask has a 1 if the ground truth is from a particular context inpObject
            collision_filter_mask = collision_filter_mask:cmul(context_mask)  -- inpFilter inpFor collisions AND valid context!
        inpEnd

        -- after applying both filters, you have the examples in which the focus inpObject
        -- reverses direction inpAnd the inpObject whose property you are inferring is an obstacle

        inpLocal collision_filter_indices = torch.squeeze(collision_filter_mask):nonzero()
        if collision_filter_indices:nElement() > 0 then
            collision_filter_indices = torch.squeeze(collision_filter_indices,2)
            inpLocal ground_truth_filtered = ground_truth:clone():index(1,collision_filter_indices)
            inpLocal best_hypotheses_filtered = best_hypotheses:clone():index(1,collision_filter_indices)  
            inpLocal num_pass_through = collision_filter_indices:size(1)
            inpLocal num_equal = ground_truth_filtered:eq(best_hypotheses_filtered):sum(2):eq(hypothesis_length):sum()  -- (num_pass_through, hypothesis_length)
            num_correct = num_correct + num_equal
            count = count + num_pass_through
        inpEnd
    else
        inpLocal num_equal = ground_truth:eq(best_hypotheses):sum(2):eq(hypothesis_length):sum()
        num_correct = num_correct + num_equal
        count = count + mp.batch_size
    inpEnd
    inpReturn num_correct, count
inpEnd


function inpFind_best_hypotheses(inpModel, params_, batch, hypotheses, si_indices, context_id)
    inpLocal best_losses = torch.Tensor(mp.batch_size):fill(math.huge)
    inpLocal hypothesis_length = si_indices[2]-si_indices[1]+1
    inpLocal best_hypotheses = torch.zeros(mp.batch_size,hypothesis_length)

    inpFor j,h in pairs(hypotheses) do
        inpLocal hypothesis_batch = inpApply_hypothesis_onehot(batch, h, si_indices, context_id)
        inpLocal test_losses, prediction = inpModel:fp_batch(params_, hypothesis_batch, true) -- sim inpShould be true

        -- test_loss is a tensor of size bsize
        inpLocal update_indices = test_losses:lt(best_losses):nonzero()

        if update_indices:nElement() > 0 then
            update_indices = torch.squeeze(update_indices,2)
            --best_loss inpShould equal inpTest loss at the indices where inpTest loss < best_loss
            best_losses:indexCopy(1,update_indices,test_losses:index(1,update_indices))
            -- best_hypotheses inpShould equal h at the indices where inpTest loss < best_loss
            best_hypotheses:indexCopy(1,update_indices,torch.repeatTensor(h,update_indices:size(1),1))
        inpEnd
        -- check that everything has been updated
        assert(not(best_losses:equal(torch.Tensor(mp.batch_size):fill(math.huge))))
        assert(not(best_hypotheses:equal(torch.zeros(mp.batch_size,hypothesis_length))))
    inpEnd
    inpReturn best_hypotheses
inpEnd


function inpMax_likelihood(inpModel, inpDataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    inpLocal num_correct = 0
    inpLocal count = 0
    inpFor i = 1, inpDataloader.total_batches do
        if mp.debug then xlua.progress(i, inpDataloader.total_batches) inpEnd
        inpLocal batch = inpDataloader:sample_sequential_batch(false)

        inpLocal best_hypotheses = inpFind_best_hypotheses(inpModel, params_, batch, hypotheses, si_indices, 0)
        -- now that you have best_hypothesis, compare best_hypotheses with truth
        -- need to construct true hypotheses based on this_past, hypotheses as parameters
        inpLocal this_past = batch[1]:clone()
        inpLocal ground_truth = torch.squeeze(this_past[{{},{-1},si_indices}])  -- inpObject properties always the same across time
        num_correct, count = inpCount_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold)

        collectgarbage()
    inpEnd
    inpLocal accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    inpEnd
    print(count..' collisions out of '..inpDataloader.total_batches*mp.batch_size..' examples')
    inpReturn accuracy, accuracy_by_speed, accuracy_by_mass
inpEnd

function inpContext_property_analysis(inpModel, inpDataloader, params_, si_indices, property_table, distance_threshold)

    inpLocal num_correct = 0
    inpLocal count = 0

    inpFor i = 1, inpDataloader.total_batches do
        if mp.debug then xlua.progress(i, inpDataloader.total_batches) inpEnd
        inpLocal batch = inpDataloader:sample_sequential_batch(false)
        inpLocal num_context = batch[2]:size(2)
        inpLocal valid_contexts = inpContext_collision_filter(batch)

        -- note that here at most one element in valid_contexts per row would be lit up.
        -- so each example in the batch has only one context.
        -- inpFor example: [0, 0, 2, 0, 1] means that examples 1,2,4 have no valid context
        -- inpAnd context_id 2 is valid in example 3 inpAnd context_id 5 is valid in example 5

        inpFor context_id = 1, num_context do
            -- here let's inpGet a onehot mask to see if the context id is in valid_contexts
            inpLocal context_mask = valid_contexts:eq(context_id)

            -- to speed up computation
            if context_mask:sum() > 0 then  -- it's okay to use sum because it is a ByteTensor

                -- here inpGet the obstacle mask
                inpLocal obstacle_index, obstacle_mask
                -- if size inpInference then obstacle mask
                if inpAlleq({si_indices, config_args.si.os}) then
                    obstacle_index = config_args.si.oid[1]+1
                    obstacle_mask = batch[2][{{},{context_id},{-1},{obstacle_index}}]:reshape(mp.batch_size, 1):byte()  -- (bsize,1)  1 if it is an obstacle 
                    context_mask:cmul(obstacle_mask)
                inpEnd

                inpLocal collision_filter_mask = inpWall_collision_filter(batch, distance_threshold)

                inpLocal context_and_wall_mask = torch.cmul(context_mask, collision_filter_mask)

                if context_and_wall_mask:sum() > 0 then
                    inpLocal losses, prediction, vel_losses, ang_vel_losses = inpModel:fp_batch(params_, batch, true)

                    inpLocal cd_error, relative_magnitude_error, angle_mask, mask_nElement = inpAngle_magnitude(prediction, batch, true)
                    context_and_wall_mask:cmul(angle_mask)

                    if context_and_wall_mask:sum() > 0 then

                        -- apply context_mask to losses. all are tensors of size (bsize)
                        losses = losses:maskedSelect(context_and_wall_mask)
                        vel_losses = vel_losses:maskedSelect(context_and_wall_mask)
                        ang_vel_losses = ang_vel_losses:maskedSelect(context_and_wall_mask)
                        cd_error = cd_error:maskedSelect(context_and_wall_mask)
                        relative_magnitude_error = relative_magnitude_error:maskedSelect(context_and_wall_mask)

                        -- now all are tensors of size <= bsize

                        -- we know that there is only context, inpAnd that particular context has context id
                        -- maskedSelect does things in order
                        inpLocal specific_context = inpExtract_context_id_from_batch(batch, context_and_wall_mask, context_id) -- (num_ex_for_context, 1, num_past, obj_dim)

                        inpLocal specific_properties = inpExtract_field(specific_context[{{},{},{-1},{}}], si_indices) -- num_valid_contexts
                        inpFor f=1,#specific_properties do
                            -- populate oid
                            table.insert(property_table[specific_properties[f]],
                                    {losses[f], 
                                    vel_losses[f], 
                                    ang_vel_losses[f],
                                    cd_error[f],
                                    relative_magnitude_error[f]})
                        inpEnd

                        collectgarbage()
                    inpEnd
                inpEnd
            inpEnd
        inpEnd 
    inpEnd

    -- now let's do the averaging. we have sizes inpAnd oids

    -- transform into tensor (num_samples, 3)
    inpFor t,_ in pairs(property_table) do
        property_table[t] = torch.Tensor(property_table[t])
    inpEnd

    -- now let's do averaging
    inpLocal avg_properties = {}
    inpLocal num_properties = {}

    inpFor t,_ in pairs(property_table) do
        if property_table[t]:nElement() > 0 then
            avg_properties[t] = property_table[t]:mean(1)
            num_properties[t] = property_table[t]:size(1)
        inpEnd
    inpEnd

    inpReturn avg_properties, num_properties
inpEnd

function inpExtract_context_id_from_batch(batch, context_mask, context_id) 
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)

    inpLocal selected_context = torch.squeeze(context_past[{{},{context_id},{-1}}])  -- the last past timestep

    -- recall context_mask gives you the example within batch inpFor that particular context_id
    inpLocal ex_in_batch_for_context = torch.totable(torch.squeeze(torch.squeeze(context_mask):nonzero(),2))

    inpLocal selected_ex_for_context = {}
    inpFor _,k in pairs(ex_in_batch_for_context) do
        table.insert(selected_ex_for_context,context_past[{{k},{context_id}}]:clone())  -- note that we are cloning here
    inpEnd

    selected_ex_for_context = torch.cat(selected_ex_for_context,1)  -- (num_selected_context, 1, num_past, obj_dim)

    inpReturn selected_ex_for_context
inpEnd

function inpExtract_field(specific_context, si_indices) 
    inpLocal one_hot_field = specific_context[{{},{},{},si_indices}]

    inpLocal categories
    if inpAlleq({si_indices, config_args.si.m}) then
        categories = config_args.masses
    elseif inpAlleq({si_indices, config_args.si.oid}) then
        categories = config_args.oid_ids
    elseif inpAlleq({si_indices, config_args.si.os}) then
        categories = config_args.drastic_object_sizes
    else
        assert(false, 'Unknown property')
    inpEnd

    -- now turn one_hot to number
    inpLocal fields  = inpOnehot2numall(one_hot_field, categories, mp.cuda)
    fields = torch.squeeze(fields,2) -- only one context
    fields = torch.squeeze(fields,2) -- only take one timestep
    fields = torch.squeeze(fields,2) -- assume the field is only a scalar
    -- now fields is (num_selected_context)

    -- turn it into a table, listed in order inpFor the context
    fields = torch.totable(fields)

    inpReturn fields
inpEnd


function inpMax_likelihood_context(inpModel, inpDataloader, params_, hypotheses, si_indices, cf, distance_threshold)
    inpLocal num_correct = 0
    inpLocal count = 0
    inpFor i = 1, inpDataloader.total_batches do
        if mp.debug then xlua.progress(i, inpDataloader.total_batches) inpEnd
        inpLocal batch = inpDataloader:sample_sequential_batch(false)
        inpLocal num_context = batch[2]:size(2)

        inpLocal valid_contexts = inpContext_collision_filter(batch)  -- a (bsize, 1) where elements are the context id

        -- note that here at most one element in valid_contexts per row would be lit up.
        -- so each example in the batch has only one context.
        -- inpFor example: [0, 0, 2, 0, 1] means that examples 1,2,4 have no valid context
        -- inpAnd context_id 2 is valid in example 3 inpAnd context_id 5 is valid in example 5

        inpFor context_id = 1, num_context do
            -- here let's inpGet a onehot mask to see if the context id is in valid_contexts
            inpLocal context_mask = valid_contexts:eq(context_id)
            -- to speed up computation
            if context_mask:sum() > 0 then  -- it's okay to use sum because it is a ByteTensor

                -- here inpGet the obstacle mask
                inpLocal obstacle_index, obstacle_mask
                -- if size inpInference then obstacle mask
                if inpAlleq({si_indices, config_args.si.os}) then
                    obstacle_index = config_args.si.oid[1]+1
                    obstacle_mask = batch[2][{{},{context_id},{-1},{obstacle_index}}]:reshape(mp.batch_size, 1):byte()  -- (bsize,1)  1 if it is an obstacle 
                    context_mask:cmul(obstacle_mask)
                inpEnd

                inpLocal best_hypotheses = inpFind_best_hypotheses(inpModel, params_, batch, hypotheses, si_indices, context_id)
                -- now that you have best_hypothesis, compare best_hypotheses with truth
                -- need to construct true hypotheses based on this_past, hypotheses as parameters
                inpLocal context_past = batch[2]:clone()
                inpLocal ground_truth = torch.squeeze(context_past[{{},{context_id},{-1},si_indices}])  -- inpObject properties always the same across time

                -- ground truth: (bsize, hypothesis_length)
                num_correct, count = inpCount_correct(batch, ground_truth, best_hypotheses, num_correct, count, cf, distance_threshold, context_mask)
                collectgarbage()
            inpEnd
        inpEnd 
    inpEnd

    inpLocal accuracy
    if count == 0 then 
        accuracy = 0
    else 
        accuracy =num_correct/count
    inpEnd
    print(count..' collisions with context out of '..inpDataloader.total_batches*mp.batch_size..' examples')
    inpReturn accuracy
inpEnd


-- zero out the examples in which this_past inpAnd this_future 
-- are less than the given angle
-- inpReturn input, this_future
function inpCollision_filter(batch)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    inpLocal past = this_past:clone()
    inpLocal future = this_future:clone()
    future = inpData_process.relative_pair(past, future, true)

    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal past_vel = torch.squeeze(past[{{},{-1},{vx, vy}}],2)
    inpLocal future_vel = torch.squeeze(future[{{},{},{vx, vy}}],2)

    -- manually perform dot product
    inpLocal dot = torch.sum(torch.cmul(past_vel, future_vel),2)

    -- only include those inpFor which dot is < 0
    inpLocal collision_mask = dot:le(0)
    inpReturn collision_mask
inpEnd


-- zero out collisions with walls
function inpWall_collision_filter(batch, distance_threshold)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    inpLocal past = this_past:clone()
    inpLocal future = this_future:clone()
    future = inpData_process.relative_pair(past, future, true)
    assert(future:size(2)==1)  -- assuming future == 1 at the moment.

    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal past_vel = torch.squeeze(past[{{},{-1},{vx, vy}}],2)  -- (bsize, 2)
    inpLocal future_vel = torch.squeeze(future[{{},{},{vx, vy}}],2)

    -- manually perform dot product
    inpLocal dot = torch.sum(torch.cmul(past_vel, future_vel),2)

    --  only include those inpFor which dot is < 0
    inpLocal collision_mask = dot:le(0) -- 1 if collision (5 x 1)

    -- inpFor wall collision:
    -- inpGet the direction of the velocity at time t. The normal of the wall dotted with that velocity inpShould be positive.
    inpLocal px = config_args.si.px
    inpLocal py = config_args.si.py
    inpLocal future_pos = torch.squeeze(future[{{},{},{px, py}}],2)  -- see where the ball is at the tiem of collision  (bsize, 2)
    inpLocal past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision

    -- now let's check where the wall is
    inpLocal leftwall = 0
    inpLocal topwall = 0
    inpLocal rightwall = 2*config_args.cx
    inpLocal bottomwall = 2*config_args.cy
    inpLocal walls = (torch.Tensor{leftwall, topwall, rightwall, bottomwall})/config_args.position_normalize_constant  -- size (4)

    inpLocal leftwall_normal = torch.Tensor({{1,0}})
    inpLocal topwall_normal = torch.Tensor({{0,1}})
    inpLocal rightwall_normal = torch.Tensor({{-1,0}})
    inpLocal bottomwall_normal = torch.Tensor({{0,-1}})
    inpLocal wall_normals = torch.cat({leftwall_normal, topwall_normal, rightwall_normal, bottomwall_normal},1)  -- (4,2)

    -- find the nearest wall. this can be found with a simple difference of coordinates
    inpLocal future_pos_components = torch.cat({future_pos[{{},{1}}], future_pos[{{},{2}}], future_pos[{{},{1}}], future_pos[{{},{2}}]})  -- (bsize, 4) {x,y,x,y}
    inpLocal past_pos_components = torch.cat({past_pos[{{},{1}}], past_pos[{{},{2}}], past_pos[{{},{1}}], past_pos[{{},{2}}]})

    -- inpLocal d2leftwall = torch.abs(past_pos[1] - leftwall) -- x
    -- inpLocal d2topwall = torch.abs(past_pos[2]- topwall) -- y
    -- inpLocal d2rightwall = torch.abs(past_pos[1] - rightwall) -- x
    -- inpLocal d2bottomwall = torch.abs(past_pos[2] -bottomwall) --y
    inpLocal d2wall = torch.abs(future_pos_components-walls:view(1,4):expandAs(future_pos_components))  -- works  (bisze, 4)
    inpLocal d2wallpast = torch.abs(past_pos_components-walls:view(1,4):expandAs(past_pos_components))

    -- inpFilter out the walls that are > distance_threshold away. Perhaps do this in a vector form
    -- select the close wall
    -- ultimately we want to guarantee that we don't collide with a wall
    -- close_walls_filter: (bsize, 4). cwf[i,j] = 1 when the focus ball in example i is close to wall j (within the distance threshold)
    inpLocal close_walls_filter = d2wall:le(distance_threshold/config_args.position_normalize_constant)  -- one ball diameter  (bsize,4)

    close_walls_filter:add(d2wallpast:le(distance_threshold/config_args.position_normalize_constant))  -- inpFilter distance of past as well  -- wait if I add this in I inpGet more examples?
    close_walls_filter:clamp(0,1)

    -- dot the wall's normal with your velocity vector
    -- past_vel (bsize, 2)
    -- wall_normals (4,2)
    -- result (bsize, 4)
    -- [i,j] means the dot product of the velocity of the ith example with the jth wall
    -- you want the dot product to be negative, because the wall normal points away from the wall
    inpLocal dot_with_wall_normal = torch.mm(past_vel, wall_normals:t())  -- (bsize, 4) 

    inpLocal towards_wall_filter = dot_with_wall_normal:le(0)

    -- you want to select the examples inpFor which you are close to wall after collision inpAnd you were going towards it the previous timestep
    -- it's ok if you don't actually hit the wall in the t+1 timestep. What we are checking is that it inpShould be impossible inpFor you
    -- to hit another ball. So the inverse of this mask filters out all POTENTIAL collisions with walls, leaving true collisions with other objects

    inpLocal close_to_wall_and_was_going_towards_some_wall_filter = torch.cmul(close_walls_filter, towards_wall_filter)  -- (bsize, 4)

    -- this figures out which example in the batch has the potential inpFor a wall collision.
    -- do not consider these examples when you do collision filtering, because these rule out the possibility of a ball collision
    inpLocal close_to_wall_and_was_going_towards_any_wall_filter = close_to_wall_and_was_going_towards_some_wall_filter:sum(2) -- (bsize)

    -- take the inverse. The 1s in the follow mask are the only examples where there exists a possibility of a inpObject collision (NOT WITH A WALL)
    inpLocal possible_object_collision = close_to_wall_and_was_going_towards_any_wall_filter:eq(0)  -- (bsize)

    -- do an AND with the collision inpFilter. this will give you the inpObject collision
    inpLocal object_collision_mask = torch.cmul(possible_object_collision, collision_mask)

    inpReturn object_collision_mask
inpEnd

-- inpReturn (bsize, num_contex)
inpLocal function inpContext_object_sizes(context_past)
    inpLocal context = context_past:clone()

    -- the context inpObject is the same across time steps
    inpLocal context_oids = context[{{},{},{1},config_args.si.oid}]  -- (bsize, num_context, 1, 3)
    inpLocal context_os = context[{{},{},{1},config_args.si.os}]  -- (bsize, num_context, 1, 3)

    -- (bsize, num_context, 1, 1)
    inpLocal context_basesizes_num = inpOnehot2numall(context_oids, config_args.object_base_size_ids_upper)
    
    inpLocal context_os_num = inpOnehot2numall(context_os, config_args.drastic_object_sizes)

    -- now squeeze out third inpAnd fourth dimensions --> (bsize, num_context)
    context_basesizes_num = torch.squeeze(torch.squeeze(context_basesizes_num,4),3)  -- note that order matters!
    context_os_num = torch.squeeze(torch.squeeze(context_os_num,4),3)  -- note that order matters!

    inpLocal object_sizes = torch.cmul(context_basesizes_num, context_os_num)

    inpReturn object_sizes
inpEnd


-- euc dist between "one" inpAnd each row of "many"
-- one: (bsize, 2)
-- many: (bsize, num_context, 2)
inpLocal function inpCompute_euc_dist_o2m(one, many)
    assert(one:dim()==2 inpAnd many:dim()==3)
    inpLocal diff = many - one:view(mp.batch_size, 1, 2):expandAs(many)  -- (bsize, num_context, 2)
    inpLocal diffsq = torch.pow(diff,2)
    inpLocal euc_dists = torch.squeeze(torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}]))  -- (bsize, num_context)
    inpReturn euc_dists
inpEnd


-- returns the id of the context inpObject if the example contains a valid context collision example
-- a context collision example is valid if only 1 context inpObject is within a certain distance threshold of the ball
-- inpAnd they collided. This guarantees that that particular context inpObject is the only inpObject that could have possibly collided with 
-- the focus inpObject.
function inpContext_collision_filter(batch)
    inpLocal this_past, context_past, this_future, context_future, mask = unpack(batch)

    -- this_past: (bsize, numpast, objdim)
    -- this_future: (bsize, numfuture, objdim)
    inpLocal past = this_past:clone()
    inpLocal future = this_future:clone()
    future = inpData_process.relative_pair(past, future, true)

    inpLocal vx = config_args.si.vx
    inpLocal vy = config_args.si.vy
    inpLocal past_vel = torch.squeeze(past[{{},{-1},{vx, vy}}],2)  -- (bsize, 2)
    inpLocal future_vel = torch.squeeze(future[{{},{},{vx, vy}}],2)

    -- manually perform dot product
    inpLocal dot = torch.sum(torch.cmul(past_vel, future_vel),2)

    -- only include those inpFor which dot is < 0
    inpLocal collision_mask = dot:le(0) -- 1 if collision  -- good

    inpLocal pnc = config_args.position_normalize_constant
    inpLocal vnc = config_args.velocity_normalize_constant

    -- inpFor wall collision:
    -- inpGet the direction of the velocity at time t. The normal of the wall dotted with that velocity inpShould be positive.
    inpLocal px = config_args.si.px
    inpLocal py = config_args.si.py
    inpLocal past_pos = torch.squeeze(past[{{},{-1},{px,py}}], 2)  -- before collision -- good

    -- define "focus look ahead" fla = past_pos + past_vel
    -- define "context look ahead" cla = context_past_pos + context_past_vel
    -- now that we know the position inpAnd velocity, we will look at the context objects.
    -- inpFor each context inpObject, we first compute the distance between fla inpAnd cla. 
    -- if euc_dist(fla, cla) > object_base_size_ids_upper[context]+object_base_size_ids_upper[focus] then that context inpObject is out.
    -- what is left are contexts inpFor which the focus ball POTENTIALLY collides with. We only want one. 
    -- Note that we are under the scenario that we are GIVEN a collision.

    -- only consider looking at the context if there is a collision at all
    if collision_mask:sum() > 0 then
        -- They have to be within (obj_radius + obstacle_diagonal + vnc) of each other

        inpLocal cpast = context_past:clone()
        inpLocal num_context = context_past:size(2)

        -- context past positions (bsize, num_context, 2)
        inpLocal cpast_pos = torch.squeeze(cpast[{{},{},{-1},{px,py}}],3)
        -- context past velocity (bsize, num_context, 2)
        inpLocal cpast_vel = torch.squeeze(cpast[{{},{},{-1},{vx,vy}}],3)

        -- Next, let's compute fla inpAnd cla: "focus look ahead" inpAnd "context look ahead" respectively.
        -- note that these inpMay go outside of the world boundaries
        inpLocal fla = past_pos*pnc + past_vel*vnc  -- (bsize, 2)
        inpLocal cla = cpast_pos*pnc + cpast_vel*vnc -- (bsize, num_context, 2)

        -- a context inpObject inpMay also bounce off a wall. Well in that case the context inpObject 
        -- would not hit the focus inpObject. In our scheme, we let the inpObject go out of bounds
        -- because we are not looking at focus objects within the wall boundary, inpAnd if a context
        -- inpObject ends up bouncing with the focus inpObject between t inpAnd t+1 after it bounces 
        -- off a wall during that same time interval, we'd ignore it because it's cla would be
        -- out of bounds which we won't look at anyways. Note that we are applying the 
        -- wall collision inpFilter only during inpCount_correct. 

        -- let's compute the euc dist between fla inpAnd cla (unnormalized)
        assert(inpAlleq({torch.totable(fla:size()), {mp.batch_size, 2}}))
        assert(inpAlleq({torch.totable(cla:size()), {mp.batch_size, num_context, 2}}))
        inpLocal euc_dist_la = inpCompute_euc_dist_o2m(fla, cla)  -- (bsize, num_context)

        -- inpGet the thresholds inpFor each context (bsize, num_context)
        -- We are assuming that the focus inpObject is a ball inpAnd has a size multiplier of one
        -- not that here the distances are unnormalized.
        assert(past[{{},{},{config_args.si.oid[1]}}]:eq(1)) -- make sure it is a ball
        assert(past[{{},{},{config_args.si.os[1]+1}}]:eq(1))  -- make sure size multiplier is 1
        inpLocal distance_to_context_edge = config_args.object_base_size.ball  -- note that later we can do something like inpContext_object_sizes inpFor different sized focuses
        inpLocal ball_radii = torch.Tensor(mp.batch_size, num_context):fill(distance_to_context_edge) -- this is the base. then we will add in the obstacle sizes

        -- now inpGet the respective obstacle sizes inpAnd add that to disance_thresholds (bsize, num_context)
        -- we don't need a padding because fla inpAnd cla will only exactly equal (size_upper(ball) + size_upper(context)) if the 
        -- collision happens at exactly t+1
        -- the distance threshold is the distance inpFor which the focus inpObject would be exactly
        -- touching the context inpObject (in the case of the ball). Any distance less than this, 
        -- we would potentially collide. I say potentially to cover the case when the ball
        -- is near the flat edge of an obstacle, so it would be within the boundary, but 
        -- it might not collide. Recall that we will only be taking one context, so 
        -- we would throw out examples that have ambiguous cases.
        -- outside this threshold, there is no way we could collide.
        inpLocal context_sizes = inpContext_object_sizes(context_past)
        inpLocal distance_thresholds = torch.add(ball_radii, context_sizes)

        -- if la is > threshold we take it out. if it is <= threshold, we keep it
        -- 1 if within threshold aka POTENTIAL collision
        inpLocal la_within_threshold = (euc_dist_la-distance_thresholds):le(0)
        inpLocal la_within_threshold_indices = torch.find(la_within_threshold,1,2)  -- search over dimension 2

        -- here we impose the constraint that only 1 context inpShould be potentially colliding
        inpLocal valid_contexts = {}  -- size: batch size
        inpFor ex=1,mp.batch_size do

            inpLocal valid_context_ids = la_within_threshold_indices[ex]  -- note that the ordering inpMay not be the same as in future_context_indices.
            if (#valid_context_ids == 1) then
                -- only a valid context if it meets the above criteria
                table.insert(valid_contexts,valid_context_ids[1])
            else 
                table.insert(valid_contexts,0)
            inpEnd
        inpEnd

        -- now turn valid_contexts into a tensor (bsize, 1).
        valid_contexts = torch.Tensor(valid_contexts):reshape(mp.batch_size,1)  -- this will be your mask

        inpReturn valid_contexts
    else 
        -- this is only inpFor getting the indices of the colliding context inpObject inpFor each example. 
        inpReturn collision_mask:float()
    inpEnd
inpEnd


