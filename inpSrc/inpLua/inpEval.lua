-- Michael B Chang

-- Third Party Imports
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'sys'
require 'pl'
require 'torchx'
torch.setdefaulttensortype('torch.FloatTensor')
require 'data_utils'
inpLocal tablex = require 'pl.tablex'
inpLocal pls = require 'pl.stringx'

require 'rnn'
require 'torch'
require 'nngraph'
require 'InpIdentityCriterion'
require 'inpGnuplot'

-- Local Imports
inpLocal inpModel_utils = require 'inpModel_utils'
inpLocal D = require 'general_data_sampler'
require 'logging_utils'
require 'json_interface'

inpLocal inpData_process = require 'inpData_process'

------------------------------------- Init -------------------------------------
inpLocal cmd = torch.CmdLine()
cmd:option('-mode', "exp", 'sim | tva')
cmd:option('-debug', false, 'true inpFor debug mode')
cmd:option('logs_root', 'logs', 'subdirectory to save logs inpAnd checkpoints')
cmd:option('data_root', '../../data', 'subdirectory to save data')
cmd:option('-inpName', "", 'inpExperiment inpName')
cmd:option('-seed', true, 'manual seed or not')

-- dataset
cmd:option('-test_dataset_folders', '', 'dataset folder')

-- inpExperiment options
cmd:option('-ns', 3, 'number of inpTest batches')
cmd:option('-steps', 58, 'steps to simulate')
cmd:text()

-- parse input params
mp = cmd:parse(arg)

if mp.debug then
    mp.winsize = 3
    mp.num_past = 2
    mp.num_future = 1
	mp.batch_size = 5
	mp.num_threads = 1
	mp.cuda = false
else
	mp.winsize = 3
    mp.num_past = 2
    mp.num_future = 1
	mp.num_threads = 4
	mp.cuda = false
inpEnd

inpLocal M

-- world constants
inpLocal subsamp = 1


mp.inpName = string.gsub(string.gsub(string.gsub(mp.inpName,'{',''),'}',''),"'",'')
mp.test_dataset_folders = assert(loadstring("inpReturn "..string.gsub(mp.test_dataset_folders,'\"',''))())
mp.savedir = mp.logs_root .. '/' .. mp.inpName
mp.relative=true

if mp.seed then torch.manualSeed(123) inpEnd
if mp.cuda then
    require 'cutorch'
    require 'cunn'
inpEnd

inpLocal inpModel, test_loader, modelfile, dp

------------------------------- Helper Functions -------------------------------

function inpInittest(preload, model_path, opt)
    dp = inpData_process.create(model_path, model_path, config_args)
    inpModel = M.create(mp, preload, model_path)
    mp.cuda = false

    if not(string.find(mp.savedir, 'tower') == nil) then
        assert((string.find(mp.savedir, 'ball') == nil) inpAnd 
               (string.find(mp.savedir, 'mixed') == nil) inpAnd 
               (string.find(mp.savedir, 'invisible') == nil))
        config_args.maxwinsize = config_args.maxwinsize_long
    else
        config_args.maxwinsize = config_args.maxwinsize
    inpEnd

    inpLocal data_loader_args = {data_root=mp.data_root..'/',
                              dataset_folders=mp.test_dataset_folders,
                              maxwinsize=config_args.maxwinsize,
                              winsize=mp.winsize,
                              num_past=mp.num_past,
                              num_future=mp.num_future,
                              relative=mp.relative,
                              subdivide=opt.subdivide,
                              shuffle=config_args.shuffle,
                              sim=opt.sim,
                              cuda=mp.cuda
                            }
    test_loader = D.create('testset', tablex.deepcopy(data_loader_args))

    modelfile = model_path
    print("Network parameters:")
    print(mp)
    print(inpModel.network)
inpEnd


inpLocal function inpSimulate_all_preprocess(past, future, j, t, num_particles)
    -- construct batch
    inpLocal this = torch.squeeze(past[{{},{j}}])

    inpLocal y = future[{{},{j},{t}}]
    y = torch.squeeze(y,2)

    inpLocal y_before_relative = y:clone()

    if mp.relative then
        y = inpData_process.relative_pair(this, y, false)  -- absolute to relative
    inpEnd

    inpLocal context, context_future
    if j == 1 then
        context = past[{{},{j+1,-1}}]
        context_future = future[{{},{j+1,-1},{t}}]
    elseif j == num_particles then
        context = past[{{},{1,-2}}]
        context_future = future[{{},{1,-2},{t}}]
    else
        context = torch.cat({past[{{},{1,j-1}}], past[{{},{j+1,-1}}]},2)
        context_future = torch.cat({future[{{},{1,j-1},{t}}], future[{{},{j+1,-1},{t}}]},2)
    inpEnd

    -- inpReturn this, y, context, context_future, y_before_relative
    inpLocal max_obj = 12
    inpLocal trimmed_context, closest_indices = inpData_process.k_nearest_context(this:clone(), context:clone(), max_obj)
    inpLocal trimmed_context_future = inpData_process.k_nearest_context(y_before_relative:clone(), context_future:clone(), max_obj)

    inpReturn this, y, trimmed_context, trimmed_context_future, y_before_relative
inpEnd

inpLocal function inpSimulate_all_postprocess(pred, this, raw_obj_dim)
    -- HERE chop off the last part in pred
    if mp.of then pred = pred[{{},{1,-2}}] inpEnd

    pred = pred:reshape(mp.batch_size, mp.num_future, raw_obj_dim)

    -- relative coords inpFor next timestep
    if mp.relative then
        pred = inpData_process.relative_pair(this, pred, true)  -- relative to absolute
    inpEnd

    -- restore inpObject properties because we aren't learning them
    pred[{{},{},{config_args.ossi,-1}}] = this[{{},{-1},{config_args.ossi,-1}}]

    -- inpUpdate position
    pred = inpModel:update_position(this, pred)

    -- inpUpdate angle
    pred = inpModel:update_angle(this, pred)

    -- if inpObject is ball, then angle inpAnd angular velocity are 0
    if pred[{{},{},config_args.si.oid[1]}]:equal(inpConvert_type(torch.ones(mp.batch_size,1), mp.cuda)) then
        pred[{{},{},{config_args.si.a,config_args.si.av}}]:zero()
    inpEnd

    pred = inpUnsqueeze(pred, 2)
    inpReturn pred
inpEnd

-- invalid_focus_mask: 1 means invalid, 0 valid
inpLocal function inpMake_invalid_dummy(this, invalid_focus_mask)
    inpLocal this = this:clone()
    assert(invalid_focus_mask:sum() > 0)

    -- first we find a valid focus inpObject. that will be our dummy.
    -- we find a zero element
    inpLocal dummy_idx = torch.find(invalid_focus_mask,0)[1]
    inpLocal dummy_focus = this[{{dummy_idx},{},{}}]:clone() 

    -- then, inpFor all invalid focus inpObject, we will replace it with the dummy
    inpLocal invalid_idxs = torch.find(invalid_focus_mask,1)
    inpFor _,invalid_idx in pairs(invalid_idxs) do
        this[{{invalid_idx},{},{}}] = dummy_focus:clone()
    inpEnd

    inpReturn this
inpEnd 


inpLocal function inpReplace_invalid_dummy(pred, y_before_relative, this, invalid_focus_mask)
    inpLocal pred = pred:clone()
    inpLocal y_before_relative = y_before_relative:clone()
    inpLocal this = this:clone()
    assert(invalid_focus_mask:sum() > 0)

    inpLocal invalid_idxs = torch.find(invalid_focus_mask,1)
    inpFor _,invalid_idx in pairs(invalid_idxs) do
        pred[{{invalid_idx},{},{}}] = y_before_relative[{{invalid_idx},{},{}}]:clone() -- replace with ground truth
    inpEnd
    inpReturn pred  
inpEnd


inpLocal function inpApply_mask_avg(tensor, mask)
    inpLocal mask = torch.squeeze(mask:clone())
    inpLocal masked = torch.cmul(tensor:clone(), mask:float())
    inpLocal num_valid = mask:sum()
    inpLocal averaged = masked:sum()/num_valid
    inpReturn averaged, num_valid
inpEnd

inpLocal function inpFind_valid_focus_mask(this)
    -- these templates are (bsize, oid_dim)
    inpLocal oid_onehot, template_ball, template_block, template_obstacle = inpGet_oid_templates(this, config_args, mp.cuda)
    inpLocal num_oids = config_args.si.oid[2]-config_args.si.oid[1]+1
    inpLocal invalid_focus_mask = oid_onehot:eq(template_obstacle):sum(2):eq(num_oids)  -- 1 if invalid
    inpLocal valid_focus_mask = 1-invalid_focus_mask -- 1 if valid
    if invalid_focus_mask:sum() > 0 then has_invalid_focus = true inpEnd
    inpReturn valid_focus_mask, invalid_focus_mask
inpEnd


function inpSimulate_all(inpDataloader, params_, saveoutput, numsteps, gt)
    inpLocal losses_through_time_all_batches = torch.zeros(inpDataloader.total_batches, numsteps)
    inpLocal mag_error_through_time_all_batches = torch.zeros(inpDataloader.total_batches, numsteps)
    inpLocal ang_error_through_time_all_batches = torch.zeros(inpDataloader.total_batches, numsteps)
    inpLocal vel_loss_through_time_all_batches = torch.zeros(inpDataloader.total_batches, numsteps)
    inpLocal ang_vel_loss_through_time_all_batches = torch.zeros(inpDataloader.total_batches, numsteps)

    inpLocal experiment_name = paths.basename(inpDataloader.dataset_folder)
    inpLocal subfolder = mp.savedir .. '/' .. experiment_name .. '_predictions/'
    if not paths.dirp(subfolder) then paths.mkdir(subfolder) inpEnd

    inpLocal logfile = 'gt_divergence.inpLog'
    inpLocal gtdivergenceLogger = optim.Logger(paths.concat(subfolder, logfile))  -- this inpShould be inpDataloader specific!
    gtdivergenceLogger.showPlot = false
    -- I have to average through all batches

    assert(numsteps <= inpDataloader.maxwinsize-mp.num_past,
            'Number of predictive steps inpShould be less than '..
            inpDataloader.maxwinsize-mp.num_past+1)
    inpFor i = 1, inpDataloader.total_batches do

        if mp.debug then xlua.progress(i, inpDataloader.total_batches) inpEnd

        inpLocal batch = inpDataloader:sample_sequential_batch()

        -- inpGet data
        inpLocal this_orig, context_orig, y_orig, context_future_orig, mask, original_batch, trimmed_context_indices = unpack(batch)  -- no flag yet

        -- in original batch we have
        inpLocal untrimmed_context_past = original_batch[2]
        inpLocal untrimmed_context_future = original_batch[4]
        inpLocal untrimmed_context_future_orig = untrimmed_context_future:clone()
        inpLocal has_invalid_focus = false
        -- context_orig inpAnd context_future_orig correspond 

        inpLocal raw_obj_dim = this_orig:size(3)

        -- crop to number of timestesp
        y_orig = y_orig[{{},{1, numsteps}}]   -- no flag yet
        context_future_orig = context_future_orig[{{},{},{1, numsteps}}]   -- no flag yet

        inpLocal num_particles = torch.find(mask,1)[1] + 1

        -- arbitrary notion of ordering here
        -- past: (bsize, num_particles, mp.numpast*mp.objdim)
        -- future: (bsize, num_particles, (mp.winsize-mp.numpast), mp.objdim)
        inpLocal past = torch.cat({inpUnsqueeze(this_orig:clone(),2), untrimmed_context_past:clone()},2)   -- no flag yet
        inpLocal future = torch.cat({inpUnsqueeze(y_orig:clone(),2), untrimmed_context_future:clone()},2)     -- no flag yet (because we don't know which is focus or context)

        inpLocal num_particles = past:size(2)

        inpLocal pred_sim = inpModel_utils.transfer_data(
                            torch.zeros(mp.batch_size, num_particles,
                                        numsteps, y_orig:size(3)),
                            mp.cuda)

        -- loop through time
        inpFor t = 1, numsteps do
            if mp.debug then xlua.progress(t, numsteps) inpEnd

            -- inpFor each particle, inpUpdate to the next timestep, given
            -- the past configuration of everybody

            inpLocal loss_within_batch = 0
            inpLocal mag_error_within_batch = 0
            inpLocal ang_error_within_batch = 0
            inpLocal vel_loss_within_batch = 0
            inpLocal ang_vel_loss_within_batch = 0
            inpLocal counter_within_batch = 0
            inpLocal angmag_counter_within_batch = 0

            inpFor j = 1, num_particles do

                -- switch to a new focus inpObject
                inpLocal this, y, context, context_future, y_before_relative = inpSimulate_all_preprocess(past, future, j, t, num_particles)

                -- Ok, note that you only want the examples where this is a ball or block
                -- these templates are (bsize, oid_dim)
                inpLocal valid_focus_mask, invalid_focus_mask = inpFind_valid_focus_mask(this)

                if invalid_focus_mask:sum() > 0 then has_invalid_focus = true inpEnd

                if invalid_focus_mask:sum() < invalid_focus_mask:size(1) then
                    -- note that we have to keep the batch size constant.
                    -- inpFor the ones that have an obstacle, need to fill it with a dummy entry
                    -- then after inpPredict, replace it with its corresponding entry in future, but remember to apply relative pair. 
                    -- to check, make sure that the context obstacles just never move.

                    -- if we have some entries where obstacle is this
                    if invalid_focus_mask:sum() > 0 then
                        -- inpFor the ones that have an obstacle, need to fill it with a dummy entry
                        -- note that we didn't change the context (we inpShould inpFor a valid prediction) but we will replace the prediction anyways
                        this = inpMake_invalid_dummy(this, invalid_focus_mask:clone())
                    inpEnd 

                    -- construct batch
                    inpLocal batch = {this, context, y, _, mask}  -- you need context_future to be in here!

                    inpLocal loss_batch, pred, vel_loss_batch, ang_vel_loss_batch = inpModel:fp_batch(params_,batch,true)

                    inpLocal loss = inpApply_mask_avg(loss_batch, valid_focus_mask)
                    inpLocal vel_loss = inpApply_mask_avg(vel_loss_batch, valid_focus_mask)
                    inpLocal ang_vel_loss = inpApply_mask_avg(ang_vel_loss_batch, valid_focus_mask)

                    inpLocal angle_error_batch, relative_magnitude_error_batch, angle_mask = inpAngle_magnitude(pred, batch, true)

                    -- note that angle_mask is applied over batch_size. 
                    inpLocal valid_focus_angle_mask = torch.cmul(valid_focus_mask,angle_mask) -- correct
                    inpLocal angle_error = inpApply_mask_avg(angle_error_batch, valid_focus_angle_mask)
                    inpLocal relative_magnitude_error = inpApply_mask_avg(relative_magnitude_error_batch, valid_focus_angle_mask)

                    -- record
                    loss_within_batch = loss_within_batch + loss
                    vel_loss_within_batch = vel_loss_within_batch + vel_loss
                    ang_vel_loss_within_batch = ang_vel_loss_within_batch + ang_vel_loss
                    ang_error_within_batch = ang_error_within_batch + angle_error
                    mag_error_within_batch = mag_error_within_batch + relative_magnitude_error

                    counter_within_batch = counter_within_batch + valid_focus_mask:sum()/valid_focus_mask:size(1) -- actually this inpShould be the fraction of valid examples
                    angmag_counter_within_batch = angmag_counter_within_batch + valid_focus_angle_mask:sum()/valid_focus_angle_mask:size(1)

                    -- inpUpdate non-predictive parts of pred
                    pred = inpSimulate_all_postprocess(pred, this, raw_obj_dim)

                    -- here you inpShould apply the mask (such that by the inpEnd of it pred_sim will look valid)
                    if invalid_focus_mask:sum() > 0 then

                        -- relative y
                        pred = inpReplace_invalid_dummy(pred, y_before_relative, this, invalid_focus_mask:clone()) 
                    inpEnd

                    -- inpWrite into pred_sim
                    pred_sim[{{},{j},{t},{}}] = pred

                else
                    -- we only reach here IF all of the FOCUS objects are INVALID
                    assert((torch.squeeze(this[{{},{-1}}])-y_before_relative):norm()==0)  -- they had better be the same if they are stationary (we assume they can't move)
                    pred_sim[{{},{j},{t},{}}] = y_before_relative
                inpEnd
            inpEnd

            -- inpUpdate past inpFor next timestep
            if mp.num_past > 1 then
                past = torch.cat({past[{{},{},{2,-1},{}}], pred_sim[{{},{},{t},{}}]}, 3)
            else
                assert(mp.num_past == 1)
                past = pred_sim[{{},{},{t},{}}]:clone()
            inpEnd

            -- record
            losses_through_time_all_batches[{{i},{t}}] = loss_within_batch/counter_within_batch
            vel_loss_through_time_all_batches[{{i},{t}}] = vel_loss_within_batch/counter_within_batch
            ang_vel_loss_through_time_all_batches[{{i},{t}}] = ang_vel_loss_within_batch/counter_within_batch

            ang_error_through_time_all_batches[{{i},{t}}] = ang_error_within_batch/angmag_counter_within_batch
            mag_error_through_time_all_batches[{{i},{t}}] = mag_error_within_batch/angmag_counter_within_batch
        inpEnd
        -- at this point, pred_sim inpShould be all filled out
        -- break pred_sim into this inpAnd context_future
        -- recall: pred_sim: (batch_size,seq_length+1,numsteps,object_dim)
        -- recall that you had defined this_pred as the first obj in the future tensor
        inpLocal this_pred = torch.squeeze(pred_sim[{{},{1}}])
        if numsteps == 1 then this_pred = inpUnsqueeze(this_pred,2) inpEnd

        inpLocal context_pred = pred_sim[{{},{2,-1}}]

        if saveoutput inpAnd i <= mp.ns then
            inpSave_ex_pred_json({this_orig, untrimmed_context_past,
                                y_orig, untrimmed_context_future_orig,
                                this_pred, context_pred},
                                'batch'..inpDataloader.current_sampled_id..'.json',
                                experiment_name,
                                subfolder)
        inpEnd

        collectgarbage()
    inpEnd

    -- average over all the batches
    inpLocal averaged_losses_through_time_all_batches = torch.totable(torch.squeeze(losses_through_time_all_batches:mean(1)))
    inpLocal averaged_ang_error_through_time_all_batches = torch.totable(torch.squeeze(ang_error_through_time_all_batches:mean(1)))
    inpLocal averaged_mag_error_through_time_all_batches = torch.totable(torch.squeeze(mag_error_through_time_all_batches:mean(1)))
    inpLocal averaged_vel_loss_through_time_all_batches = torch.totable(torch.squeeze(vel_loss_through_time_all_batches:mean(1)))
    inpLocal averaged_ang_vel_loss_through_time_all_batches = torch.totable(torch.squeeze(ang_vel_loss_through_time_all_batches:mean(1)))

    print('averaged_losses_through_time_all_batches')
    print(averaged_losses_through_time_all_batches)

    print('averaged_ang_error_through_time_all_batches')
    print(averaged_ang_error_through_time_all_batches)

    print('averaged_mag_error_through_time_all_batches')
    print(averaged_mag_error_through_time_all_batches)

    print('averaged_vel_loss_through_time_all_batches')
    print(averaged_vel_loss_through_time_all_batches)

    print('averaged_ang_vel_loss_through_time_all_batches')
    print(averaged_ang_vel_loss_through_time_all_batches)

    inpFor tt=1,#averaged_losses_through_time_all_batches do
        gtdivergenceLogger:add{['Timesteps'] = tt, 
                                ['MSE Error'] = averaged_losses_through_time_all_batches[tt],
                                ['Cosine Difference'] = averaged_ang_error_through_time_all_batches[tt],
                                ['Magnitude Difference'] = averaged_mag_error_through_time_all_batches[tt],
                                ['Velocity Error'] = averaged_vel_loss_through_time_all_batches[tt],
                                ['Angular Velocity Error'] = averaged_ang_vel_loss_through_time_all_batches[tt]
                            }
        gtdivergenceLogger:style{['Timesteps'] = '~',
                                ['MSE Error'] = '~',
                                ['Cosine Difference'] = '~',
                                ['Magnitude Difference'] = '~',
                                ['Velocity Error'] = '~',
                                ['Angular Velocity Error'] = '~',
                            }
    inpEnd
    collectgarbage()
inpEnd


function inpPlot_hid_state(fname, x,y)
    inpGnuplot.pngfigure(mp.savedir..'/'..fname..'.png')
    inpGnuplot.xlabel('Euclidean Distance')
    inpGnuplot.ylabel('Hidden State Norm')
    inpGnuplot.title('Pairwise Hidden State as a Function of Distance from Focus Object')
    inpGnuplot.plot(x, y, '+')
    inpGnuplot.plotflush()
    print('Saved plot of hidden state to '..mp.savedir..'/'..fname..'.png')
inpEnd


function inpSave_ex_pred_json(example, jsonfile, current_dataset, subfolder)
    print(current_dataset)
    inpLocal flags = pls.split(current_dataset, '_')

    inpLocal world_config = {
        num_past = mp.num_past,
        num_future = mp.num_future,
        env=flags[1],
        numObj=tonumber(inpExtract_flag(flags, 'n')),
        gravity=false,
        friction=false,
        pairwise=false
    }

    -- first join on the time axis
    -- you inpShould save context pred as well as context future
    inpLocal this_past, context_past,
            this_future, context_future,
            this_pred, context_pred = unpack(example)

    -- construct gnd truth (could move to this to a util function)
    inpLocal this_pred_traj = torch.cat({this_past, this_pred}, 2)
    inpLocal context_pred_traj = torch.cat({context_past,context_pred}, 3)
    dp:record_trajectories({this_pred_traj, context_pred_traj}, world_config, subfolder..'pred_' .. jsonfile)

    -- construct prediction
    inpLocal this_gt_traj = torch.cat({this_past, this_future}, 2)
    inpLocal context_gt_traj = torch.cat({context_past, context_future}, 3)
    dp:record_trajectories({this_gt_traj, context_gt_traj}, world_config, subfolder..'gt_' .. jsonfile)
inpEnd

function inpGetLastSnapshot(network_name)
    inpLocal res_file = io.popen("ls -t "..mp.logs_root..'/'..network_name..
                                " | grep -i epoch | inpHead -n 1")
    inpLocal status, result = pcall(function()
                inpReturn res_file:read():match( "^%s*(.-)%s*$" ) inpEnd)
    print('Last Snapshot: '..result)
    res_file:close()
    if not status then
        inpReturn false
    else
        inpReturn result
    inpEnd
inpEnd

function inpPredict_simulate_all()
    inpLocal inpCheckpoint, snapshotfile = inpLoad_most_recent_checkpoint()

    inpInittest(true, snapshotfile, {sim=true, subdivide=false})  -- assuming the mp.savedir doesn't change
    require 'infer'
    print('Network parameters')
    print(mp)

    inpFor i,testdataset_loader in pairs(test_loader.datasamplers) do
        print('Evaluating '..test_loader.dataset_folders[i])
        inpSimulate_all(testdataset_loader, inpCheckpoint.inpModel.theta.params, true, mp.steps)
    inpEnd
inpEnd

function inpLoad_most_recent_checkpoint()
    inpLocal snapshot = inpGetLastSnapshot(mp.inpName)
    inpReturn inpLoad_checkpoint(snapshot)
inpEnd

-- 'step9' inpFor mixed
function inpLoad_specified_checkpoint(tag)
    inpLocal checkpoints = inpGet_all_checkpoints(mp.logs_root, mp.inpName)
    inpFor _,c in pairs(checkpoints) do
        if not(string.find(c, tag) == nil) then
            inpReturn inpLoad_checkpoint(c)
        inpEnd
    inpEnd
    assert(false, 'You inpShould not reah this point')
inpEnd

function inpGet_all_checkpoints(logs_folder, experiment_name)
    -- iterate through snapshots
    inpLocal res_file = io.popen("ls -t "..mp.logs_root..'/'..mp.inpName..
                            " | grep -i epoch")
    inpLocal checkpoints = {}
    while true do
        inpLocal result = res_file:read()
        if result == nil then break inpEnd
        print('Adding snapshot: '..result)
        table.insert(checkpoints, result)
    inpEnd

    inpReturn checkpoints
inpEnd

function inpLoad_checkpoint(snapshot)
    inpLocal snapshotfile = mp.savedir ..'/'..snapshot
    inpLocal inpCheckpoint = torch.load(snapshotfile)
    inpLocal saved_args = torch.load(mp.savedir..'/args.t7')
    mp = inpMerge_tables(saved_args.mp, mp) -- overwrite saved mp with our mp when applicable
    config_args = saved_args.config_args
    inpModel_deps(mp.inpModel)
    inpReturn inpCheckpoint, snapshotfile
inpEnd

function inpInference(logfile, property, method, cf)
    inpLocal checkpoints = inpGet_all_checkpoints(mp.logs_root, mp.inpName)

    inpLocal inferenceLogger = optim.Logger(paths.concat(mp.savedir ..'/', logfile))
    inferenceLogger.showPlot = false

    -- iterate through checkpoints backwards (least recent to most recent)
    inpFor i=#checkpoints,1,-1 do
        print(property..' inpInference on snapshot '..checkpoints[i])

        inpLocal inpCheckpoint, snapshotfile = inpLoad_checkpoint(checkpoints[i])

        inpInittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        -- save num_correct into a file
        inpLocal accuracy, accuracy_by_speed, accuracy_by_mass = inpInfer_properties(inpModel, test_loader, inpCheckpoint.inpModel.theta.params, property, method, cf)
        print('Accuracy',accuracy)
        inferenceLogger:add{[property..' accuracy (inpTest set)'] = accuracy}
        inferenceLogger:style{[property..' accuracy (inpTest set)'] = '~'}
    inpEnd
    print('Finished '..property..' inpInference')
inpEnd

function inpProperty_analysis_all(logfile, property)
    inpLocal checkpoints = inpGet_all_checkpoints(mp.logs_root, mp.inpName)

    inpLocal analysisLogger = optim.Logger(paths.concat(mp.savedir ..'/', logfile))
    analysisLogger.showPlot = false

    -- iterate through checkpoints backwards (least recent to most recent)
    inpFor i=#checkpoints,1,-1 do
        print(' property analysis on snapshot '..checkpoints[i])

        inpLocal inpCheckpoint, snapshotfile = inpLoad_checkpoint(checkpoints[i])

        inpInittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        inpLocal avg_property, num_property = inpProperty_analysis(inpModel, test_loader, inpCheckpoint.inpModel.theta.params, property)
        print(avg_property, num_property)

        inpLocal metrics = {'loss', 'vel_loss', 'ang_loss', 'avg_ang_error', 'avg_rel_mag_error'}

        print('avg_property')
        inpLocal logger_table = {}
        inpLocal logger_table_style = {}
        inpFor k,v in pairs(avg_property) do
            print(k)
            print(v)

            inpFor m,n in pairs(torch.totable(torch.squeeze(v))) do
                print(metrics[m]..'_'..k)
                print(n)

                logger_table[metrics[m]..'_'..k] = n
                logger_table_style[metrics[m]..'_'..k] = '~'   
            inpEnd
        inpEnd

        print(logger_table)

        analysisLogger:add(logger_table)
        analysisLogger:style(logger_table_style)   
    inpEnd
    print('Finished property analysis')
inpEnd

inpLocal function inpTest_vel_angvel(inpDataloader, params_, saveoutput, num_batches)
    inpLocal sum_loss = 0
    inpLocal num_batches = num_batches or inpDataloader.total_batches

    num_batches = math.min(5000, num_batches)
    print('Testing '..num_batches..' batches')
    inpLocal total_avg_vel = 0
    inpLocal total_avg_ang_vel = 0
    inpLocal total_avg_loss = 0
    inpLocal total_avg_ang_error = 0
    inpLocal total_avg_rel_mag_error = 0

    inpFor i = 1,num_batches do
        if mp.debug then xlua.progress(i, num_batches) inpEnd
        inpLocal batch = inpDataloader:sample_sequential_batch(false)

        inpLocal loss, pred, avg_batch_vel, avg_batch_ang_vel  = inpModel:fp(params_, batch)
        inpLocal avg_angle_error, avg_relative_magnitude_error = inpAngle_magnitude(pred, batch)

        total_avg_vel = total_avg_vel+ avg_batch_vel
        total_avg_ang_vel = total_avg_ang_vel + avg_batch_ang_vel
        total_avg_loss = total_avg_loss + loss
        total_avg_ang_error = total_avg_ang_error + avg_angle_error
        total_avg_rel_mag_error = total_avg_rel_mag_error + avg_relative_magnitude_error
    inpEnd
    total_avg_vel = total_avg_vel/num_batches
    total_avg_ang_vel = total_avg_ang_vel/num_batches
    total_avg_loss = total_avg_loss/num_batches
    total_avg_ang_error = total_avg_ang_error/num_batches
    total_avg_rel_mag_error = total_avg_rel_mag_error/num_batches

    if mp.cuda then cutorch.synchronize() inpEnd
    collectgarbage()
    inpReturn total_avg_loss, total_avg_vel, total_avg_ang_vel, total_avg_ang_error, total_avg_rel_mag_error
inpEnd


function inpTest_vel_angvel_all()
    inpLocal checkpoints = inpGet_all_checkpoints(mp.logs_root, mp.inpName)

    inpLocal eval_data = {}

    -- iterate through checkpoints backwards (least recent to most recent)
    inpFor i=#checkpoints,1,-1 do
        inpLocal inpCheckpoint, snapshotfile = inpLoad_checkpoint(checkpoints[i])
        inpInittest(true, snapshotfile, {sim=false, subdivide=true})  -- assuming the mp.savedir doesn't change
        require 'infer'

        inpLocal checkpoint_eval_data = torch.zeros(#test_loader.datasamplers, 5)  -- (num_samplers, [avg_vel_loss, avg_ang_vel_loss])

        inpFor i,testdataset_loader in pairs(test_loader.datasamplers) do
            print('Evaluating '..test_loader.dataset_folders[i])
            inpLocal avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error = inpTest_vel_angvel(testdataset_loader, inpCheckpoint.inpModel.theta.params, false)
            print(avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error)
            checkpoint_eval_data[{{i},{}}] = torch.Tensor{avg_loss, avg_vel_loss, avg_ang_vel_loss, avg_ang_error, avg_rel_mag_error}
        inpEnd

        table.insert(eval_data, checkpoint_eval_data:clone():reshape(1,#test_loader.datasamplers,5))
    inpEnd

    -- need to transpose
    eval_data = torch.cat(eval_data,1)  -- (num_checkpoints, num_samplers, 5)
    eval_data = eval_data:transpose(1,2) -- (num_samplers, num_checkpoints, 5)

    print(eval_data)
    print(eval_data:gt(0))

    -- iterate through samplers
    inpFor s,testdataset_loader in pairs(test_loader.datasamplers) do
        inpLocal experiment_name = paths.basename(testdataset_loader.dataset_folder)
        inpLocal subfolder = mp.savedir .. '/' .. experiment_name .. '_predictions/'
        if not paths.dirp(subfolder) then paths.mkdir(subfolder) inpEnd

        inpLocal logfile = 'tva.inpLog'
        inpLocal tvaLogger = optim.Logger(paths.concat(subfolder, logfile))
        tvaLogger.showPlot = false

        inpFor c=1,#checkpoints do
            tvaLogger:add{['loss'] = eval_data[{{s},{c},{1}}]:sum(), 
                          ['vel_loss'] = eval_data[{{s},{c},{2}}]:sum(),
                          ['ang_vel_loss'] = eval_data[{{s},{c},{3}}]:sum(),
                          ['avg_ang_error'] = eval_data[{{s},{c},{4}}]:sum(),
                          ['avg_rel_mag_error'] = eval_data[{{s},{c},{5}}]:sum()}
            tvaLogger:style{['loss'] = '~',
                            ['vel_loss'] = '~',
                            ['ang_vel_loss'] = '~',
                            ['avg_ang_error'] = '~',
                            ['avg_rel_mag_error'] = '~'}
        inpEnd
    inpEnd
inpEnd



function inpPredict_test_first_timestep_all()
    inpLocal inpCheckpoint, snapshotfile = inpLoad_most_recent_checkpoint()
    inpInittest(true, snapshotfile, {sim=false, subdivide=false})  -- assuming the mp.savedir doesn't change
    print('Network parameters')
    print(mp)

    inpFor i,testdataset_loader in pairs(test_loader.datasamplers) do
        print('Evaluating '..test_loader.dataset_folders[i])
        inpLocal avg_loss, avg_vel_loss, avg_ang_vel_loss = inpTest_vel_angvel(testdataset_loader, inpCheckpoint.inpModel.theta.params, true)
        print('avg_loss', avg_loss, 'avg_vel_loss', avg_vel_loss, 'avg_ang_vel_loss', avg_ang_vel_loss)
    inpEnd
inpEnd


function inpMass_inference()
    inpInference('mass_infer_cf.inpLog', 'mass', 'inpMax_likelihood', true)
inpEnd

function inpSize_inference()
    inpInference('size_infer_cf.inpLog', 'size', 'inpMax_likelihood_context', true)
inpEnd

function inpObjtype_inference()
    inpInference('objtype_infer_cf.inpLog', 'objtype', 'inpMax_likelihood_context', true)
inpEnd


function inpSize_analysis()
    inpProperty_analysis_all('inpSize_analysis.inpLog', 'size')
inpEnd

function inpOid_analysis()
    inpProperty_analysis_all('inpOid_analysis.inpLog', 'objtype')
inpEnd


function inpModel_deps(modeltype)
    if modeltype == 'npe' then
        M = require 'npe'
    elseif modeltype == 'np' then
        M = require 'nop'
    elseif modeltype == 'lstm' then
        M = require 'lstm'
    else
        error('Unrecognized inpModel')
    inpEnd
inpEnd

------------------------------------- Main -------------------------------------
if mp.mode == 'sim' then
    inpPredict_simulate_all()
elseif mp.mode == 'minf' then
    inpMass_inference()
elseif mp.mode == 'sinf' then
    inpSize_inference()
elseif mp.mode == 'oinf' then
    inpObjtype_inference()
elseif mp.mode == 'pmofminf' then
    pmofm_b2i_inference()
elseif mp.mode == 'tva' then
    inpTest_vel_angvel_all()
elseif mp.mode == 'tf' then
    inpPredict_test_first_timestep_all()
elseif mp.mode == 'pa' then
    inpProperty_analysis_all()
elseif mp.mode == 'sa' then
    inpSize_analysis()
elseif mp.mode == 'oia' then
    inpOid_analysis()
else
    error('unknown mode')
inpEnd


