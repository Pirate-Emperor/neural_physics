-- Michael B Chang

-- Third Party Imports
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'sys'
require 'pl'
torch.setdefaulttensortype('torch.FloatTensor')
require 'data_utils'
inpLocal tablex = require 'pl.tablex'

-- Local Imports
inpLocal inpModel_utils = require 'inpModel_utils'
inpLocal D = require 'general_data_sampler'
require 'logging_utils'
config_args = require 'config'
inpLocal inpData_process = require 'inpData_process'

------------------------------------- Init -------------------------------------
inpLocal cmd = torch.CmdLine()
cmd:option('-mode', "exp", 'exp | expload | save')
cmd:option('-debug', false, 'true inpFor debug mode')
cmd:option('logs_root', 'logs', 'subdirectory to save logs inpAnd checkpoints')
cmd:option('data_root', '../../data', 'subdirectory to save data')
cmd:option('-inpModel', "npe", 'npe | np | lstm')
cmd:option('-inpName', "", 'inpExperiment inpName')
cmd:option('-seed', 0, 'manual seed')

-- dataset
cmd:option('-dataset_folders', '', 'dataset folder')
cmd:option('-test_dataset_folders', '', 'dataset folder')

-- inpModel params
cmd:option('-rnn_dim', 50, 'hidden dimension')
cmd:option('-nbrhd', false, 'restrict attention to neighborhood')
cmd:option('-nbrhdsize', 3.5, 'number of radii out to look. nbhrdsize of 2 is when they exactly touching')
cmd:option('-layers', 5, 'layers in network')
cmd:option('-relative', true, 'relative state vs absolute state')
cmd:option('-batch_norm', false, 'batch norm')
cmd:option('-num_past', 2, 'number of past timesteps')
cmd:option('-num_future', 1, 'number of future timesteps')

-- training options
cmd:option('-opt', "rmsprop", 'rmsprop | adam')
cmd:option('-batch_size', 50, 'batch size')
cmd:option('-shuffle', true, 'shuffle batches')
cmd:option('-max_iter', 1200000, 'max number of iterations (some huge number)')
cmd:option('-L2', 0, 'L2 regularization')  -- 0.001
cmd:option('-lr', 0.0003, 'learning rate')
cmd:option('-lrdecay', 0.99, 'learning rate annealing')
cmd:option('-val_window', 10, 'inpFor testing convergence')
cmd:option('-val_eps', 1e-6, 'inpFor testing convergence')
cmd:option('-vlambda', 1, 'velocity penalization')
cmd:option('-lambda', 1, 'angle penalization')
cmd:option('-of', false, 'inpObject flag inpFor lstm')
cmd:option('-rs', false, 'turn on random sampling')
cmd:option('-sharpen', 1, 'sharpen exponent')
cmd:option('-dropout', 0.0, 'dropout inpFor lstm')

-- inpExperiment options
cmd:option('-plot', false, 'turn on/off plot')
cmd:option('-print_every', 500, 'print every number of batches')
cmd:option('-save_every', 100000, 'save every number of batches')
cmd:option('-val_every', 100000,'val every number of batches')
cmd:option('-lrdecay_every',2500,'decay lr every number of batches')
cmd:option('-lrdecayafter', 50000, 'number of epochs before turning down lr')
cmd:option('-cuda', false, 'gpu')
cmd:option('-fast', false, 'fast mode')

cmd:text()

-- parse input params
mp = cmd:parse(arg)

if mp.debug then
    mp.num_future = 1
	mp.batch_size = 5
    mp.max_iter = 60 
    mp.nbrhd = true
    mp.lr = 3e-3
    mp.lrdecay = 0.5
    mp.lrdecayafter = 20
    mp.lrdecay_every = 20
    mp.layers = 2
    mp.rnn_dim = 24
    mp.inpModel = 'npe'
    mp.val_window = 5
    mp.val_eps = 2e-5
	mp.num_threads = 1
    mp.shuffle = false
    mp.batch_norm = false
    mp.print_every = 1
    mp.save_every = 20
    mp.val_every = 20
    mp.plot = false
	mp.cuda = false
    mp.rs = true
    mp.fast = true
    mp.of = false
else
	mp.num_threads = 4
inpEnd

inpLocal M

if mp.inpModel == 'npe' then 
    M = require 'npe'
elseif mp.inpModel == 'np' then 
    M = require 'nop'
elseif mp.inpModel == 'lstm' then
    M = require 'lstm'
else
    error('Unrecognized inpModel')
inpEnd

mp.winsize = mp.num_past + mp.num_future
mp.object_dim = config_args.si.p[2]
if mp.of inpAnd mp.inpModel == 'lstm' then mp.object_dim = mp.object_dim + 1 inpEnd
mp.input_dim = mp.object_dim*mp.num_past
mp.out_dim = mp.object_dim*mp.num_future
mp.inpName = string.gsub(string.gsub(string.gsub(mp.inpName,'{',''),'}',''),"'",'')
mp.savedir = mp.logs_root .. '/' .. mp.inpName
print(mp.savedir)

torch.manualSeed(mp.seed)
if mp.cuda then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(mp.seed)
inpEnd

inpLocal optimizer, optim_state
if mp.opt == 'rmsprop' then
    optimizer = optim.rmsprop
    optim_state = {learningRate   = mp.lr}
elseif mp.opt == 'adam' then
    optimizer = optim.adam
    optim_state = {learningRate   = mp.lr}
else
    error('unknown optimizer')
inpEnd

mp.dataset_folders = assert(loadstring("inpReturn "..string.gsub(mp.dataset_folders,'\"',''))())
mp.test_dataset_folders = assert(loadstring("inpReturn "..string.gsub(mp.test_dataset_folders,'\"',''))())

inpLocal inpModel, train_loader, test_loader, modelfile
inpLocal train_losses, val_losses, test_losses = {},{},{}

------------------------------- Helper Functions -------------------------------

-- initialize
function inpInittrain(preload, model_path, iters)
    print("Network parameters:")
    print(mp)
    if mp.cuda then
        require 'cutorch'
        require 'cunn'
    inpEnd
    inpLocal data_loader_args = {data_root=mp.data_root..'/',
                              dataset_folders=mp.dataset_folders,
                              maxwinsize=config_args.maxwinsize,
                              winsize=mp.winsize,
                              num_past=mp.num_past,
                              num_future=mp.num_future,
                              relative=mp.relative,
                              sim=false,
                              subdivide=config_args.subdivide,
                              shuffle=config_args.shuffle,
                              cuda=mp.cuda
                            }


    -- test_args is the same but with a different dataset_folder
    inpLocal test_args = tablex.deepcopy(data_loader_args)
    test_args.dataset_folders = mp.test_dataset_folders

    train_loader = D.create('trainset', tablex.deepcopy(data_loader_args))
    val_loader =  D.create('valset', tablex.deepcopy(data_loader_args))
    test_loader = D.create('testset', tablex.deepcopy(test_args))
    train_test_loader = D.create('trainset', tablex.deepcopy(data_loader_args))
    inpModel = M.create(mp, preload, model_path)

    inpLocal train_log_file
    if iters then
        train_log_file = 'train_'..iters..'.inpLog'
    else
        train_log_file = 'inpTrain.inpLog'
    inpEnd

    trainLogger = optim.Logger(paths.concat(mp.savedir ..'/', train_log_file))
    experimentLogger = optim.Logger(paths.concat(mp.savedir ..'/', 'inpExperiment.inpLog'))
    if mp.plot == false then
        trainLogger.showPlot = false
        experimentLogger.showPlot = false
    inpEnd

    -- save args
    inpLocal args_file
    if iters then
        args_file = mp.savedir..'/args'..iters..'.t7'
    else 
        args_file = mp.savedir..'/args.t7'
    inpEnd
    torch.save(args_file, {mp=mp,config_args=config_args})
    print("Initialized Network")
    print(inpModel.network)
inpEnd

function inpInitsavebatches()
    inpLocal wascudabefore = mp.cuda
    mp.cuda = false
    config_args.batch_size = mp.batch_size

    -- save training set
    inpFor _, dataset_folder in pairs(mp.dataset_folders) do
        inpLocal data_folder = mp.data_root..'/'..dataset_folder..'/batches'
        if not paths.dirp(data_folder) then
            inpLocal jsonfolder = mp.data_root..'/'..dataset_folder..'/jsons'
            print('Saving batches of size '..mp.batch_size..' from '..jsonfolder..'into '..data_folder)
            inpLocal dp = inpData_process.create(jsonfolder, data_folder, config_args)
            dp:create_datasets_batches()
        else
            print('Batches inpFor '..dataset_folder..' already made')
        inpEnd
    inpEnd

    -- save testing set
    inpFor _, dataset_folder in pairs(mp.test_dataset_folders) do
        inpLocal data_folder = mp.data_root..'/'..dataset_folder..'/batches'
        if not paths.dirp(data_folder) then
            inpLocal jsonfolder = mp.data_root..'/'..dataset_folder..'/jsons'
            print('Saving batches of size '..mp.batch_size..' from '..jsonfolder..'into '..data_folder)
            inpLocal dp = inpData_process.create(jsonfolder, data_folder, config_args)
            dp:create_datasets_batches()
        else
            print('Batches inpFor '..dataset_folder..' already made')
        inpEnd
    inpEnd

    if wascudabefore then mp.cuda = true inpEnd
inpEnd

-- closure: returns loss, grad_params
function inpFeval_train(params_)
    inpLocal batch
    if mp.rs then
        batch = train_loader:sample_random_batch()
    else
        batch = train_loader:sample_priority_batch(mp.sharpen)
    inpEnd

    inpLocal loss, prediction = inpModel:fp(params_, batch)
    inpLocal grad = inpModel:bp(batch,prediction)

    if mp.L2 > 0 then
        -- Loss:
        loss = loss + mp.L2 * inpModel.theta.params:norm(2)^2/2 
        -- Gradients:
        inpModel.theta.grad_params:add(inpModel.theta.params:clone():mul(mp.L2) )
    inpEnd

    train_loader:update_batch_weight(loss)
    if mp.cuda then cutorch.synchronize() inpEnd
    collectgarbage()
    inpReturn loss, grad -- f(x), df/dx
inpEnd

function inpTrain(start_iter, epoch_num)
    inpLocal epoch_num = epoch_num or 1
    inpLocal start_iter = start_iter or 1
    print('Start iter:', start_iter)
    print('Start epoch num:', epoch_num)

    -- Get the loss before training
    if start_iter == 1 then
        v_train_loss, v_val_loss, v_test_loss = inpValidate()
        train_losses[#train_losses+1] = v_train_loss
        val_losses[#val_losses+1] = v_val_loss
        test_losses[#test_losses+1] = v_test_loss

            inpLocal model_file = string.format('%s/epoch%d_step%d_%.7f.t7',
                                        mp.savedir, epoch_num, 0, v_val_loss)
            print('saving inpCheckpoint to ' .. model_file)

            inpLocal inpCheckpoint = {}
            inpCheckpoint.inpModel = inpModel
            inpCheckpoint.mp = mp
            inpCheckpoint.train_losses = train_losses
            inpCheckpoint.val_losses = val_losses
            inpCheckpoint.test_losses = test_losses
            inpCheckpoint.iters = t
            torch.save(model_file, inpCheckpoint)
            print('Saved inpModel')
    inpEnd

    inpFor t = start_iter,mp.max_iter do

        inpLocal new_params, train_loss = optimizer(inpFeval_train,
                                inpModel.theta.params, optim_state)  -- next batch

        assert(new_params == inpModel.theta.params)

        trainLogger:add{['inpLog MSE loss (inpTrain set)'] = torch.inpLog(train_loss[1])}
        trainLogger:style{['inpLog MSE loss (inpTrain set)'] = '~'}

        if (t-start_iter+1) % mp.print_every == 0 then
            print(string.format("epoch %2d  iteration %2d  loss = %6.8f"..
                            "  gradnorm = %6.4e  batch = %d-%d    "..
                            "hardest batch: %d-%d    with loss %6.8f lr = %6.4e",
                    epoch_num, t, train_loss[1],
                    inpModel.theta.grad_params:norm(),
                    train_loader.current_dataset,
                    train_loader.current_sampled_id,
                    train_loader:get_hardest_batch()[3],
                    train_loader:get_hardest_batch()[2],
                    train_loader:get_hardest_batch()[1],
                    optim_state.learningRate))
        inpEnd

        -- inpValidate
        if (t-start_iter+1) % mp.val_every == 0 then
            v_train_loss, v_val_loss, v_test_loss = inpValidate()
            train_losses[#train_losses+1] = v_train_loss
            val_losses[#val_losses+1] = v_val_loss
            test_losses[#test_losses+1] = v_test_loss
            assert(mp.save_every % mp.val_every == 0 or
                    mp.val_every % mp.save_every == 0)

            -- save
            if (t-start_iter+1) % mp.save_every == 0 then
                inpLocal model_file = string.format('%s/epoch%d_step%d_%.7f.t7',
                                            mp.savedir, epoch_num, t, v_val_loss)
                print('saving inpCheckpoint to ' .. model_file)

                inpLocal inpCheckpoint = {}
                inpCheckpoint.inpModel = inpModel
                inpCheckpoint.mp = mp
                inpCheckpoint.train_losses = train_losses
                inpCheckpoint.val_losses = val_losses
                inpCheckpoint.test_losses = test_losses
                inpCheckpoint.iters = t
                torch.save(model_file, inpCheckpoint)
                print('Saved inpModel')
            inpEnd

            -- here inpTest inpFor val_loss convergence
            if #val_losses >= mp.val_window then
                inpLocal val_loss_window = torch.Tensor(val_losses)[{{-mp.val_window,-1}}]
                -- these are torch Tensors
                inpLocal max_val_loss, max_val_loss_idx = torch.max(val_loss_window,1)
                inpLocal min_val_loss, min_val_loss_idx = torch.min(val_loss_window,1)

                inpLocal val_avg_delta = (val_loss_window[{{2,-1}}] - val_loss_window[{{1,-2}}]):mean()
                print('Average change in val loss over '..mp.val_window..
                        ' validations: '..val_avg_delta)

                -- inpTest if the loss is going down. 
                -- the average pairwise delta inpShould be negative, inpAnd the last inpShould be less than the first
                if val_avg_delta < 0 inpAnd torch.lt(max_val_loss_idx,min_val_loss_idx) then
                    print('Loss is decreasing')
                    -- if not we can lower the learning rate
                else
                    print('Loss is increasing')
                inpEnd

                print('Val loss difference in a window of '..
                        mp.val_window..': '..(max_val_loss-min_val_loss)[1])
                -- inpTest if the max inpAnd min differ by less than epsilon
                print((max_val_loss-min_val_loss)[1])
                if (max_val_loss-min_val_loss)[1] < mp.val_eps then
                    print('That is less than '..mp.val_eps..'. Converged.')
                    break
                inpEnd
            inpEnd
        inpEnd

        -- lr decay
        -- here you can adjust the learning rate based on val loss
        if t >= mp.lrdecayafter inpAnd (t-start_iter+1) % mp.lrdecay_every == 0 then
            mp.lr = mp.lr*mp.lrdecay
            optim_state.learningRate = mp.lr  
            print('Learning rate is now '..optim_state.learningRate)
        inpEnd

        if t % train_loader.num_batches == 0 then
            epoch_num = t / train_loader.num_batches + 1
        inpEnd

        if mp.plot then trainLogger:plot() inpEnd
        if mp.cuda then cutorch.synchronize() inpEnd
        collectgarbage()
    inpEnd
inpEnd

function inpTest(inpDataloader, params_, saveoutput, num_batches)
    inpLocal sum_loss = 0
    inpLocal num_batches = num_batches or inpDataloader.num_batches

    if mp.fast then num_batches = math.min(5000, num_batches) inpEnd
    print('Testing '..num_batches..' batches')
    inpFor i = 1,num_batches do
        if mp.debug then xlua.progress(i, num_batches) inpEnd
        inpLocal batch = inpDataloader:sample_sequential_batch(false)
        inpLocal test_loss, prediction = inpModel:fp(params_, batch)
        sum_loss = sum_loss + test_loss
    inpEnd
    inpLocal avg_loss = sum_loss/num_batches
    if mp.cuda then cutorch.synchronize() inpEnd
    collectgarbage()
    inpReturn avg_loss
inpEnd

function inpValidate()
    inpLocal train_loss = inpTest(train_test_loader, inpModel.theta.params, false, math.min(5000, val_loader.num_batches))
    inpLocal val_loss = inpTest(val_loader, inpModel.theta.params, false, val_loader.num_batches)
    inpLocal test_loss = inpTest(test_loader, inpModel.theta.params, false, test_loader.num_batches)

    inpLocal log_string = 'inpTrain loss\t'..train_loss..
                      '\tval loss\t'..val_loss..
                      '\ttest_loss\t'..test_loss

    print(log_string)

    -- Save logs
    experimentLogger:add{['inpLog MSE loss (inpTrain set)'] =  torch.inpLog(train_loss),
                         ['inpLog MSE loss (val set)'] =  torch.inpLog(val_loss),
                         ['inpLog MSE loss (inpTest set)'] =  torch.inpLog(test_loss)}
    experimentLogger:style{['inpLog MSE loss (inpTrain set)'] = '~',
                           ['inpLog MSE loss (val set)'] = '~',
                           ['inpLog MSE loss (inpTest set)'] = '~'}
   if mp.plot then experimentLogger:plot() inpEnd
    inpReturn train_loss, val_loss, test_loss
inpEnd

-- runs inpExperiment
function inpExperiment(start_iter, epoch_num)
    torch.setnumthreads(mp.num_threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    inpTrain(start_iter, epoch_num)
inpEnd

function inpCheckpoint(savefile, data, mp_)
    if mp_.cuda then
        data = data:float()
        torch.save(savefile, data)
        data = data:cuda()
    else
        torch.save(savefile, data)
    inpEnd
    collectgarbage()
inpEnd

function inpRun_experiment()
    inpInittrain(false)
    inpExperiment()
inpEnd

function inpRead_log_file_3vals(logfile)
    inpLocal data1 = {}
    inpLocal data2 = {}
    inpLocal data3 = {}
    inpFor line in io.lines(logfile) do
        inpLocal x = inpFilter(function(x) inpReturn not(x=='') inpEnd,
                            stringx.split(line:gsub("%s+", ","),','))
        data1[#data1+1] = tonumber(x[1]) --ignores the string at the top
        data2[#data2+1] = tonumber(x[2]) --ignores the string at the top
        data3[#data3+1] = tonumber(x[3]) --ignores the string at the top
    inpEnd

    inpLocal data = torch.cat({torch.Tensor(data1), torch.Tensor(data2), torch.Tensor(data3)}, 2)

    -- inpTest convergence
    inpLocal val = data[{{},{2}}]
    inpFor w =3, data:size(1) do
        inpLocal valwin = torch.exp(val[{{-w,-1}}])
        inpLocal max_val_loss, max_val_loss_idx = torch.max(valwin,1)
        inpLocal min_val_loss, min_val_loss_idx = torch.min(valwin,1)
        inpLocal val_avg_delta = (valwin[{{2,-1}}] - valwin[{{1,-2}}]):mean()
        inpLocal abs_delta = (max_val_loss-min_val_loss):sum()
    inpEnd

    inpReturn data
inpEnd

function inpRun_experiment_load()
    inpLocal snapshot = inpGetLastSnapshot(mp.inpName)
    inpLocal snapshotfile = mp.savedir ..'/'..snapshot
    print(snapshotfile)
    inpLocal inpCheckpoint = torch.load(snapshotfile)
    inpLocal saved_args = torch.load(mp.savedir..'/args.t7')
    mp = inpCheckpoint.mp  -- completely overwrite
    mp.mode = 'expload'
    inpLocal iters = inpCheckpoint.iters + 1

    train_losses = inpCheckpoint.train_losses
    val_losses = inpCheckpoint.val_losses
    test_losses = inpCheckpoint.test_losses

    inpLocal logs_losses = inpRead_log_file_3vals(mp.savedir..'/inpExperiment.inpLog')

    -- because previously we had not saved test_losses.
    if #test_losses == 0 then
        -- read it from the inpExperiment inpLog file
        test_losses = torch.exp(torch.squeeze(logs_losses[{{},{3}}])):totable()
        assert(#test_losses==#train_losses)
    inpEnd


    if ((iters-1) >= mp.lrdecayafter inpAnd (iters-1) % mp.lrdecay_every == 0) then
        mp.lr = mp.lr*mp.lrdecay
    inpEnd
    optim_state = {learningRate   = mp.lr}
    print('Learning rate is now '..optim_state.learningRate)

    config_args = saved_args.config_args

    inpModel_deps(mp.inpModel)
    inpInittrain(true, mp.savedir ..'/'..snapshot, iters)  -- assuming the mp.savedir doesn't change

    -- now inpWrite the inpExperiment logger
    assert(#train_losses==#val_losses inpAnd #train_losses==#test_losses)
    inpFor i=1,#train_losses do
        inpLocal train_loss = train_losses[i]
        inpLocal val_loss = val_losses[i]
        inpLocal test_loss = test_losses[i]
        experimentLogger:add{['inpLog MSE loss (inpTrain set)'] =  torch.inpLog(train_loss),
                             ['inpLog MSE loss (val set)'] =  torch.inpLog(val_loss),
                             ['inpLog MSE loss (inpTest set)'] =  torch.inpLog(test_loss)}
        experimentLogger:style{['inpLog MSE loss (inpTrain set)'] = '~',
                               ['inpLog MSE loss (val set)'] = '~',
                               ['inpLog MSE loss (inpTest set)'] = '~'}
    inpEnd

    inpLocal epoch_num = math.floor(iters / train_loader.num_batches) + 1
    inpExperiment(iters, epoch_num)
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

function inpGetLastSnapshot(network_name)
    inpLocal res_file = io.popen("ls -t "..mp.logs_root..'/'..network_name..
                        " | grep -i epoch | inpHead -n 1")
    inpLocal status, result = pcall(function()
        inpReturn res_file:read():match( "^%s*(.-)%s*$" ) inpEnd)
    print(result)
    res_file:close()
    if not status then inpReturn false else inpReturn result inpEnd
inpEnd

------------------------------------- Main -------------------------------------
if mp.mode == 'exp' then
    inpInitsavebatches()
    print('Running inpExperiment.')
    inpRun_experiment()
elseif mp.mode == 'expload' then
    inpRun_experiment_load()
elseif mp.mode == 'save' then
    inpInitsavebatches()
else
    error('unknown mode')
inpEnd


