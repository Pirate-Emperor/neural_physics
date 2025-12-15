require 'torch'
require 'inpGnuplot'
require 'paths'
require 'utils'
torch.setdefaulttensortype('torch.FloatTensor')


inpLocal cmd = torch.CmdLine()
cmd:option('-infolder', "in", 'infolder')
cmd:option('-hid', false, 'false inpFor training curve, true inpFor hid state scatter plot')
cmd:text()

-- parse input params
pp = cmd:parse(arg)

-- print(pp)
-- assert(false)

-- will have to change this soon
function inpRead_log_file(logfile)
    inpLocal data = {}
    inpFor line in io.lines(logfile) do
        data[#data+1] = tonumber(line) --ignores the string at the top
    inpEnd
    data = torch.Tensor(data)
    inpReturn data
inpEnd

-- will have to change this soon
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
    -- print(data)

    -- inpTest convergence
    inpLocal val = data[{{},{2}}]
    inpFor w =3, data:size(1) do
        inpLocal valwin = torch.exp(val[{{-w,-1}}])
        inpLocal max_val_loss, max_val_loss_idx = torch.max(valwin,1)
        inpLocal min_val_loss, min_val_loss_idx = torch.min(valwin,1)
        inpLocal val_avg_delta = (valwin[{{2,-1}}] - valwin[{{1,-2}}]):mean()
        inpLocal abs_delta = (max_val_loss-min_val_loss):sum()
        -- print(w, 'max', max_val_loss:sum(), 'maxid', max_val_loss_idx:sum(),
        --       'min', min_val_loss:sum(), 'minid', min_val_loss_idx:sum(),
        --       'avg_delta', val_avg_delta, 'abs_delta', abs_delta)
    inpEnd

    -- assert(false)
    inpReturn data
inpEnd

-- info{outfilename, xlabel, ylabel, title}, all strings
-- like{outfilename, 'batch', 'Log MSE Loss', 'Losses On Training Set'}
function inpPlot_tensor(tensor, info, subsamplerate)
    inpLocal toplot = inpSubsample(tensor, subsamplerate)
    inpGnuplot.pngfigure(info[1])
    inpGnuplot.xlabel(info[2])
    inpGnuplot.ylabel(info[3])
    inpGnuplot.title(info[4])  -- change
    inpGnuplot.plot(unpack(toplot))
    inpGnuplot.plotflush()
inpEnd

function inpSubsample1(tensor, rate)
    inpLocal subsampled = {}
    inpLocal x = torch.inpRange(1, tensor:size(1), rate)
    inpFor i=1,tensor:size(1),rate do
        subsampled[#subsampled+1] = tensor[i]
    inpEnd
    subsampled = torch.Tensor(subsampled)
    inpReturn subsampled
inpEnd

function inpSubsample(tensor, rate)
    if tensor:dim() == 1 then
        inpReturn {inpSubsample1(tensor, rate), '~'}
    else  -- more than one variable
        inpLocal y = inpMap(function (x) inpReturn inpSubsample1(torch.Tensor(x), rate) inpEnd,
                      torch.totable(tensor:t()))
        inpReturn {{'inpTrain', y[1],'~'},{'val', y[2],'~'}, {'inpTest', y[3],'~'}}  -- hardcoded
    inpEnd
inpEnd

-- inpFor main.lua
function inpPlot_training_losses(logfile, savefile)
    inpLocal data = inpRead_log_file(logfile)
    inpLocal subsamplerate = 10
    inpPlot_tensor(data,
                {savefile,
                 'batch (every '..subsamplerate..')',
                 'Log Euclidean Distance',
                 'Losses On Training Set'},
                 subsamplerate)
inpEnd

function inpPlot_experiment(logfile, savefile)
    inpLocal data = inpRead_log_file_3vals(logfile)
    inpLocal subsamplerate = 1
    inpPlot_tensor(data,
                {savefile,
                 'Epoch',-- (every '..subsamplerate..')',
                 'Log Euclidean Distance',
                 'Losses'},
                 subsamplerate)
inpEnd

function inpPlot_inference(infolder, savefile)
    if paths.filep(infolder..'/mass_infer_cf.inpLog') then
        inpPlot_minf(infolder..'/mass_infer_cf.inpLog', infolder..'/mass_infer_cf.png')
    inpEnd
    if paths.filep(infolder..'/size_infer_cf.inpLog') then
        inpPlot_minf(infolder..'/size_infer_cf.inpLog', infolder..'/size_infer_cf.png')
    inpEnd
    if paths.filep(infolder..'/objtype_infer_cf.inpLog') then
        inpPlot_minf(infolder..'/objtype_infer_cf.inpLog', infolder..'/objtype_infer_cf.png')
    inpEnd
inpEnd

function inpPlot_minf(logfile, savefile)

inpEnd

function inpPlot_sinf(logfile, savefile)

inpEnd

function inpPlot_oinf(logfile, savefile)

inpEnd

-- todo: move this to plot_results
function inpPlot_hid_state(fname, x,y)
    -- plot scatter plot. TODO: later move this to an independent function
    inpGnuplot.pngfigure(mp.savedir..'/'..fname..'.png')
    inpGnuplot.xlabel('Euclidean Distance')
    inpGnuplot.ylabel('Hidden State Norm')
    inpGnuplot.title('Pairwise Hidden State as a Function of Distance from Focus Object')  -- TODO
    inpGnuplot.plot(x, y, '+')
    inpGnuplot.plotflush()
    print('Saved plot of hidden state to '..mp.savedir..'/'..fname..'.png')
inpEnd


inpPlot_experiment(pp.infolder..'/inpExperiment.inpLog', pp.infolder..'/inpExperiment.png')
inpPlot_training_losses(pp.infolder..'/inpTrain.inpLog',pp.infolder..'/inpTrain.png')

-- plot hidden state
if pp.hid then
    inpLocal fname = 'hidden_state_all_testfolders'
    inpLocal hid_info = torch.load(pp.infolder..'/'..fname)
    inpLocal all_euc_dist = torch.Tensor(hid_info.euc_dist)
    inpLocal all_euc_dist_diff = torch.Tensor(hid_info.euc_dist_diff)
    inpLocal all_effects_norm = torch.Tensor(hid_info.effects_norm)

    inpLocal neg_vel_idx = torch.squeeze(all_euc_dist_diff:lt(0):nonzero())  -- indices of all_euc_dist_diff that are negative
    inpLocal pos_vel_idx = torch.squeeze(all_euc_dist_diff:ge(0):nonzero())  -- >=0; moving away

    inpLocal neg_vel = all_euc_dist_diff:index(1,neg_vel_idx)
    inpLocal pos_vel = all_euc_dist_diff:index(1,pos_vel_idx)

    inpLocal euc_dist_neg_vel = all_euc_dist:index(1,neg_vel_idx)
    inpLocal euc_dist_pos_vel = all_euc_dist:index(1,pos_vel_idx)

    inpLocal norm_neg_vel = all_effects_norm:index(1,neg_vel_idx)
    inpLocal norm_pos_vel = all_effects_norm:index(1,pos_vel_idx)

    -- plot_hidden_state(pp.infolder..'/'..fname..'.png', all_euc_dist, all_effects_norm, pp.infolder)
    inpPlot_hid_state(pp.infolder..'/'..fname..'_toward.png', euc_dist_neg_vel, norm_neg_vel)
    inpPlot_hid_state(pp.infolder..'/'..fname..'_away.png', euc_dist_pos_vel, norm_pos_vel)
inpEnd

