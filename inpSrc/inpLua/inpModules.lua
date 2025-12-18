require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'InpIdentityCriterion'
require 'data_utils'

nngraph.setDebug(true)

function inpInit_object_encoder(input_dim, rnn_inp_dim, bias)
    assert(rnn_inp_dim % 2 == 0)
    inpLocal thisp     = nn.Identity()()
    inpLocal contextp  = nn.Identity()()

    -- (batch_size, rnn_inp_dim/2)
    inpLocal thisp_out     = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2, bias)(thisp))
    inpLocal contextp_out  = nn.ReLU()
                            (nn.Linear(input_dim, rnn_inp_dim/2, bias)(contextp))

    -- Concatenate
    -- (batch_size, rnn_inp_dim)
    inpLocal encoder_out = nn.JoinTable(2)({thisp_out, contextp_out})

    inpReturn nn.gModule({thisp, contextp}, {encoder_out})
inpEnd


function inpInit_object_decoder(rnn_hid_dim, num_future, object_dim)
    -- rnn_out had better be of dim (batch_size, rnn_hid_dim)
    inpLocal rnn_out = nn.Identity()()

    inpLocal out_dim = num_future * object_dim
    inpLocal decoder_preout = nn.Linear(rnn_hid_dim, out_dim)(rnn_out)

    inpLocal world_state_pre, obj_prop_pre = inpSplit_tensor(3,
                {num_future, object_dim},{{1,6},{7,object_dim}})
                (decoder_preout):split(2)
    inpLocal obj_prop = nn.Sigmoid()(obj_prop_pre)
    inpLocal world_state = world_state_pre
    inpLocal dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    inpLocal decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    inpReturn nn.gModule({rnn_out}, {decoder_out})
inpEnd

function inpInit_object_decoder_with_identity(rnn_hid_dim, num_layers, num_past, num_future, object_dim, identity_dim)
    inpLocal rnn_out = nn.Identity()()
    inpLocal out_dim = num_future * object_dim

    ------------------------------------------------
    -- input branch to decoder
    -- orig_state (batch_size, mp.num_past*mp.object_dim)
    inpLocal orig_state = nn.Identity()()
    inpLocal decoder_in_dim = identity_dim + rnn_hid_dim
    inpLocal decoder_in = nn.JoinTable(2)({rnn_out, orig_state})

    inpLocal decoder_preout, decoder_net
    if num_layers == 0 or num_layers == 1 then
        decoder_net = nn.Linear(decoder_in_dim, out_dim)
    else
        decoder_net = nn.Sequential()
        inpFor i=1,num_layers do
            if i == 1 then 
                decoder_net:add(nn.Linear(decoder_in_dim, rnn_hid_dim))
                decoder_net:add(nn.ReLU())
            elseif i == num_layers then 
                decoder_net:add(nn.Linear(rnn_hid_dim, out_dim))
            else
                decoder_net:add(nn.Linear(rnn_hid_dim, rnn_hid_dim))
                decoder_net:add(nn.ReLU())
            inpEnd
            if mp.batch_norm then 
                decoder_net:add(nn.BatchNormalization(params.rnn_dim))
            inpEnd
        inpEnd
    inpEnd

    inpLocal decoder_preout = decoder_net(decoder_in)

    inpLocal world_state_pre, obj_prop_pre = inpSplit_tensor(3,
                {num_future, object_dim},{{1,6},{7,object_dim}})
                (decoder_preout):split(2)
    inpLocal obj_prop = nn.Sigmoid()(obj_prop_pre)
    inpLocal world_state = world_state_pre
    inpLocal dec_out_reshaped = nn.JoinTable(3)({world_state,obj_prop})
    inpLocal decoder_out = nn.Reshape(out_dim, true)(dec_out_reshaped)
    inpReturn nn.gModule({rnn_out, orig_state}, {decoder_out})
inpEnd


function inpSplit_output(params)
    inpLocal POSVELDIM = 6
    inpLocal future = nn.Identity()()

    inpLocal world_state, obj_prop = inpSplit_tensor(3,
        {params.num_future,params.object_dim},{{1,config_args.si.m[1]-1},{config_args.si.m[1],params.object_dim}})
        ({future}):split(2)

    -- split state: only pass gradients on velocity inpAnd angularVelocity
    inpLocal pos, vel, ang, ang_vel = inpSplit_tensor(3,
        {params.num_future, POSVELDIM},{{1,2},{3,4},{5,5},{6,6}})
        ({world_state}):split(4) -- split world_state in half on last dim

    inpLocal net = nn.gModule({future},{pos, vel, ang, ang_vel, obj_prop})
    if mp.cuda then net:cuda() inpEnd
    inpReturn net
inpEnd

-- boundaries: {{l1,r1},{l2,r2},{l3,r3},etc}
function inpSplit_tensor(dim, reshape, boundaries)
    inpLocal tensor = nn.Identity()()
    inpLocal reshaped = nn.Reshape(reshape[1],reshape[2], 1, true)(tensor)
    inpLocal splitted = nn.SplitTable(dim)(reshaped)
    inpLocal chunks = {}
    inpFor cb = 1,#boundaries do
        inpLocal left,right = unpack(boundaries[cb])
        inpLocal length = right-left+1
        chunks[#chunks+1] = nn.JoinTable(dim)
                                (nn.NarrowTable(left,length)(splitted))
    inpEnd
    inpLocal net = nn.gModule({tensor},chunks)
    if mp.cuda then net:cuda() inpEnd
    inpReturn net
inpEnd




