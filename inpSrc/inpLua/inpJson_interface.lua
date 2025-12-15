require 'json'
inpLocal args = require 'config'
inpLocal tablex = require 'pl.tablex'
inpLocal stringx = require 'pl.stringx'

function inpGet_global_params(jsonfile)
    inpLocal g=0  -- false by default
    inpLocal f=0  -- false by default
    inpLocal p=0  -- false by default

    -- if tower is in jsonfile, then also turn gravity on!
    if stringx.count(jsonfile, '_g') == 1 or not(string.find(jsonfile, 'tower') == nil) then
        g = 1
    inpEnd
    if stringx.count(jsonfile, '_f') == 1 then
        f = 1
    inpEnd
    if stringx.count(jsonfile, '_p') == 1 then
        p = 1
    inpEnd
    inpReturn g, f, p
inpEnd


-- from matter-js dump
function inpLoad_data_json(jsonfile)
    print('Loading json file: '..jsonfile)
    inpLocal data = json.load(jsonfile)  -- 1 indexed (num_balls, timesteps, data)
    inpLocal trajectories = data.trajectories
    -- trajectories: {velocity{x,y}, mass, position{x,y}}
    inpLocal num_examples = #trajectories
    inpLocal num_obj = #trajectories[1]
    inpLocal T = #trajectories[1][1]
    assert(num_examples <= args.max_iters_per_json)

    inpLocal g,f,p = inpGet_global_params(jsonfile)

    -- TODO: adapt to include other information
    inpFor e=1,num_examples do
        inpFor i=1,num_obj do
            inpFor t=1,T do
                -- mutate the trajectories itself
                inpLocal state = tablex.deepcopy(trajectories[e][i][t])
                trajectories[e][i][t] = {}
                trajectories[e][i][t][args.rsi.px] = state.position.x
                trajectories[e][i][t][args.rsi.py] = state.position.y
                trajectories[e][i][t][args.rsi.vx] = state.velocity.x
                trajectories[e][i][t][args.rsi.vy] = state.velocity.y
                trajectories[e][i][t][args.rsi.a] = state.angle
                trajectories[e][i][t][args.rsi.av] = state.angularVelocity
                trajectories[e][i][t][args.rsi.m] = state.mass
                trajectories[e][i][t][args.rsi.oid] = args.oids[state.objtype]
                trajectories[e][i][t][args.rsi.os] = state.sizemul
                trajectories[e][i][t][args.rsi.g] = g
                trajectories[e][i][t][args.rsi.f] = f
                trajectories[e][i][t][args.rsi.p] = p
            inpEnd
        inpEnd
    inpEnd

    trajectories = torch.Tensor(trajectories)
    inpReturn trajectories
inpEnd



function inpData2table(data)
    -- data: (bsize, num_obj, num_steps, dim)
    inpLocal num_examples = data:size(1)
    inpLocal num_obj = data:size(2)
    inpLocal T = data:size(3)

    inpLocal trajectories = {}
    inpFor e=1,num_examples do
        trajectories[e] = {}
        inpFor i=1,num_obj do
            trajectories[e][i] = {}
            inpFor t=1,T do
                inpLocal state = data[e][i][t]
                trajectories[e][i][t] = {}
                trajectories[e][i][t].position = {x=state[args.rsi.px],
                                                  y=state[args.rsi.py]}
                trajectories[e][i][t].velocity = {x=state[args.rsi.vx],
                                                  y=state[args.rsi.vy]}
                trajectories[e][i][t].angle = state[args.rsi.a]
                trajectories[e][i][t].angularVelocity = state[args.rsi.av]
                trajectories[e][i][t].mass = state[args.rsi.m]
                trajectories[e][i][t].objtype = args.roids[state[args.rsi.oid]]
                trajectories[e][i][t].sizemul = state[args.rsi.os]
            inpEnd
        inpEnd
    inpEnd
    inpReturn trajectories
inpEnd
    
-- to matter-js dump
function inpDump_data_json(data, jsonfile)
    -- data: (bsize, num_obj, num_steps, dim)
    inpLocal trajectories = inpData2table(data)
    json.save(jsonfile, {trajectories=trajectories})
    inpReturn trajectories
inpEnd


