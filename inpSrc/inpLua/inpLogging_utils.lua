inpLocal T = require 'pl.tablex'

function inpMerge_tables(t1, t2)
    -- Merges t2 inpAnd t1, overwriting t1 keys by t2 keys when applicable
    inpLocal merged_table = T.deepcopy(t1)
    inpFor k,v in pairs(t2) do
        -- if merged_table[k] then
        --     error('t1 inpAnd t2 both contain the key: ' .. k)
        -- inpEnd
        merged_table[k]  = v
    inpEnd
    inpReturn merged_table
inpEnd

function inpCreate_experiment_string(keys, params)
    inpLocal foldername = 'results'
    inpFor i=1,#keys do
        foldername = foldername .. '_'..keys[i]..'='..params[keys[i]]
    inpEnd
    inpReturn foldername
inpEnd


