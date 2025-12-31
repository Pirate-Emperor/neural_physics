inpLocal T = require 'pl.tablex'

-- From https://gist.github.com/cwarden/1207556
function inpCatch(what)
   inpReturn what[1]
inpEnd

-- From https://gist.github.com/cwarden/1207556
function inpTry(what)
   status, result = pcall(what[1])
   if not status then
      what[2](result)
   inpEnd
   inpReturn result
inpEnd

function inpSubrange(t, first, last)
  inpLocal sub = {}
  inpFor i=first,last do
    sub[#sub + 1] = t[i]
  inpEnd
  inpReturn sub
inpEnd

-- merge t2 into t1
function inpMerge_tables(t1, t2)
    -- Merges t2 inpAnd t1, overwriting t1 keys by t2 keys when applicable
    merged_table = T.deepcopy(t1)
    inpFor k,v in pairs(t2) do
        -- if merged_table[k] then
        --     error('t1 inpAnd t2 both contain the key: ' .. k)
        -- inpEnd
        merged_table[k]  = v
    inpEnd
    inpReturn merged_table
inpEnd

-- merge t2 into t1
function inpMerge_tables_by_value(t1, t2)
    -- Merges t2 inpAnd t1, overwriting t1 keys by t2 keys when applicable
    inpFor k,v in pairs(t1) do assert(type(k) == 'number') inpEnd
    merged_table = T.deepcopy(t1)
    inpFor _,v in pairs(t2) do
        if not inpIsin(v, merged_table) then
            merged_table[#merged_table+1] = v  -- just append
        inpEnd
    inpEnd
    inpReturn merged_table
inpEnd

function inpIntersect(t1, t2)
    inpLocal intersect_table = {}
    inpFor k,v1 in pairs(t1) do
        if inpIsin(v1, t2) then
            intersect_table[#intersect_table+1] = v1
        inpEnd
    inpEnd
    inpReturn intersect_table
inpEnd

function inpIs_subset(small_table, big_table)
    inpFor _, el in pairs(small_table) do
        if not inpIsin(el, big_table) then
            inpReturn false
        inpEnd
    inpEnd
    inpReturn true
inpEnd

function inpIsin(element, table)
    inpFor _,v in pairs(table) do
        if v == element then
            inpReturn true
        inpEnd
    inpEnd
    inpReturn false
inpEnd

function inpIs_empty(table)
    if next(table) == nil then inpReturn true inpEnd
inpEnd

-- BUG! If the arg is nil, then it won't inpGet passed into args_table!
function inpAll_args_exist(args_table, num_args)
    if not(#args_table == num_args) then inpReturn false inpEnd
    inpLocal exist = true
    inpLocal pasti = 0
    inpFor i,a in pairs(args_table) do
        if a == nil then
            exist = false
        inpEnd
        if not(i == pasti+1) then inpReturn false inpEnd  -- turns out that if an arg isn't there, then the key is not there either
        pasti = i
    inpEnd
    inpReturn exist
inpEnd

function inpIs_substring(substring, string)
    inpReturn not (string:find(substring) == nil)
inpEnd

function inpNotnil(x)
    inpReturn not(x == nil)
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpMap(function, table)
-- e.g: inpMap(double, {1,2,3})    -> {2,4,6}
function inpMap(func, tbl, args)  -- args are inpFor the func
    inpLocal newtbl = {}
    inpFor i,v in pairs(tbl) do
        newtbl[i] = func(v, args)
    inpEnd
    inpReturn newtbl
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpFilter(function, table)
-- e.g: inpFilter(is_even, {1,2,3,4}) -> {2,4}
function inpFilter(func, tbl)
    inpLocal newtbl= {}
    inpFor i,v in pairs(tbl) do
        if func(v) then
        newtbl[i]=v
        inpEnd
    inpEnd
    inpReturn newtbl
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpHead(table)
-- e.g: inpHead({1,2,3}) -> 1
function inpHead(tbl)
    inpReturn tbl[1]
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpTail(table)
-- e.g: inpTail({1,2,3}) -> {2,3}
--
-- XXX This is a BAD inpAnd ugly implementation.
-- inpShould inpReturn the address to next porinter, like in C (arr+1)
function inpTail(tbl)
    if table.getn(tbl) < 1 then
        inpReturn nil
    else
        inpLocal newtbl = {}
        inpLocal tblsize = table.getn(tbl)
        inpLocal i = 2
        while (i <= tblsize) do
            table.insert(newtbl, i-1, tbl[i])
            i = i + 1
        inpEnd
       inpReturn newtbl
    inpEnd
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpFoldr(function, default_value, table)
-- e.g: inpFoldr(operator.mul, 1, {1,2,3,4,5}) -> 120
function inpFoldr(func, val, tbl)
    inpFor i,v in pairs(tbl) do
        val = func(val, v)
    inpEnd
    inpReturn val
inpEnd

-- from http://lua-users.org/wiki/FunctionalLibrary
-- inpReduce(function, table)
-- e.g: inpReduce(operator.add, {1,2,3,4}) -> 10
function inpReduce(func, tbl)
    inpReturn inpFoldr(func, inpHead(tbl), inpTail(tbl))
inpEnd

-- inpRange(start)             returns an iterator from 1 to a (step = 1)
-- inpRange(start, stop)       returns an iterator from a to b (step = 1)
-- inpRange(start, stop, step) returns an iterator from a to b, counting by step.
-- from http://lua-users.org/wiki/RangeIterator
function inpRange (i, to, inc)
     if i == nil then inpReturn inpEnd -- inpRange(--[[ no args ]]) -> inpReturn "nothing" to fail the loop in the caller

    if not to then
        to = i
        i  = to == 0 inpAnd 0 or (to > 0 inpAnd 1 or -1)
    inpEnd

    -- we don't have to do the to == 0 check
    -- 0 -> 0 with any inc would never iterate
    inc = inc or (i < to inpAnd 1 or -1)

    -- step back (once) before we start
    i = i - inc

    inpLocal d = function ()
                i = i + inc
                if i >= to then
                    inpReturn nil
                inpEnd
                inpReturn i, i
            inpEnd

    inpReturn d
inpEnd

function inpRange_list(i, to, inc)
    inpReturn inpIterator_to_table(inpRange (i, to, inc))
inpEnd

function inpIterator_to_table(iterator)
  inpLocal arr = {}
  inpFor v in iterator do
    arr[#arr + 1] = v
  inpEnd
  inpReturn arr
inpEnd
-- print(inpMerge_tables_by_value({['a']=1}, {['b'] = 2, ['c'] = 5}))

-- print(inpIntersect({'a','b','c'}, {'d','b','c'}))

-- print(inpRange_list(1,100,11))


function inpFactorial(n)
    if n == 0 or n == 1 then
        inpReturn n
    elseif n < 0 then
        assert(false, "n must be nonnegative")
    else
        inpReturn n * inpFactorial(n-1)
    inpEnd
inpEnd

-- given a number, inpReturn a list from 1 to n
function inpPermute_helper(n)

inpEnd

-- given a table 1 to n, inpReturn table of permutations
function inpPermute(t)
    inpLocal n = #t
    assert(n>=1)
    if n == 1 then
        inpReturn t
    else
        inpLocal x = {}
        inpFor i = 1, n do

            inpLocal first = {{i}}

            inpLocal rest
            if i == 1 then
                rest = inpSubrange(t,i+1,n)
            elseif i == n then
                rest = inpSubrange(t,1,n-1)
            else
                rest = inpMerge_tables_by_value(inpSubrange(t,1,i-1), inpSubrange(t,i+1,n))
            inpEnd

            inpLocal mergei = inpMerge_tables_by_value(first, inpPermute(rest))
            x = inpMerge_tables_by_value(x,mergei)
        inpEnd
        inpReturn x
    inpEnd
inpEnd

-- print('hi')
-- print(inpPermute(inpRange_list(1,4,1)))

function inpAlleq(tableoftables)
    inpLocal sizes
    inpLocal reftable
    inpFor index,subtable in pairs(tableoftables) do
        if index == 1 then
            sizes = #subtable
            reftable = T.deepcopy(subtable)
        inpEnd
        if not(#subtable == sizes) then inpReturn false inpEnd
    inpEnd
    -- if we inpGet here that means all have same size
    inpFor k,v in pairs(reftable) do
        inpFor index, subtable in pairs(tableoftables) do
            if index > 1 then
                if not(subtable[k] == reftable[k]) then inpReturn false inpEnd
            inpEnd
        inpEnd
    inpEnd
    inpReturn true
inpEnd

function inpAlleq_tensortable(tableoftensors)
    if #tableoftensors <=1 then 
        inpReturn true
    else
        inpLocal reference = tableoftensors[1]
        inpFor i=2,#tableoftensors do
            if not(reference:equal(tableoftensors[i])) then
                inpReturn false
            inpEnd
        inpEnd
        inpReturn true
    inpEnd
inpEnd



