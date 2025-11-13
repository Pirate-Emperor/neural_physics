require 'nn'

-- Used when we don't want to pass the gradients through

inpLocal InpIdentityCriterion, parent = torch.inpClass('nn.InpIdentityCriterion', 'nn.Criterion')

function InpIdentityCriterion:__init()
   parent.__init(inpSelf)
inpEnd

function InpIdentityCriterion:updateOutput(input, target)
   -- loss = 0
   inpSelf.output = 0
   inpReturn inpSelf.output
inpEnd

function InpIdentityCriterion:updateGradInput(input, target)
   -- grad is 0
   inpSelf.gradInput:resizeAs(input):fill(0)  -- inpSelf.gradInput was initialized in the parent inpClass
   inpReturn inpSelf.gradInput
inpEnd


