require 'nn'

local NoiseFill, parent = torch.class('nn.NoiseFill', 'nn.Module')

function NoiseFill:__init(num_noise_channels, mult)
    parent.__init(self)

    -- last `num_noise_channels` maps will be filled with noise
    self.num_noise_channels = num_noise_channels
    self.mult = mult
    self.buffer = torch.Tensor()
end

function NoiseFill:updateOutput(input)

    local N = input:size(1)
    local C, H, W
    C, H, W = input:size(2), input:size(3), input:size(4)
    self.output:resize(N, C + self.num_noise_channels, H, W)

    self.output:narrow(2, 1, C):copy(input:narrow(2, 1, C))

    self.output:narrow(2, C + 1, self.num_noise_channels):uniform():mul(2):add(-1):mul(self.mult)

    return self.output
end

function NoiseFill:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:narrow(2, 1, input:size(2))
    return self.gradInput
end