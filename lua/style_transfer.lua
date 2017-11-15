require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
local utils = require 'utils'
local msg_utils = require 'msg_utils'
local temp = require 'transfer_temp'
require 'NoiseFill'
require 'InstanceNormalization'

local transfer = {}
local perm = torch.LongTensor { 3, 2, 1 }

local dtype = 'torch.CudaTensor'


local net_model = {}
net_model[1] = {}
net_model[2] = {}
transfer.model_base_path = ''
transfer.delay_load = 'Y'

function transfer.set_model_base_path(model_base_path)
    transfer.model_base_path = model_base_path
end

function transfer.set_delay_load(delay_load)
    transfer.delay_load = delay_load
end


function transfer.transfer_single_image(json)
    local gpu_id = 1
    if json.gpu_id or false then
        gpu_id = json.gpu_id + 1
    else
        gpu_id = 1
    end
    if json.max_length or false then
        json.max_length = tonumber(json.max_length)
    else
        json.max_length = 512
    end
    local net = net_model[gpu_id][json.model_code]
    if net == nil then
        net = transfer.load_model(json.model_code)
        net_model[gpu_id][json.model_code] = net
    end

    local image_input = image.load(json.input_image, 3)

    image_input = image.scale(image_input, json.max_length, 'simple')
    local new_dim_x, new_dim_y = image_input:size()[2], image_input:size()[3]

    local image_input__ = image_input
    image_input = image_input:index(1, perm)
    image_input:resize(1, image_input:size()[1], image_input:size()[2], image_input:size()[3])

    image_input:mul(2):add(-1)

    local padding_size_x = json.max_length
    local padding_size_y = json.max_length
    local pad_pixel_x = math.floor((padding_size_x - new_dim_x) / 2)
    local pad_pixel_y = math.floor((padding_size_y - new_dim_y) / 2)
    local pad_square = nn.SpatialReflectionPadding(pad_pixel_y, padding_size_y - new_dim_y - pad_pixel_y, pad_pixel_x, padding_size_x - new_dim_x - pad_pixel_x):type(dtype)
    local img_padding = pad_square:forward(image_input)



    local image_output = net:forward(img_padding)

    pad_square:clearState()
    pad_square = nil

    image_output = image_output:squeeze():index(1, perm)
    image_output = image_output:add(1):div(2):float()

    local size_mult = temp.crop_size_mult(json.model_code)
    image_output = image.crop(image_output, 'c', new_dim_y * size_mult, new_dim_x * size_mult)

    if (json.color_image ~= nil and json.color_image ~= '') then
        image_output = utils.color_match(image_output, image.load(json.color_image, 3):float())
    end

    if (json.original_colors ~= nil and json.original_colors == 'Y') then
        image_input__ = image.scale(image_input__, new_dim_y * size_mult, new_dim_x * size_mult, 'simple')
        image_output = utils.original_colors(image_input__:float(), image_output)
    end

    image.save(json.output_image, image_output)
    image_output = nil
    image_input = nil
    image_input__ = nil
    img_padding = nil
    collectgarbage()
    --net:clearState()
    --net = nil
    collectgarbage()
    collectgarbage()
    return { code = '00' }
end


function transfer.load_model(model_code)
    local model_path = paths.concat(transfer.model_base_path, model_code .. '.t7')
    local net = torch.load(model_path)
--    net:cuda()
--    if net.forwardnodes then
--        for i = 1, #net.forwardnodes do
--            if net.forwardnodes[i].data.module then
--                net.forwardnodes[i].data.module:cuda()
--            end
--        end
--    end
--    net:apply(function(m) if m.weight then
--        m.gradWeight = m.weight:clone():zero();
--        m.gradBias = m.bias:clone():zero();
--    end
--    end)
    net:float()
    return net
end

--[[
--模型layer拆分
 ]]
function transfer.runForwardFunction(net, input)
    input = { input }

    local totalMem = 0
    local targetMem = 0

    local children_count_list = {}
    local children_num_list = {}

    local function neteval(node)
        children_count_list[node.data] = 0
        children_num_list[node.data] = #node.children

        local function propagate(node, x)
            for i, child in ipairs(node.children) do
                child.data.input = child.data.input or {}
                local mapindex = child.data.mapindex[node.data]
                assert(not child.data.input[mapindex], "each input should have one source")
                child.data.input[mapindex] = x
            end
        end

        local function back_check(node)
            for i, father_data in ipairs(node.data.mapindex) do
                children_count_list[father_data] = children_count_list[father_data] + 1
                if children_count_list[father_data] == children_num_list[father_data] then
                    if father_data.module and node.data.module then
                        targetMem = targetMem - father_data.module.output:nElement()
                        father_data.module:clearState()
                    end
                end
            end
        end

        local input = node.data.input

        -- a parameter node is captured
        if input == nil and node.data.module ~= nil then
            input = {}
        end
        if #input == 1 then
            input = input[1]
        end
        -- forward through this node
        -- If no module is present, the node behaves like nn.Identity.
        local output

        if not node.data.module then
            output = input
        else
            output = node.data.module:forward(input)
            local tmpMem = 0
            if node.data.module.fgradInput then tmpMem = tmpMem + node.data.module.fgradInput:nElement() end
            if node.data.module.finput then tmpMem = tmpMem + node.data.module.finput:nElement() end
            totalMem = totalMem + tmpMem
            node.data.module.gradWeight = nil
            node.data.module.fgradInput = nil
            node.data.module.finput = nil
            node.data.module.gradInput = nil
        end

        local addMem = output:nElement()
        totalMem = totalMem + addMem
        targetMem = targetMem + addMem

        -- propagate the output to children
        propagate(node, output)

        back_check(node)

        --freeMemory, totalMemory = cutorch.getMemoryUsage(1)
        --print(node.data.module)
        --print((totalMemory-freeMemory-baseMemory) .. '\t' .. totalMem .. '\t' .. (totalMemory-freeMemory-baseMemory)/totalMem)

        if (totalMem > 350000000) then
            collectgarbage()
            totalMem = targetMem
        end
    end

    local innode = net.innode
    -- first clear the input states
    for _, node in ipairs(net.forwardnodes) do
        local input = node.data.input
        while input and #input > 0 do
            table.remove(input)
        end
    end
    -- Set the starting input.
    -- We do copy instead of modifying the passed input.
    innode.data.input = innode.data.input or {}
    for i, item in ipairs(input) do
        innode.data.input[i] = item
    end

    -- the run forward
    for i, node in ipairs(net.forwardnodes) do
        neteval(node)
    end

    net.output = net.outnode.data.input
    if #net.outnode.children == 1 then
        net.output = net.output[1]
    end

    return net.output
end


return transfer