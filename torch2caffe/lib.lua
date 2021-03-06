--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local M = {}

require 'nn'
require 'image'

local pl = require 'pl.import_into'()
local py = require 'fb.python'
local torch = require 'torch'
require 'fbtorch'
local torch_layers = require 'torch2caffe.torch_layers'
local t2c = py.import('torch2caffe.lib_py')

local utils = require 'torch2caffe.utils'
local preprocess = require 'torch2caffe.preprocess'

function M.evaluate_caffe(caffe_net, inputs)
    local input_kwargs = {}
    for i=1,#inputs do
        local input_spec = inputs[i]
        input_kwargs[input_spec.name] = input_spec.tensor
    end
    print("start forward caffe")
    print(input_kwargs)
    local py_caffe_output = caffe_net.forward(py.kwargs, input_kwargs)
    print("start to reval output")
    local caffe_output_list = py.reval(t2c.format_output(py_caffe_output))
    print("start to val outputlist")
    local caffe_output_length = py.eval("len(a)", {a=caffe_output_list})
    print("val output success")
    --print(caffe_output_length)
    local caffe_outputs = {}
    for i=0,caffe_output_length-1 do
        table.insert(caffe_outputs,
                     torch.FloatTensor(
                         torch.totable(py.eval(caffe_output_list[i]))))
    end
    return caffe_outputs
end

local function debug_nets(caffe_net, torch_net)
    py.reval(t2c.debug_net(caffe_net))
    torch_net:apply(
        function(m)
--            print("debug torch")
            if m.output then
--                print("print torch_net.output")
                local sizes = {}
                local sums = {}
                if type(m.output) == 'table' then
                    for i=1,#m.output do
                        table.insert(sizes, m.output[i]:size())
                        table.insert(sums, torch.sum(m.output[i]))
                    end
                else
                    sizes = torch.totable(m.output:size())
                    sums = torch.sum(m.output)
                end
                --logging.infof("Layer %s, %s, Sum: %s", torch.typename(m), sizes, sums)
--                print(("Layer %s, %s, Sum: %s").format(torch.typename(m), sizes, sums))
                print(torch.typename(m))
--                print(sizes)
--                print(m.output)
                print(sums)
            end
        end
    )
end

function M.compare(opts, torch_net)
    if not opts.test then
        torch_net:apply(function(m) m:evaluate() end)
    end

    print("compare...1")
    local inputs = {}
    for i=1,#opts.inputs do
        local input_spec = opts.inputs[i]
        local tensor
--        if input_spec.tensor then
--            tensor = input_spec.tensor
--        else
--            tensor = torch.rand(table.unpack(input_spec.input_dims)):float()
--        end

--        tensor = torch.Tensor(1, 3, 256, 256)
--        s = tensor:storage()
--        for i=1,s:size() do -- fill up the Storage
--          s[i] = 1
--        end
--        print("tensor input : ", tensor)

        -- input a image
        local img = image.load(opts.imgpath, 3)
        if opts.image_size > 0 then
          img = image.scale(img, opts.image_size, opts.image_size, 'bilinear')
        end
        local H, W = img:size(2), img:size(3)
        print("size")
        print(H)
        print(W)
        tensor = img:view(1, 3, H, W)

        table.insert(inputs, {name=input_spec.name, tensor=tensor})
    end

    print("compare...2", inputs)
    -- Legacy code
    if opts.input_tensor then
        assert(inputs[1].name == "data")
        inputs[1].tensor = opts.input_tensor
    end

    print("compare...3")
    local caffe_net = t2c.load(opts)
    print("compare...3 result")
    --print(inputs)
    local caffe_outputs = M.evaluate_caffe(caffe_net, inputs)

    print("compare...4")
    -- Torch multi-inputs take an ordered Table.
    local function inputs_to_torch_inputs(inputs, type)
        if #inputs == 1 then
            return inputs[1].tensor:type(type)
        end
        local tensors = {}
        for i=1,#inputs do
            table.insert(tensors, inputs[i].tensor:type(type))
        end
        return tensors
    end
    local torch_outputs
    print("compare...5")
    -- Some networks only accept CUDA input.
    local ok, err = pcall(function()
            torch_net:float()
            local torch_inputs = inputs_to_torch_inputs(
                inputs, 'torch.FloatTensor')
            torch_outputs = torch_net:forward(torch_inputs)
    end)
    print("compare...6")
    if not ok then
        --logging.infof("Got error running forward: %s", err)
        print(("Got error running forward: %s").format(err))
        torch_net:cuda()
        local torch_inputs = inputs_to_torch_inputs(
            inputs, 'torch.CudaTensor')
        torch_outputs = torch_net:forward(torch_inputs)
    end

    print("compare...7")
    if type(torch_outputs) == "table" then
        for i=1,#torch_outputs do
            torch_outputs[i] = torch_outputs[i]:float()
        end
    else
        torch_outputs = {torch_outputs:float()}
    end

    print("compare...8")
    if #caffe_outputs ~= #torch_outputs then
        --logging.errorf("Inconsistent output blobs: Caffe: %s, Torch: %s", #caffe_outputs, #torch_outputs)
        error(string.format("Inconsistent output blobs: Caffe: %s, Torch: %s", #caffe_outputs, #torch_outputs))
        error("Inconsistent output blobs")
    end

    print("compare...9")
    for i = 1,#caffe_outputs do
--        print(i)
        local torch_output = torch_outputs[i]
        local caffe_output = caffe_outputs[i]
        --logging.infof("Caffe norm: %s, Torch norm: %s", torch.norm(caffe_output), torch.norm(torch_output))
        print("caffe output**************")
        print(caffe_output)
        print("torch output**************")
        print(torch_output)
        print("Caffe norm: ", torch.norm(caffe_output), "  Torch norm: ", torch.norm(torch_output))
        if not caffe_output:isSameSizeAs(torch_output) then
            --logging.errorf("Inconsistent output size: Caffe: %s, Torch: %s", caffe_output:size(), torch_output:size())
            error(string.format("Inconsistent output size: Caffe: %s, Torch: %s", caffe_output:size(), torch_output:size()))
            error("Inconsistent output sizes")
        end

        --save torch output
--        print('Writing output image to ' .. opts.out_path)
--        local img_out = utils.median_filter(torch_output, 3)
--        image.save(opts.out_path, img_out)

        local max_absolute_error = (caffe_output - torch_output):abs():max()
        --logging.infof("Maximum difference between Caffe and Torch output: %s", max_absolute_error)
        print("Maximum difference between Caffe and Torch output: ", max_absolute_error)
        if (max_absolute_error > 0.001) then
            debug_nets(caffe_net, torch_net)
            if os.getenv('LUA_DEBUG_ON_ERROR') then
                require('fb.debugger').enter()
            end
            error("Error in conversion!")
        end
    end
    print("compare...10")
    if os.getenv('LUA_DEBUG_ON_ERROR') then
        require('fb.debugger').enter()
    end
end

function M.convert(opts, torch_net)
    assert(opts)
    assert(torch_net)
    -- torch_net:float() -- convert the model to a CPU-only model
    local net_builder = py.reval(t2c.initialize())
    local bottom_edges = py.eval(t2c.setup_inputs(opts, net_builder))
    local top_edges = py.eval(t2c.setup_outputs(opts, net_builder))
    torch_layers.add(net_builder, torch_net, bottom_edges, top_edges)
    t2c.finalize(opts, net_builder)
end

function M.forward(opts, torch_net)
    torch_net:apply(function(m) m:evaluate() end)
    print("forward...1")
    local inputs = {}
    for i=1,#opts.inputs do
        local input_spec = opts.inputs[i]
        local tensor
        if input_spec.tensor then
            tensor = input_spec.tensor
        else
            tensor = torch.rand(table.unpack(input_spec.input_dims)):float()
        end
        table.insert(inputs, {name=input_spec.name, tensor=tensor})
    end

    print("forward...2")
    -- Legacy code
    if opts.input_tensor then
        assert(inputs[1].name == "data")
        inputs[1].tensor = opts.input_tensor
    end

    print("forward...3")
    print(inputs)
    -- Torch multi-inputs take an ordered Table.
    local function inputs_to_torch_inputs(inputs, type)
        if #inputs == 1 then
            return inputs[1].tensor:type(type)
        end
        local tensors = {}
        for i=1,#inputs do
            table.insert(tensors, inputs[i].tensor:type(type))
        end
        return tensors
    end
    local torch_outputs
    print("forward...4")
    -- Some networks only accept CUDA input.
    local ok, err = pcall(function()
            torch_net:float()
            local torch_inputs = inputs_to_torch_inputs(
                inputs, 'torch.FloatTensor')
            torch_outputs = torch_net:forward(torch_inputs)
    end)
    if ok then
        print("forward end...")
    else
        print("forward fail...")
    end
end

function M.run(opts, torch_net)
    -- print(torch_net)

    if opts.forward > 0 then
        print("forward torch")
        M.forward(opts, torch_net)
    end
    M.convert(opts, torch_net)
    M.compare(opts, torch_net)
end

function M.printModule(model)
    if model.modules == nil then
        print("-------- : ", torch.typename(model))
        if torch.typename(model) == 'nn.SpatialConvolution' then
            print(model.weight[2][1][1][1])
            print(model.weight[1][2][1][1])
            -- print(model.weight)
            print(model.bias)
        end
    else
        local count = table.getn(model.modules)
        if count > 0 then
            local layertemp;
            for i=1, count do
                layertemp=model:get(i);
                M.printModule(layertemp)
            end
        else
            print("======== : ", torch.typename(model))
            if torch.typename(model) == 'nn.SpatialConvolution' then
                -- print(model.weight)
                print(model.bias)
                print(model.weight[2][1][1][1])
                print(model.weight[1][2][1][1])
            end
        end
    end
end

function M.test(opts, torch_net)
    if opts.test then
        torch_net:apply(function(m) m:evaluate() end)
        M.printModule(torch_net)
    end
    return M.compare(opts, torch_net)
end

function M.main(opts)
    --logging.infof("Opts: %s", pl.pretty.write(opts))
    print(("Opts: %s"):format(pl.pretty.write(opts)))
    if opts.input_tensor ~= "" then
        opts.input_tensor = torch.load(opts.input_tensor)
    else
        opts.input_tensor = nil
    end

    -- Initialize fbcunn, fbnn, random includes, additions to
    -- t2c.CONVERTER, etc

    local model
    if opts.format == "lua" then
        model = assert(torch.load(opts.input))
    elseif opts.format == "luathrift" then
        local f = assert(io.open(opts.input))
        local thrift = require 'fb.thrift'
        model = thrift.from_file(f)
    end

    if opts.preprocessing and opts.preprocessing ~= "" then
        paths.dofile(opts.preprocessing)
    end
    if g_t2c_preprocess then
        model = g_t2c_preprocess(model, opts)
    end

    opts.imgpath = "chicago.jpg"
    opts.out_path = "torch.jpg"
    opts.image_size = 256
    local dims = {"1", "3", "256", "256"}
    if not opts.inputs then
        -- opts.inputs = {{name="data", input_dims=opts.input_dims}}
        opts.inputs = {{name="data", input_dims=dims}}
    end

    --logging.infof("Parsed opts: %s", pl.pretty.write(opts))
    print(("Parsed opts: %s").format(pl.pretty.write(opts)))

    if opts.compare > 0 then
        -- opts.test = true
        return M.test(opts, model)
    else
        if opts.verify ~= "" then
            return M.compare(opts, model)
        else
            return M.run(opts, model)
        end
    end

end

return M
